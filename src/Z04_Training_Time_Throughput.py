import os, json, time, argparse, torch
from tqdm import tqdm
from model.model import Model
from config.get_config import get_config
from utils.profiler import Profiler
from utils.set_random_seeds import set_random_seeds
from data.get_dataloader import get_dataloader_train

# Pytorch settings
torch.set_float32_matmul_precision("high")
torch.backends.cuda.matmul.allow_tf32  = True
torch.backends.cudnn.enabled           = True
torch.backends.cudnn.allow_tf32        = True
torch.backends.cudnn.benchmark         = True

# CLI
parser = argparse.ArgumentParser()
parser.add_argument("--config_file", type=str, required=True)
parser.add_argument("-p", "--path", type=str, help="Checkpoint path (model only)")
parser.add_argument("--max_steps", type=int, default=200)
args = parser.parse_args()

# Config
print("Loading config file from", args.config_file)
config = get_config(args.config_file)
# Config Override
config.data_name = "FineWebEdu1B"
config.perf_use_8bit_adamw = True
print(config)

if config.repro_use_random_seed:
    set_random_seeds(config)

config.runtime["profiler"] = Profiler(enabled=config.perf_use_profiler)

# Model
from model.ops.unified_computation import get_block_mask_info_attn
assert config.context_window % config.perf_flex_attn_block_size == 0
config.runtime["block_mask_info_attn"] = get_block_mask_info_attn(
    config.batch_size_fwd, 1, config.context_window,
    config.perf_flex_attn_block_size, device="cuda"
)

model = Model(config).cuda()
from utils.report_parameter_count import report_parameter_count
total_trainable, total_non_embed = report_parameter_count(model)
total_dormant = 2 * config.num_block * config.ffwd_num_head * (config.ffwd_num_expert - config.ffwd_num_expert_active) * config.ffwd_expert_size * config.ffwd_head_size
model = model.cuda()

if args.path:
    print("Loading checkpoint from", args.path)
    ckpt = torch.load(args.path, map_location="cpu")
    new_state = {k.replace("_orig_mod.", "").replace(".ffwd.fc_1.", ".ffwd.fc_in.").replace(".ffwd.fc_2.", ".ffwd.fc_out."): v
                 for k, v in ckpt["model_state_dict"].items()}
    res = model.load_state_dict(new_state, strict=False)
    print("Missing keys:", res.missing_keys)
    print("Unexpected keys:", res.unexpected_keys)

if config.perf_use_compile:
    model = torch.compile(model, mode=config.perf_compile_mode)

# Data
dataloader_train = get_dataloader_train(config)

# Optimizer
if getattr(config, "perf_use_8bit_adamw", False):
    import bitsandbytes as bnb
    optimizer = bnb.optim.AdamW(
        params=model.parameters(),
        lr=0.0,
        betas=(config.adamw_beta_1, config.adamw_beta_2),
        eps=config.adamw_eps,
        optim_bits=8,
    )
else:
    optimizer = torch.optim.AdamW(
        params=model.parameters(),
        lr=0.0,
        betas=(config.adamw_beta_1, config.adamw_beta_2),
        eps=config.adamw_eps,
        fused=True,
    )

# Benchmark loop
iter_times, warmup_skip = [], 5
token_per_iter = config.batch_size_fwd * config.accu_steps * config.context_window
for idx_step, (inputs, targets) in enumerate(tqdm(dataloader_train, disable=False)):
    if idx_step >= args.max_steps:
        break
    
    model.train()
    inputs, targets = inputs.cuda(non_blocking=True), targets.cuda(non_blocking=True)
    
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    
    optimizer.zero_grad(set_to_none=True)
    
    loss_avg     = torch.zeros(1, dtype=torch.float32, device="cuda")
    accu_steps   = config.accu_steps
    device       = "cuda"
    profiler     = config.runtime["profiler"]
    use_autocast = config.perf_use_autocast
    
    inputs  = torch.chunk(inputs,  accu_steps, dim=0)
    targets = torch.chunk(targets, accu_steps, dim=0)
    for idx in range(accu_steps):
        inputs_current  = inputs[idx].to(device=device, non_blocking=True)
        targets_current = targets[idx].to(device=device, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_autocast):
            loss, telemetry_local = model(inputs_current, targets_current)
        loss.backward()
    optimizer.step()
    
    torch.cuda.synchronize()
    t2 = time.perf_counter()
    
    iter_times.append(t2 - t1)
    config.runtime["profiler"].print_summary()
    config.runtime["profiler"].reset()

print("Training run finished.")

# Stats & JSON + H100-hours estimate
usable = iter_times[warmup_skip:] if len(iter_times) > warmup_skip else iter_times
usable_sorted = sorted(usable)
lo, hi = int(0.10 * len(usable_sorted)), int(0.90 * len(usable_sorted))
trimmed = usable_sorted[lo:hi] or usable_sorted
avg_time = sum(trimmed) / len(trimmed)
throughput = token_per_iter / avg_time

print(f"Average iter time (p10-p90 trimmed): {avg_time * 1000:.2f} ms")
print(f"Throughput: {throughput:.1f} tokens / second")

# H100-hours estimate
total_tokens = 9_900_000_000  # 9,900M tokens
sec_one_gpu = total_tokens / throughput
hrs_one_gpu = sec_one_gpu / 3600
h100_hours = hrs_one_gpu

print(f"Training time (single H100): {hrs_one_gpu:.2f} h")

os.makedirs("Training Time Throughput", exist_ok=True)
out_path = os.path.join("Training Time Throughput", config.misc_run_name + ".json")
with open(out_path, "w") as f:
    json.dump({"h100_hours": h100_hours,
               "param_total_trainable": total_trainable,
               "param_total_non_embed": total_non_embed,
               "param_total_dormant": total_dormant}, f, indent=2)
