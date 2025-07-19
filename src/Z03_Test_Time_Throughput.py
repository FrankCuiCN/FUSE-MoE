import os, json, time, argparse, torch
from tqdm import tqdm
from model.model import Model
from config.get_config import get_config
from utils.profiler import Profiler
from utils.set_random_seeds import set_random_seeds
from utils.get_project_directories import get_project_directories
from data.get_dataloader import get_dataloader_val

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ----- #
# PyTorch settings
# ----- #
torch.set_float32_matmul_precision("high")
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.enabled = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True
# ----- #

# ----- #
# CLI
# ----- #
parser = argparse.ArgumentParser()
parser.add_argument("--config_file", type=str, required=True)
parser.add_argument("-p", "--path", type=str, required=True, help="Checkpoint path")
parser.add_argument("--max_steps", type=int, default=200, help="Steps to benchmark")
args = parser.parse_args()
# ----- #

print("Loading config file from", args.config_file)
config = get_config(args.config_file)

# DEBUG overrides
config.num_gpu            = 1
config.batch_size_fwd     = 1
config.context_window     = 512
config.perf_use_compile   = True
config.perf_use_profiler  = False
# config.attn_name = "SelfAttnFlex"
print(config)

SUB_DIRS = []
get_project_directories("2a - Evaluation", config, True, SUB_DIRS)
get_project_directories("2a - Evaluation", config, False, SUB_DIRS)

if config.repro_use_random_seed:
    set_random_seeds(config)

print("Loading model checkpoint from", args.path)
checkpoint = torch.load(args.path, map_location=torch.device("cuda"))

# Profiler
config.runtime["profiler"] = Profiler(enabled=config.perf_use_profiler)

# Model
from model.ops.unified_computation import get_block_mask_info_attn
assert config.context_window % config.perf_flex_attn_block_size == 0
config.runtime["block_mask_info_attn"] = get_block_mask_info_attn(
    config.batch_size_fwd, 1, config.context_window,
    config.perf_flex_attn_block_size, device="cuda"
)
model = Model(config).cuda()
new_state = {}
for k, v in checkpoint["model_state_dict"].items():
    k = k.replace("_orig_mod.", "").replace(".ffwd.fc_1.", ".ffwd.fc_in.").replace(".ffwd.fc_2.", ".ffwd.fc_out.")
    new_state[k] = v
res = model.load_state_dict(new_state, strict=False)
print("Missing keys:", res.missing_keys)
print("Unexpected keys:", res.unexpected_keys)
if config.perf_use_compile:
    model = torch.compile(model, mode=config.perf_compile_mode)
model.eval()

# Data
dataloader_val = get_dataloader_val(config)

# Throughput measurement
iter_times, warmup_skip = [], 5
token_per_iter = config.batch_size_fwd * config.context_window
for step, (inputs, targets) in enumerate(tqdm(dataloader_val, disable=False)):
    if step >= args.max_steps:
        break
    inputs = inputs.to("cuda", non_blocking=True)
    torch.cuda.synchronize(); t1 = time.perf_counter()
    with torch.autocast("cuda", torch.bfloat16, enabled=config.perf_use_autocast):
        with torch.no_grad():
            _ = model(inputs, y=None)
    torch.cuda.synchronize(); t2 = time.perf_counter()
    iter_times.append(t2 - t1)

    config.runtime["profiler"].print_summary()
    config.runtime["profiler"].reset()

print("Training run finished.")

# Stats
usable = iter_times[warmup_skip:] if len(iter_times) > warmup_skip else iter_times
usable_sorted = sorted(usable)
lo = int(0.10 * len(usable_sorted))
hi = int(0.90 * len(usable_sorted))
trimmed = usable_sorted[lo:hi] or usable_sorted
avg_time = sum(trimmed) / len(trimmed)
throughput = 1 / avg_time  # Note: Doing 1 / avg_time to simulate next token prediction (even though we processed 512 tokens, we only generate 1 token)

print(f"Average iter time (p10-p90 trimmed): {avg_time*1000:.2f} ms")
print(f"Throughput: {throughput:.1f} tokens / second")

# Save JSON
os.makedirs("Test Time Throughput", exist_ok=True)
out_path = os.path.join("Test Time Throughput", config.misc_run_name + ".json")
with open(out_path, "w") as f:
    json.dump({"test_time_throughput": throughput}, f, indent=2)
