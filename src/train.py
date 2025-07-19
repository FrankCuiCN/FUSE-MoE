import os
import gc
import time
import torch
import wandb
import shutil
import argparse
from tqdm import tqdm
from pathlib import Path
from accelerate import Accelerator, DataLoaderConfiguration
from config.get_config import get_config
from model.model import Model
from data.get_dataloader import get_dataloader_train, get_dataloader_val
from utils.profiler import Profiler
from utils.telemetry import Telemetry
from utils.synchronize import synchronize
from utils.set_random_seeds import set_random_seeds
from utils.to_cpu_recursive import to_cpu_recursive
from utils.report_parameter_count import report_parameter_count
from utils.get_project_directories import get_project_directories
from training_utils.validate import validate
from training_utils.save_checkpoint import save_checkpoint
from training_utils.get_learning_rate import get_learning_rate
from training_utils.gradient_accumulation import gradient_accumulation

# BUG: Known Issue: "MLP-2048 (1.3B)" + "max-autotune-no-cudagraphs" leads to NaN loss
# BUG: Known Issue: profiler context managers, even if disabled, still make forward passes slower (w/ torch.compile on)

# ----- #
# Accelerate
# ----- #
dataloader_config = DataLoaderConfiguration(
    split_batches=True, dispatch_batches=True
)
accelerator = Accelerator(dataloader_config=dataloader_config)
# Ask: should we set even_batches as True?
#     Currently, we enforce divisibility
# accelerator.even_batches(True)
# Workaround for the message:
# ...using GPU 3 to perform barrier as devices used by this process are
#     currently unknown. This can potentially cause a hang if this rank to
#     GPU mapping is incorrect
rank = accelerator.local_process_index
print(f"\nUSING GPU RANK: {rank}")
torch.cuda.set_device(rank)
torch.distributed.barrier(device_ids=[rank])  # debug: assuming 0,1,2,3 GPUs
# ----- #


# ----- #
# Config
# ----- #
# Parse command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument(
    "--config_file",
    type=str,
    help="Path to configuration file",
)
parser.add_argument(
    "--continue_path",
    type=str,
    help="Optional: Path to the continued run",
)
parser.add_argument(
    "--use_best_ckpt",
    action="store_true",
    help="Optional: Should you use the best or last checkpoint?",
)
args = parser.parse_args()
# Initialize config
config = get_config(args.config_file)
if accelerator.is_main_process:
    print()
    print("Loaded config from", args.config_file)
    print(config)
    print()
# Create runtime variables
config.runtime["config_file"]   = args.config_file
config.runtime["continue_path"] = args.continue_path
config.runtime["use_best_ckpt"] = args.use_best_ckpt
config.runtime["accelerator"]   = accelerator
config.runtime["telemetry"] = Telemetry(accelerator=accelerator)
# ----- #

# ----- #
# Self-Check
# ----- #
assert config.num_gpu == accelerator.num_processes
# ----- #

# ----- #
# Profiler
# ----- #
# Create the profiler
config.runtime["profiler"] = Profiler(enabled=accelerator.is_main_process if config.perf_use_profiler else False)
# ----- #

# ----- #
# Continuation Check and Handling
# Note: We are reading from the highest saved iteration downwards
#   so that we can essentially read from a" reserve pool"
# ----- #
if config.runtime["continue_path"]:
    if not os.path.isdir(config.runtime["continue_path"]):
        raise ValueError("ERR >> INVALID Continuation Path")
CHECKPOINT = {}
if config.runtime["use_best_ckpt"]:
    accelerator.print("USING BEST CHECKPOINT")
if config.runtime["continue_path"]:
    key_value = "last" if not config.runtime["use_best_ckpt"] else "best"
    directory = Path("{}/checkpoints/".format(config.runtime["continue_path"]))
    last_files = sorted(
        list(directory.glob(f"*{key_value}*")),
        key=lambda x: int(x.stem.split("_")[-1]),
        reverse=True,
    )
    CHECKPOINT = None
    for file in last_files:
        try:
            accelerator.print("Attempting to read checkpoint from", str(file))
            # Workaround: map_location="cpu" to fix duplication issues
            #   Otherwise all models are loaded to the main node (verify?)
            CHECKPOINT = torch.load(file, map_location="cpu")
            accelerator.print("Successfully loaded checkpoint from", str(file))
            accelerator.print(
                "Keys in the checkpoint:\n", str(CHECKPOINT.keys())
            )
            break
        except Exception as e:
            accelerator.print(f"Failed to load checkpoint from {file}: {e}")
            continue
    if CHECKPOINT is None:
        raise RuntimeError(
            "No valid checkpoint could be loaded from the available files."
        )
# ----- #


# ----- #
# Random seed
# ----- #
# Ask: (DDP) Should we use different seeds on different nodes?
if config.repro_use_random_seed:
    set_random_seeds(config)
# ----- #


# ----- #
# PyTorch
# ----- #
# Configure PyTorch performance settings
torch.set_float32_matmul_precision("high")
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.enabled = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True
# ----- #


# ----- #
# Project directory
# ----- #
SUB_DIRS = ["checkpoints"]
if not config.runtime["continue_path"]:
    if accelerator.is_main_process:
        RESULT_PATH, sub_dirs = get_project_directories(
            "1 - Trained Model", config, True, SUB_DIRS
        )
    # Ensure all processes wait until the directories are created
    synchronize()
    RESULT_PATH, sub_dirs = get_project_directories(
        "1 - Trained Model", config, False, SUB_DIRS
    )
    CKPT_PATH = sub_dirs[0]
else:
    RESULT_PATH = config.runtime["continue_path"]
    CKPT_PATH = os.path.join(RESULT_PATH, SUB_DIRS[0])
# Note: Also save the config file to our project directory
# Consider: Recreate a yaml file from "config" because config file may change
if accelerator.is_main_process and not config.runtime["continue_path"]:
    shutil.copy2(args.config_file, os.path.join(RESULT_PATH, Path(args.config_file).name))
# ----- #


# ----- #
# Wandb
# ----- #
if accelerator.is_main_process:
    # Read the old wandb_id for continuation
    wandb_id = None
    if config.runtime["continue_path"]:
        with open(f"{RESULT_PATH}/wandb/wand_job_id.txt", "r") as file:
            wandb_id = file.read().strip()
    # Initialize Wandb
    # Debug: This step may time out
    wandb.login()
    run = wandb.init(
        project=config.misc_project_name,
        dir=RESULT_PATH,
        name=config.misc_run_name,
        id=wandb_id,
        resume="must" if wandb_id else None,
    )
    # Record the job id, useful if we resumed training
    if not config.runtime["continue_path"]:
        with open(f"{RESULT_PATH}/wandb/wand_job_id.txt", "w") as file:
            file.write(wandb.run.id)
else:
    run = None
synchronize()
# ----- #


# ----- #
# Data
# ----- #
# Setup dataset and dataloader
if accelerator.is_main_process:
    dataloader_train = get_dataloader_train(config)
    dataloader_val = get_dataloader_val(config)
else:
    # Workaround for Accelerate; Used together with dispatch_batches
    # So that we load the dataset to RAM only once, as per
    # https://github.com/huggingface/accelerate/issues/3001
    dataloader_train = get_dataloader_train(config, dummy=True)
    dataloader_val = get_dataloader_val(config, dummy=True)
synchronize()
# ----- #


# ----- #
# Model
# ----- #
# Get block_mask_info_attn
# Debug: Consider put this in Model()
from model.ops.unified_computation import get_block_mask_info_attn
assert config.context_window % config.perf_flex_attn_block_size == 0
config.runtime["block_mask_info_attn"] = get_block_mask_info_attn(
    config.batch_size_fwd,
    1,
    config.context_window,
    config.perf_flex_attn_block_size,
    device=accelerator.device
)
# Build model
model = Model(config)
# Load weights to it if a checkpoint is provided
if CHECKPOINT:
    new_checkpoint = {}
    for key, value in CHECKPOINT["model_state_dict"].items():
        new_key = key.replace("_orig_mod.", "")
        new_checkpoint[new_key] = value
    load_state_dict_result = model.load_state_dict(new_checkpoint, strict=False)
    print("Missing keys:", load_state_dict_result.missing_keys)
    print("Unexpected keys:", load_state_dict_result.unexpected_keys)
    del new_checkpoint
    del CHECKPOINT["model_state_dict"]
    gc.collect()
    accelerator.print("Model Weights Loaded and Garbage Collected")
# Report parameter count
if accelerator.is_main_process:
    report_parameter_count(model)
    # Report dormant parameters
    # Debug: This is a temporary workaround
    print(
        "Dormant parameter count (if applicable):",
        2 * config.num_block * config.ffwd_num_head * (config.ffwd_num_expert - config.ffwd_num_expert_active) * config.ffwd_expert_size * config.ffwd_head_size
    )
# Retain the unwrapped model
unwrapped_model = model
# Send to the devices
model = model.to(accelerator.device)
# Compile the model
if config.perf_use_compile:
    # BUG: Currently, "cudagraphs" does not seem to work; Try to make it work
    model = torch.compile(model, mode=config.perf_compile_mode)
accelerator.print(model)
# ----- #


# ----- #
# Optimization
# ----- #
# Construct params_decay
params_decay = list()
for module in getattr(model, "_orig_mod", model).modules():
    if hasattr(module, "params_decay"):
        for p in module.params_decay:
            if p.requires_grad:  # Exclude frozen params
                params_decay.append(p)
# Construct params_no_decay
params_no_decay = list()
for name, p in getattr(model, "_orig_mod", model).named_parameters():
    if p.requires_grad:  # Exclude frozen params
        if not any(p is d for d in params_decay):
            params_no_decay.append(p)
            if accelerator.is_main_process:
                print(f"{name} is in params_no_decay")
# Define params
params = [
    {"params": params_decay,    "weight_decay": config.adamw_weight_decay},
    {"params": params_no_decay, "weight_decay": 0.0},
]
# Define optimizer
if config.perf_use_8bit_adamw:
    import bitsandbytes as bnb
    optimizer = bnb.optim.AdamW(
        params=params,
        betas=(config.adamw_beta_1, config.adamw_beta_2),
        eps=config.adamw_eps,
        optim_bits=8,
        min_8bit_size=50_000_000,
    )
else:
    optimizer = torch.optim.AdamW(
        params=params,
        betas=(config.adamw_beta_1, config.adamw_beta_2),
        eps=config.adamw_eps,
        fused=True,
    )
# Optionally load optimizer state
if CHECKPOINT:
    # Ask: How do we make sure the state_dict is strictly 1:1?
    # Analysis: If we pass the model loading check, presumably optim loading is also 1:1?
    optimizer.load_state_dict(CHECKPOINT["optimizer_state_dict"])
# ----- #


# ----- #
# Prepare with Accelerator
# ----- #
model, optimizer, dataloader_train, dataloader_val = accelerator.prepare(
    model, optimizer, dataloader_train, dataloader_val
)
# ----- #


# ----- #
# Broadcast len(dataloader_train) and len(dataloader_val) across nodes
# ----- #
if accelerator.is_main_process:
    len_dataloader_train = len(dataloader_train)
    len_dataloader_val   = len(dataloader_val)
else:
    len_dataloader_train = -1
    len_dataloader_val   = -1
# Assumption: "accelerator.gather" orders the values by rank
len_dataloader_train = accelerator.gather(torch.tensor(
    data=[len_dataloader_train], device=accelerator.device, dtype=torch.int32
))[0].item()
len_dataloader_val = accelerator.gather(torch.tensor(
    data=[len_dataloader_val],   device=accelerator.device, dtype=torch.int32
))[0].item()
# Validation
assert len_dataloader_train != -1
assert len_dataloader_val   != -1
# ----- #


# ----- #
# Training
# ----- #
best_loss = CHECKPOINT.get("best_loss", float("inf"))
# Compatibility issue: Some early version calls it "iteration"
# Below is a temporary workaround
prev_idx_iter = CHECKPOINT.get("idx_iter", CHECKPOINT.get("iteration", 0))
accelerator.print(f"Previous iter: {prev_idx_iter}")
for idx_iter, (inputs, targets) in enumerate(tqdm(
    iterable=dataloader_train,
    desc="Training model",
    total=len_dataloader_train,
    disable=not accelerator.is_main_process,
)):
    # ----- #
    # Workaround: Skip until the resumption point
    # ----- #
    if idx_iter < prev_idx_iter:
        continue
    # ----- #
    
    
    # ----- #
    # Reset wandb_log
    # ----- #
    wandb_log = {}
    wandb_log["step"] = idx_iter
    # ----- #
    
    # ----- #
    # Reset telemetry
    # ----- #
    config.runtime["telemetry"].reset()
    # ----- #
    
    
    # ----- #
    # Start the timer
    # ----- #
    synchronize()
    t1 = time.perf_counter()
    # ----- #
    
    
    # ----- #
    # Set the model to training mode
    # ----- #
    model.train()
    # ----- #
    
    
    # ----- #
    # Set learning rate
    # ----- #
    # Get learning rate
    lr = get_learning_rate(
        idx_iter=idx_iter,
        max_lr=config.lrsched_max_lr,
        min_lr=config.lrsched_min_lr,
        warmup_steps=config.lrsched_warmup_steps,
        decay_steps=config.lrsched_decay_steps,
        num_iter=len_dataloader_train,  # Use the broadcasted length
    )
    # Apply learning rate
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    # Update wandb_log
    wandb_log["lr"] = lr
    # ----- #
    
    # ----- #
    # Zero out gradients
    # ----- #
    optimizer.zero_grad(set_to_none=True)
    # ----- #
    
    # ----- #
    # Gradient accumulation
    # ----- #
    loss_train = gradient_accumulation(config, inputs, targets, model, accelerator)
    # Update wandb_log
    wandb_log["loss"] = loss_train  # Debug: Call it loss_lm_train
    # ----- #
    
    
    # ----- #
    # Gradient clipping
    # ----- #
    if config.gradclip_enabled:
        accelerator.clip_grad_norm_(
            parameters=model.parameters(),
            max_norm=config.gradclip_max_norm,
            norm_type=config.gradclip_norm_type,
        )
    # ----- #
    
    
    # ----- #
    # Update the parameters
    # ----- #
    optimizer.step()
    # ----- #
    
    
    # ----- #
    # Stop the timer
    # ----- #
    synchronize()
    t2 = time.perf_counter()
    # Get iter_time
    iter_time = t2 - t1
    # Update wandb_log
    wandb_log["iter_time"] = iter_time
    # ----- #
    
    
    # ----- #
    # Validation and Checkpointing
    # ----- #
    if (
        ((idx_iter % 100 == 0) and (idx_iter != 0)) or  # Is multiple of 100, excluding 0
        (idx_iter == (len_dataloader_train - 1))        # Is last iteration
    ):
        # Workaround: Avoid hiding the training progress bar
        if accelerator.is_main_process:
            print("\n\n")
        # Validation
        synchronize()
        loss_val = validate(early_stopping=True, payload=(config, model, dataloader_val, accelerator, len_dataloader_val))
        # Update wandb_log
        wandb_log["loss_val"] = loss_val
        # Synchronize before proceeding
        synchronize()
    
    if (
        ((idx_iter % 1000 == 0) and (idx_iter != 0)) or  # Is multiple of 1000, excluding 0
        (idx_iter == (len_dataloader_train - 1))         # Is last iteration
    ):
        synchronize()
        if accelerator.is_main_process:
            # Define checkpoint (cpu only)
            checkpoint = to_cpu_recursive(
                {
                    "lr": lr,
                    "idx_iter": idx_iter,
                    "best_loss": best_loss,
                    "model_state_dict": unwrapped_model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                }
            )
            # Save the checkpoint
            try:
                save_checkpoint(checkpoint, idx_iter, CKPT_PATH, save_best=False)
            except TimeoutError as e:
                print(f"Unexpected error: Saving timed out! {e}")
            # Release the memory (This is a workaround)
            del checkpoint
            gc.collect()
        # Synchronize before proceeding
        synchronize()
    # ----- #
    
    # ----- #
    # Telemetry handling
    # ----- #
    # Assumption: All tensors in telemetry are 0D tensors
    for key in config.runtime["telemetry"].keys():
        wandb_log[key] = config.runtime["telemetry"].get(key)
    # ----- #
    
    # ----- #
    # Submit wandb_log
    # ----- #
    # Consider: Periodic uploading to wandb server
    if accelerator.is_main_process:
        # Background: When resuming training, the actual training step at
        #     interruption can differ from the checkpointed step
        #     E.g.: wandb point is at 120, but resumption point is at 100
        # Workaround: Log future steps only
        # Assumption: Training is deterministic
        if idx_iter >= wandb.run._step:
            run.log(wandb_log, step=idx_iter)
    # ----- #
    
    
    # ----- #
    # Profiler handling
    # ----- #
    config.runtime["profiler"].print_summary()
    config.runtime["profiler"].reset()
    # ----- #
    
    
    # ----- #
    # Cleanup
    # ----- #
    # Ask: Should we synchronize here?
    synchronize()
    # ----- #


# ----- #
# Cleanup
# ----- #
# End the wandb run
if accelerator.is_main_process:
    run.finish()
# Ensure all processes finish
synchronize()
# ----- #
