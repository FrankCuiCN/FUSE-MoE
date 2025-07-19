import torch
from tqdm import tqdm
import argparse
from model.model import Model
from config.get_config import get_config
from utils.profiler import Profiler
from utils.set_random_seeds import set_random_seeds
from utils.get_project_directories import get_project_directories

import os
import json
# Workaround: To suppress a huggingface warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ----- #
# PyTorch Settings
# ----- #
torch.set_float32_matmul_precision("high")
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.enabled = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True
# ----- #

# ----- #
# Preparation
# ----- #
# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument(
    "--config_file", type=str, required=True,
    help="Path to configuration file"
)
parser.add_argument(
    "-p",
    "--path",
    type=str,
    required=True,
    help="The path of the saved checkpoint file",
)
args = parser.parse_args()
# Load data
print("Loading config file from", args.config_file)
config = get_config(args.config_file)
# Only the main process creates the directories
SUB_DIRS = []
RESULT_PATH, _ = get_project_directories(
    "2a - Evaluation", config, True, SUB_DIRS
)
# Ensure all processes wait until the directories are created
RESULT_PATH, _ = get_project_directories(
    "2a - Evaluation", config, False, SUB_DIRS
)
# Set random seed if enabled
if config.repro_use_random_seed:
    set_random_seeds(config)
# Get the checkpoint data
print("Loading model checkpoint from {}".format(args.path))
checkpoint = torch.load(args.path, map_location=torch.device("cuda"))
# ----- #


# ----- #
# Profiler
# ----- #
config.runtime["profiler"] = Profiler(enabled=False)
# ----- #


# ----- #
# Model
# ----- #
# Get block_mask_info_attn
from model.ops.unified_computation import get_block_mask_info_attn
assert config.context_window % config.perf_flex_attn_block_size == 0
config.runtime["block_mask_info_attn"] = get_block_mask_info_attn(
    config.batch_size_fwd,
    1,
    config.context_window,
    config.perf_flex_attn_block_size,
    device="cuda",
)
model = Model(config).cuda()
# Load from checkpoint
new_checkpoint = {}
for k, v in checkpoint["model_state_dict"].items():
    k = k.replace("_orig_mod.", "")
    k = k.replace(".ffwd.fc_1.", ".ffwd.fc_in.")
    k = k.replace(".ffwd.fc_2.", ".ffwd.fc_out.")
    new_checkpoint[k] = v
load_state_dict_result = model.load_state_dict(new_checkpoint, strict=False)
print("Missing keys:", load_state_dict_result.missing_keys)
print("Unexpected keys:", load_state_dict_result.unexpected_keys)
# Compile the model
if config.perf_use_compile:
    model = torch.compile(model, mode=config.perf_compile_mode)
model.eval()
# ----- #


# ----- #
# Data
# ----- #
from data.get_dataloader import get_dataloader_val
dataloader_val = get_dataloader_val(config)
len_dataloader_val = len(dataloader_val)
# ----- #


early_stopping = False
loss_avg = torch.zeros(1, dtype=torch.float32, device="cuda")
device   = "cuda"
use_autocast = config.perf_use_autocast
token_count_per_node = 0
token_limit_per_node = 5_242_880
# Set the model to evaluation mode
model.eval()
for idx_iter, (inputs, targets) in enumerate(tqdm(dataloader_val, desc="Validating model", total=len_dataloader_val,)):
    # Ask: Should we use non_blocking?
    inputs  =  inputs.to(device=device, non_blocking=True)
    targets = targets.to(device=device, non_blocking=True)
    # Forward pass
    # Ask: Should we use model.no_sync() here?
    with torch.no_grad():
        with torch.autocast("cuda", torch.bfloat16, enabled=use_autocast):
            loss, telemetry_local = model(inputs, targets)
            loss_avg += telemetry_local["loss_lm"]
    # Early stopping
    token_count_per_node += inputs.numel()
    if early_stopping:
        if token_count_per_node >= token_limit_per_node:
            break
# Normalize loss_avg
loss_avg = loss_avg / (idx_iter + 1)

out = loss_avg.exp().item()

print(out)

os.makedirs("Perplexity FineWeb Results", exist_ok=True)
out_path = os.path.join("Perplexity FineWeb Results", config.misc_run_name + ".json")

# Create a proper dictionary for JSON output
json_data = {"fineweb_perplexity": out}

with open(out_path, "w") as f:
    json.dump(json_data, f, indent=2)
