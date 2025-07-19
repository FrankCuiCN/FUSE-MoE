import torch
import argparse
import lm_eval
from lm_eval import tasks, evaluator
from lm_eval.models.huggingface import HFLM
from transformers import AutoTokenizer
from hf_minimal_wrapper import MinimalGPTConfig, MinimalGPTWrapper
from model.model import Model
from config.get_config import get_config
from utils.profiler import Profiler
from utils.set_random_seeds import set_random_seeds
from utils.get_project_directories import get_project_directories

import os
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


# Build model and tokenizer
raw_model  = model
vocab_size = config.vocab_size
tokenizer  = AutoTokenizer.from_pretrained("gpt2")    # must match your vocab

hf_model = MinimalGPTWrapper(
    model=raw_model,
    config=MinimalGPTConfig(vocab_size=vocab_size),
)
# Wrap for lm_eval
lm = HFLM(
    pretrained=hf_model,
    tokenizer=tokenizer,
    backend="causal",
    device="cuda",
    batch_size=config.batch_size_fwd,
)
# Run evaluation
# Note: This is the single source of truth
task_dict = tasks.get_task_dict([
    "hellaswag",
    "piqa",
    "lambada_openai",
    "arc_easy",
    "arc_challenge",
])
results   = evaluator.evaluate(lm, task_dict)
print(results["results"])

import os, json
save_keys = {
    "arc_challenge": "acc,none",
    "arc_easy": "acc,none",
    "hellaswag": "acc,none",
    "lambada_openai": "acc,none",
    "piqa": "acc,none",
}
out = {}
for task, key in save_keys.items():
    v = results["results"][task].get(key, None)
    if v is not None:
        out[task] = float(v)
os.makedirs("Terminal Eval Results", exist_ok=True)
out_path = os.path.join("Terminal Eval Results", config.misc_run_name + ".json")
with open(out_path, "w") as f:
    json.dump(out, f, indent=2)
