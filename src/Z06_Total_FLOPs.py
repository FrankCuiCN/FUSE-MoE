import os, json, yaml
from pathlib import Path

pairs = [
  ("G", "G01-SMoE-768-1.3B-G1-ReLU"),
  ("G", "G02-SMoE-768-1.3B-G2-ReLU"),
  ("G", "G03-SMoE-768-1.3B-G4-ReLU"),
  ("E", "E05-SMoE-1024-2.2B-G2-Sigmoid-ReLU"),
  ("B", "B01-MLP-768-125M"),
  ("P", "P01-MLP-768-360M"),
  ("P", "P02-MLP-768-760M"),
  ("P", "P03-MLP-768-1.4B"),
  ("B", "B02-MLP-1024-350M"),
  ("B", "B04-MLP-2048-1.3B"),
  ("M", "M04-SMoE-768-1.4B"),
  ("B", "B03-MLP-1536-760M"),
  ("M", "M01-SMoE-768-125M"),
  ("M", "M02-SMoE-768-360M"),
  ("M", "M03-SMoE-768-760M"),
  ("U", "U01-UMoE-768-125M"),
  ("U", "U02-UMoE-768-360M"),
  ("U", "U03-UMoE-768-760M"),
  ("U", "U04-UMoE-768-1.4B"),
  ("A", "A06-UMoE-768-1.4B-ReLU"),
  ("A", "A08-SMoE-768-1.4B-Softmax"),
]
CONFIG_DIR = Path("./config_files")
OUT_DIR    = Path("Z06_Total_FLOPs")
OUT_DIR.mkdir(exist_ok=True)

def get_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def flops_from_cfg(cfg):
    cw      = cfg["context_window"]
    n_blk   = cfg["num_block"]
    d_model = cfg["emb_size"]
    d_ff    = cfg["ffwd_hid_size"]
    n_exp   = cfg["ffwd_num_expert"]
    n_act   = cfg["ffwd_num_expert_active"]
    d_exp   = cfg["ffwd_expert_size"]
    vocab   = cfg["vocab_size"]
    ffwd    = cfg["ffwd_name"]
    tokens  = 9_900_000_000

    sparse  = ffwd in {"SMoE", "UMoE", "Unified"}
    emb_fwd = 4 * d_model
    emb_bwd = 2 * emb_fwd
    att_fwd = 2*n_blk*d_model*d_model*3 + 2*n_blk*d_model*cw + 2*n_blk*d_model*d_model
    att_bwd = 2 * att_fwd
    if sparse:
        ff_fwd = (2*n_blk*d_model*d_model +
                  2*n_blk*2*d_model*d_exp*n_act +
                  2*n_blk*d_model*d_model+
                  2*n_blk*d_model*n_exp)
    else:
        ff_fwd = 2*n_blk*2*d_model*d_ff
    ff_bwd  = 2 * ff_fwd
    de_fwd  = 2 * d_model * vocab
    de_bwd  = 2 * de_fwd

    total   = tokens * (emb_fwd+emb_bwd+att_fwd+att_bwd+ff_fwd+ff_bwd+de_fwd+de_bwd)
    return f"{total:.2e}"

for cls, name in pairs:
    cfg_path = CONFIG_DIR / f"{name}.yaml"
    print(f"Loading {cfg_path}")
    cfg = get_config(cfg_path)
    total_flops = flops_from_cfg(cfg)
    run_name = cfg.get("mist_run_name", name)
    out_file = OUT_DIR / f"{run_name}.json"
    json.dump({"total_flops": total_flops}, open(out_file, "w"), indent=2)
    print(" →", out_file, total_flops)
