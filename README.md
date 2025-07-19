# FUSE-MoE
This is the codebase for FUSE: Flexible and Unified Sparse mixture-of-Experts.

## System Requirements
```
Tested on:
- Rocky Linux 8.10
- Python 3.12.9
- CUDA 12.7
```

## Install Dependencies:
Step 1: Create and activate a virtual environment
- Install Miniconda (See [link](https://www.anaconda.com/docs/getting-started/miniconda/install))
- Run `conda create -n fuse python=3.12.9`
- Run `conda activate fuse`

Step 2: Run the command below
```
pip install \
yq==3.4.3 \
tqdm==4.67.1 \
numpy==2.3.1 \
PyYAML==6.0.2 \
pandas==2.3.1 \
pynvml==12.0.0 \
lm-eval==0.4.9 \
tiktoken==0.9.0 \
pydantic==2.11.7 \
gitingest==0.1.5 \
liger-kernel==0.6.0 \
transformers==4.53.2 \
accelerate==1.9.0
```

Step 3: Install and set up wandb
  - Run `pip install wandb==0.21.0`
  - Run `wandb login` and follow the prompt

Step 4: Install torch, torchvision, and torchaudio
  - Run `pip install --index-url https://download.pytorch.org/whl/cu126 torch==2.7.1+cu126 torchvision==0.22.1+cu126 torchaudio==2.7.1+cu126`
  - Verify that `torch.cuda.is_available()` is `True`

Step 5: Install torchtune
  - Make sure `torch==2.7.1+cu126` and `torchvision==0.22.1+cu126` have been installed
  - Run `pip install torchao==0.11.0 torchtune==0.6.1`

Step 6: (Optional) Install bitsandbytes:
  - Install bitsandbytes 0.46.0 (See [link](https://huggingface.co/docs/bitsandbytes/main/en/installation)) 
