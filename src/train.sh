#!/usr/bin/env bash

# Basic settings
__NUM_GPU=1
__CONFIG_FILE="./config_files/B01-MLP-768-125M.yaml"
__PORT=$((10000 + RANDOM % 10000))

# expert CUDA_VISIBLE_DEVICES=0,1,2,3
# For bitsandbytes
# export BNB_CUDA_VERSION=129
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"

# Start the run
accelerate launch \
  --main_process_port "${__PORT}" \
  --config_file "./config_files/accelerate/gpu${__NUM_GPU}.yaml" \
  train.py \
  --config_file "${__CONFIG_FILE}"
