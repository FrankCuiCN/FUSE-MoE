export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
export CUDA_VISIBLE_DEVICES=0

pairs=(
  "B B01-MLP-768-125M"
  "B B02-MLP-1024-350M"
  "B B03-MLP-1536-760M"
  "B B04-MLP-2048-1.3B"
  "P P01-MLP-768-360M"
)

for item in "${pairs[@]}"; do
  read cls name <<< "$item"
  echo "Running: $cls $name"
  python Z02_Perplexity_FineWeb.py \
    --config_file "./config_files/final/${name}.yaml" \
    --path "./final_checkpoints/${cls}/${name}/checkpoints/checkpoint_last_15027.pt"
done