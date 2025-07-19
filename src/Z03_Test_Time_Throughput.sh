export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
export CUDA_VISIBLE_DEVICES=0

pairs=(
  "B B01-MLP-768-125M"
  "B B02-MLP-1024-350M"
  "B B03-MLP-1536-760M"
  "B B04-MLP-2048-1.3B"
  "P P01-MLP-768-360M"
  "P P02-MLP-768-760M"
  "P P03-MLP-768-1.4B"
  "M M01-SMoE-768-125M"
  "M M02-SMoE-768-360M"
  "M M03-SMoE-768-760M"
  "M M04-SMoE-768-1.4B"
  "U U01-UMoE-768-125M"
  "U U02-UMoE-768-360M"
  "U U03-UMoE-768-760M"
  "U U04-UMoE-768-1.4B"
  "G G01-SMoE-768-1.3B-G1-ReLU"
  "G G02-SMoE-768-1.3B-G2-ReLU"
  "G G03-SMoE-768-1.3B-G4-ReLU"
  "E E05-SMoE-1024-2.2B-G2-Sigmoid-ReLU"
  "A A06-UMoE-768-1.4B-ReLU"
  "A A08-SMoE-768-1.4B-Softmax"
)

for item in "${pairs[@]}"; do
  read cls name <<< "$item"
  echo "Running: $cls $name"
  python Z03_Test_Time_Throughput.py \
    --config_file "./config_files/final/${name}.yaml" \
    --path "./final_checkpoints/${cls}/${name}/checkpoints/checkpoint_last_15027.pt" \
    --max_steps 10000
done