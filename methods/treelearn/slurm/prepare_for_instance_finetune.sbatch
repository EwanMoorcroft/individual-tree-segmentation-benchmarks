#!/bin/bash -l
#SBATCH --job-name=treelearn_ft_prep
#SBATCH --partition=gpu-l40s-low
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=12
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --output=logs/treelearn_for_instance/treelearn_ft_prep_%j.out
#SBATCH --error=logs/treelearn_for_instance/treelearn_ft_prep_%j.err
set -euo pipefail
cd "$HOME/scratch/tree-seg-benchmark"
test "${TREELEARN_FINETUNE_DEV_CONFIRMED:-0}" = 1
test "$(git rev-parse HEAD)" = "$TREELEARN_EXPECTED_BENCHMARK_COMMIT"
test -z "$(git status --porcelain)"
module purge
module load miniforge3/25.3.0-python3.12.10
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$TREELEARN_ENV"
python methods/treelearn/scripts/prepare_for_instance_finetune.py \
  --development-manifest "$TREELEARN_DEV_MANIFEST_JSON" \
  --run-root "$TREELEARN_FINETUNE_ROOT" \
  --treelearn-repo "$TREELEARN_REPO" \
  --checkpoint "$TREELEARN_CHECKPOINT" \
  --crop-count "$TREELEARN_FINETUNE_CROPS"
cd "$TREELEARN_REPO"
python tools/data_gen/gen_train_data.py \
  --config "$TREELEARN_FINETUNE_ROOT/configs/generate_crops.yaml"
test "$(find "$TREELEARN_FINETUNE_ROOT/data/train/random_crops/npz" -name '*.npz' | wc -l)" -ge 32
echo status=development-training-data-prepared
echo held_out_test_accessed=false
