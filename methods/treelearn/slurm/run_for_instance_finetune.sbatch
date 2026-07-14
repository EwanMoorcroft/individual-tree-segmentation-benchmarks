#!/bin/bash -l
#SBATCH --job-name=treelearn_finetune
#SBATCH --partition=gpu-l40s-low
#SBATCH --time=23:30:00
#SBATCH --cpus-per-task=12
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --output=logs/treelearn_for_instance/treelearn_finetune_%j.out
#SBATCH --error=logs/treelearn_for_instance/treelearn_finetune_%j.err
set -euo pipefail
cd "$HOME/scratch/tree-seg-benchmark"
test "${TREELEARN_FINETUNE_DEV_CONFIRMED:-0}" = 1
test "$(git rev-parse HEAD)" = "$TREELEARN_EXPECTED_BENCHMARK_COMMIT"
test -z "$(git status --porcelain)"
MODE="${TREELEARN_FINETUNE_MODE:?}"
case "$MODE" in
  smoke) CONFIG=finetune_smoke.yaml ;;
  full) CONFIG=finetune_full.yaml ;;
  *) echo "invalid fine-tune mode: $MODE" >&2; exit 2 ;;
esac
module purge
module load miniforge3/25.3.0-python3.12.10
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$TREELEARN_ENV"
cd "$TREELEARN_FINETUNE_ROOT"
python "$TREELEARN_REPO/tools/training/train.py" \
  --config "$TREELEARN_FINETUNE_ROOT/configs/$CONFIG"
test -f "$TREELEARN_FINETUNE_ROOT/work_dirs/${CONFIG%.yaml}/epoch_${TREELEARN_EXPECTED_EPOCH}.pth"
echo "status=treelearn-finetune-$MODE-completed"
echo held_out_test_accessed=false
