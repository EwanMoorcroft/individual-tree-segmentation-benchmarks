#!/bin/bash -l
#SBATCH --job-name=treelearn_ft_val_sum
#SBATCH --partition=nodes
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=logs/treelearn_for_instance/treelearn_ft_val_sum_%j.out
#SBATCH --error=logs/treelearn_for_instance/treelearn_ft_val_sum_%j.err
set -euo pipefail
cd "$HOME/scratch/tree-seg-benchmark"
test "$(git rev-parse HEAD)" = "$TREELEARN_EXPECTED_BENCHMARK_COMMIT"
test -z "$(git status --porcelain)"
source "$HOME/fastscratch/venvs/treebench/bin/activate"
python methods/treelearn/scripts/summarise_for_instance_finetune_validation.py \
  --freeze "$TREELEARN_FINETUNE_FREEZE" \
  --run-id "$TREELEARN_FINETUNE_RUN_ID" \
  --metadata-root "$TREELEARN_FINETUNE_METADATA_ROOT" \
  --table-root "$TREELEARN_FINETUNE_TABLE_ROOT" \
  --checkpoint-sha256 "$TREELEARN_FINETUNE_CHECKPOINT_SHA256"
echo status=fine-tuned-development-validation-completed
echo held_out_test_accessed=false
