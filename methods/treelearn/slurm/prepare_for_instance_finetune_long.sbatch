#!/bin/bash -l
#SBATCH --job-name=treelearn_long_prep
#SBATCH --partition=nodes
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=logs/treelearn_for_instance/treelearn_long_prep_%j.out
#SBATCH --error=logs/treelearn_for_instance/treelearn_long_prep_%j.err
set -euo pipefail
cd "$HOME/scratch/tree-seg-benchmark"
test "${TREELEARN_FINETUNE_LONG_CONFIRMED:-0}" = 1
test "$(git rev-parse HEAD)" = "$TREELEARN_EXPECTED_BENCHMARK_COMMIT"
test -z "$(git status --porcelain)"
test "$(git -C "$TREELEARN_REPO" rev-parse HEAD)" = fd240ce7caa4c444fe3418aca454dc578bc557d4
test -z "$(git -C "$TREELEARN_REPO" status --porcelain)"
module purge
module load miniforge3/25.3.0-python3.12.10
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$TREELEARN_ENV"
python methods/treelearn/scripts/prepare_for_instance_finetune_long.py \
  --development-manifest "$TREELEARN_DEV_MANIFEST_JSON" \
  --run-root "$TREELEARN_LONG_ROOT" \
  --treelearn-repo "$TREELEARN_REPO" \
  --checkpoint "$TREELEARN_LONG_INITIAL_CHECKPOINT" \
  --evaluation-template methods/treelearn/configs/for_instance_development.yml \
  --crops-per-plot "${TREELEARN_LONG_CROPS_PER_PLOT:-32}"
test -f "$TREELEARN_LONG_FREEZE"
echo status=treelearn-long-development-prepared
echo held_out_test_accessed=false
