#!/bin/bash -l
#SBATCH --job-name=treelearn_long_select
#SBATCH --partition=nodes
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=logs/treelearn_for_instance/treelearn_long_select_%j.out
#SBATCH --error=logs/treelearn_for_instance/treelearn_long_select_%j.err
set -euo pipefail
cd "$HOME/scratch/tree-seg-benchmark"
test "${TREELEARN_FINETUNE_LONG_CONFIRMED:-0}" = 1
test "$(git rev-parse HEAD)" = "$TREELEARN_EXPECTED_BENCHMARK_COMMIT"
test -z "$(git status --porcelain)"
source "$HOME/fastscratch/venvs/treebench/bin/activate"
python methods/treelearn/scripts/verify_for_instance_finetune_long_contract.py \
  --freeze "$TREELEARN_LONG_FREEZE" \
  --treelearn-repo "$TREELEARN_REPO"
python methods/treelearn/scripts/select_for_instance_finetune_long.py \
  --freeze "$TREELEARN_LONG_FREEZE" \
  --development-manifest "$TREELEARN_LONG_DEV_MANIFEST" \
  --metadata-base "$TREELEARN_LONG_METADATA_BASE" \
  --tables-base "$TREELEARN_LONG_TABLES_BASE" \
  --selection-root "$TREELEARN_LONG_ROOT/selection"
echo status=long-validation-selected-at-common-epoch-35
echo held_out_test_accessed=false
