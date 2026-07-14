#!/bin/bash -l
#SBATCH --job-name=treelearn_long_gate
#SBATCH --partition=nodes
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=logs/treelearn_for_instance/treelearn_long_gate_%j.out
#SBATCH --error=logs/treelearn_for_instance/treelearn_long_gate_%j.err
set -euo pipefail
TREELEARN_PROJECT_ROOT="${TREELEARN_PROJECT_ROOT:-$HOME/scratch/tree-seg-benchmark}"
cd "$TREELEARN_PROJECT_ROOT"
test "${TREELEARN_FINETUNE_LONG_CONFIRMED:-0}" = 1
test "$(git rev-parse HEAD)" = "$TREELEARN_EXPECTED_BENCHMARK_COMMIT"
test -z "$(git status --porcelain)"
source "$HOME/fastscratch/venvs/treebench/bin/activate"
python methods/treelearn/scripts/verify_for_instance_finetune_long_contract.py \
  --freeze "$TREELEARN_LONG_FREEZE" \
  --treelearn-repo "$TREELEARN_REPO"
python methods/treelearn/scripts/retain_for_instance_finetune_long_selection.py \
  --long-freeze "$TREELEARN_LONG_FREEZE" \
  --crop-inventory "$TREELEARN_LONG_ROOT/crop_inventory.json" \
  --selection-freeze "$TREELEARN_LONG_SELECTION_FREEZE" \
  --output-root "$TREELEARN_LONG_DURABLE_ROOT" \
  --checkpoint-output "$TREELEARN_LONG_DURABLE_CHECKPOINT" \
  --manifest-output "$TREELEARN_LONG_RETENTION_MANIFEST"
python methods/treelearn/scripts/freeze_for_instance_finetune_long_selected.py \
  --selection-freeze "$TREELEARN_LONG_SELECTION_FREEZE" \
  --retention-manifest "$TREELEARN_LONG_RETENTION_MANIFEST" \
  --output "$TREELEARN_LONG_SELECTED_FREEZE"
echo status=long-selected-checkpoint-frozen-before-test
echo "next_gate=manual_review_before_any_held_out_test_submission"
echo "No held-out test job was submitted."
