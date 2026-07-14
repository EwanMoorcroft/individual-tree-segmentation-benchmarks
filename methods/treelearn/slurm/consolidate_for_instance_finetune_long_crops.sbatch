#!/bin/bash -l
#SBATCH --job-name=treelearn_long_merge
#SBATCH --partition=nodes
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=logs/treelearn_for_instance/treelearn_long_merge_%j.out
#SBATCH --error=logs/treelearn_for_instance/treelearn_long_merge_%j.err
set -euo pipefail
cd "$HOME/scratch/tree-seg-benchmark"
test "${TREELEARN_FINETUNE_LONG_CONFIRMED:-0}" = 1
test "$(git rev-parse HEAD)" = "$TREELEARN_EXPECTED_BENCHMARK_COMMIT"
test -z "$(git status --porcelain)"
module purge
module load miniforge3/25.3.0-python3.12.10
source "$HOME/fastscratch/venvs/treebench/bin/activate"
python methods/treelearn/scripts/verify_for_instance_finetune_long_contract.py \
  --freeze "$TREELEARN_LONG_FREEZE" \
  --treelearn-repo "$TREELEARN_REPO"
python methods/treelearn/scripts/consolidate_for_instance_finetune_long_crops.py \
  --freeze "$TREELEARN_LONG_FREEZE" \
  --output "$TREELEARN_LONG_ROOT/crop_inventory.json"
echo status=treelearn-long-crops-consolidated
echo held_out_test_accessed=false
