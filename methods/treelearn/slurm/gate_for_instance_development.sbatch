#!/bin/bash -l
#SBATCH --job-name=treelearn_dev_gate
#SBATCH --partition=nodes
#SBATCH --time=00:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --output=logs/treelearn_for_instance/treelearn_dev_gate_%j.out
#SBATCH --error=logs/treelearn_for_instance/treelearn_dev_gate_%j.err

set -euo pipefail

cd "$HOME/scratch/tree-seg-benchmark"
mkdir -p logs/treelearn_for_instance
module purge
module load miniforge3/25.3.0-python3.12.10
source "$HOME/fastscratch/venvs/treebench/bin/activate"

TREELEARN_DEV_RUN_ID="${TREELEARN_DEV_RUN_ID:?TREELEARN_DEV_RUN_ID is required}"
TREELEARN_DEV_RUN_SUMMARY="${TREELEARN_DEV_RUN_SUMMARY:?TREELEARN_DEV_RUN_SUMMARY is required}"
TREELEARN_EXPECTED_BENCHMARK_COMMIT="${TREELEARN_EXPECTED_BENCHMARK_COMMIT:?TREELEARN_EXPECTED_BENCHMARK_COMMIT is required}"

test "$(git rev-parse HEAD)" = "$TREELEARN_EXPECTED_BENCHMARK_COMMIT"
test -z "$(git status --porcelain)"

python methods/treelearn/scripts/validate_for_instance_development_run.py \
  --run-summary "$TREELEARN_DEV_RUN_SUMMARY" \
  --expected-benchmark-commit "$TREELEARN_EXPECTED_BENCHMARK_COMMIT"

echo "run_id=$TREELEARN_DEV_RUN_ID"
echo "next_gate=manual_review_before_any_held_out_test_route"
echo "No held-out test job was submitted."
