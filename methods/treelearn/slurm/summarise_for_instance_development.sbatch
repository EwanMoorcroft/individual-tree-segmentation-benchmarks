#!/bin/bash -l
#SBATCH --job-name=treelearn_dev_summary
#SBATCH --partition=nodes
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=logs/treelearn_for_instance/treelearn_dev_summary_%j.out
#SBATCH --error=logs/treelearn_for_instance/treelearn_dev_summary_%j.err

set -euo pipefail

cd "$HOME/scratch/tree-seg-benchmark"
mkdir -p logs/treelearn_for_instance
module purge
module load miniforge3/25.3.0-python3.12.10
source "$HOME/fastscratch/venvs/treebench/bin/activate"

TREELEARN_DEV_RUN_ID="${TREELEARN_DEV_RUN_ID:?TREELEARN_DEV_RUN_ID is required}"
TREELEARN_DEV_MANIFEST="${TREELEARN_DEV_MANIFEST:?TREELEARN_DEV_MANIFEST is required}"
TREELEARN_EXPECTED_BENCHMARK_COMMIT="${TREELEARN_EXPECTED_BENCHMARK_COMMIT:?TREELEARN_EXPECTED_BENCHMARK_COMMIT is required}"
METADATA_ROOT="results/metadata/treelearn_for_instance/development_runs/$TREELEARN_DEV_RUN_ID"
TABLE_ROOT="results/tables/treelearn_for_instance/development_runs/$TREELEARN_DEV_RUN_ID"

test "$(git rev-parse HEAD)" = "$TREELEARN_EXPECTED_BENCHMARK_COMMIT"
test -z "$(git status --porcelain)"
test -f "$TREELEARN_DEV_MANIFEST"
test -d "$METADATA_ROOT"
mkdir -p "$TABLE_ROOT/per_plot"

python methods/treelearn/scripts/summarise_for_instance_development.py \
  --manifest "$TREELEARN_DEV_MANIFEST" \
  --run-id "$TREELEARN_DEV_RUN_ID" \
  --evaluation-root "$TABLE_ROOT/per_plot" \
  --metadata-root "$METADATA_ROOT" \
  --output-root "$TABLE_ROOT" \
  --expected-benchmark-commit "$TREELEARN_EXPECTED_BENCHMARK_COMMIT"

echo "run_id=$TREELEARN_DEV_RUN_ID"
echo "run_summary=$TABLE_ROOT/run_summary.json"
echo "site_summary=$TABLE_ROOT/site_summary.csv"
echo "status=treelearn-development-accounted-and-summarised"
echo "held_out_test_accessed=false"
