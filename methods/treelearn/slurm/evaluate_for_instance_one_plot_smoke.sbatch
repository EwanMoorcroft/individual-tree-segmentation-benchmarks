#!/bin/bash -l
#SBATCH --job-name=treelearn_smoke_eval
#SBATCH --partition=nodes
#SBATCH --time=00:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --output=logs/treelearn_for_instance/treelearn_smoke_eval_%j.out
#SBATCH --error=logs/treelearn_for_instance/treelearn_smoke_eval_%j.err

set -euo pipefail

cd "$HOME/scratch/tree-seg-benchmark"
module purge
module load miniforge3/25.3.0-python3.12.10
source "$HOME/fastscratch/venvs/treebench/bin/activate"

TREELEARN_RUN_ID="${TREELEARN_RUN_ID:?TREELEARN_RUN_ID is required}"
TREELEARN_EXPECTED_BENCHMARK_COMMIT="${TREELEARN_EXPECTED_BENCHMARK_COMMIT:?TREELEARN_EXPECTED_BENCHMARK_COMMIT is required}"
SAFE_PLOT_ID="CULS_plot_1_annotated"
RELATIVE_PATH="CULS/plot_1_annotated.las"
PREDICTION_NPZ="data/predictions/treelearn/for_instance_smokes/$TREELEARN_RUN_ID/$SAFE_PLOT_ID/${SAFE_PLOT_ID}_treelearn_smoke_predictions.npz"
INFERENCE_METADATA="results/metadata/treelearn_for_instance/one_plot_smokes/$TREELEARN_RUN_ID/${SAFE_PLOT_ID}_inference.json"
TABLE_ROOT="results/tables/treelearn_for_instance/one_plot_smokes/$TREELEARN_RUN_ID"

test "$(git rev-parse HEAD)" = "$TREELEARN_EXPECTED_BENCHMARK_COMMIT"
test -z "$(git status --porcelain)"

test -f "$PREDICTION_NPZ"
test -f "$INFERENCE_METADATA"
if [[ -e "$TABLE_ROOT" ]]; then
  echo "Refusing to overwrite existing TreeLearn smoke evaluation: $TABLE_ROOT"
  exit 1
fi
mkdir -p "$(dirname "$TABLE_ROOT")"
PARTIAL_ROOT=$(mktemp -d "${TABLE_ROOT}.partial.${SLURM_JOB_ID:-manual}.XXXXXX")

python methods/treelearn/scripts/evaluate_for_instance_one_plot_smoke.py \
  --prediction-npz "$PREDICTION_NPZ" \
  --inference-metadata "$INFERENCE_METADATA" \
  --run-id "$TREELEARN_RUN_ID" \
  --plot-id "CULS/plot_1_annotated" \
  --relative-path "$RELATIVE_PATH" \
  --split dev \
  --metrics-json "$PARTIAL_ROOT/metrics.json" \
  --harmonized-matches-csv "$PARTIAL_ROOT/matches.csv" \
  --unmatched-predictions-csv "$PARTIAL_ROOT/unmatched_predictions.csv" \
  --unmatched-references-csv "$PARTIAL_ROOT/unmatched_references.csv"

test -f "$PARTIAL_ROOT/metrics.json"
test -f "$PARTIAL_ROOT/matches.csv"
test -f "$PARTIAL_ROOT/unmatched_predictions.csv"
test -f "$PARTIAL_ROOT/unmatched_references.csv"
if [[ -e "$TABLE_ROOT" ]]; then
  echo "TreeLearn smoke evaluation appeared concurrently: $TABLE_ROOT"
  exit 1
fi
mv "$PARTIAL_ROOT" "$TABLE_ROOT"

echo "treelearn_run_id=$TREELEARN_RUN_ID"
echo "treelearn_metrics=$TABLE_ROOT/metrics.json"
echo "status=treelearn-development-smoke-evaluated"
echo "next_gate=manual-alignment-review-before-full-development-evaluation"
