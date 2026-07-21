#!/bin/bash -l
#SBATCH --job-name=tls2trees_dt_finalise
#SBATCH --partition=nodes
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --no-requeue
#SBATCH --output=logs/tls2trees_for_instance/%x_%j.out
#SBATCH --error=logs/tls2trees_for_instance/%x_%j.err

set -euo pipefail
cd "${TLS2TREES_PROJECT_ROOT:-$HOME/scratch/tree-seg-benchmark}"
test "${TLS2TREES_FINALIZE_RESULTS_CONFIRMED:-0}" = "1"
test "$(git rev-parse HEAD)" = "$TLS2TREES_FINALIZE_BENCHMARK_COMMIT"
test -z "$(git status --porcelain)"

module purge
module load miniforge3/25.3.0-python3.12.10
# shellcheck disable=SC1090
source "$TLS2TREES_TEST_TREEBENCH_ENV/bin/activate"
export PYTHONNOUSERSITE=1

python methods/tls2trees/scripts/evaluation/finalise_tls2trees_held_out_results.py \
  --project-root "$PWD" \
  --summary-json "$TLS2TREES_TEST_SUMMARY_JSON" \
  --manifest-json "$TLS2TREES_TEST_MANIFEST_JSON" \
  --final-selection-json "$TLS2TREES_TEST_FINAL_SELECTION_JSON" \
  --final-selection-sha256 "$TLS2TREES_TEST_FINAL_SELECTION_SHA256" \
  --run-id "$TLS2TREES_TEST_RUN_ID" \
  --benchmark-commit "$TLS2TREES_FINALIZE_BENCHMARK_COMMIT" \
  --evaluation-benchmark-commit "$TLS2TREES_FINAL_EVALUATION_BENCHMARK_COMMIT" \
  --examples-dir methods/tls2trees/examples \
  --results-csv outputs/for_instance_benchmark_metrics/for_instance_method_benchmark_results.csv \
  --diagnostics-csv outputs/for_instance_benchmark_metrics/for_instance_method_development_diagnostics.csv \
  --retention-registry outputs/for_instance_benchmark_metrics/for_instance_prediction_retention_registry.csv \
  --receipt-json "$TLS2TREES_FINALIZE_RECEIPT_JSON"
