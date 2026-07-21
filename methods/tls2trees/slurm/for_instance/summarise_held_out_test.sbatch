#!/bin/bash -l
#SBATCH --job-name=tls2trees_dt_test_summary
#SBATCH --partition=nodes
#SBATCH --time=00:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --no-requeue
#SBATCH --output=logs/tls2trees_for_instance/%x_%j.out
#SBATCH --error=logs/tls2trees_for_instance/%x_%j.err

set -euo pipefail
cd "${TLS2TREES_PROJECT_ROOT:-$HOME/scratch/tree-seg-benchmark}"
test "${TLS2TREES_HELD_OUT_TEST_CONFIRMED:-0}" = "1"
test "$(git rev-parse HEAD)" = "$TLS2TREES_EXPECTED_BENCHMARK_COMMIT"
test -z "$(git status --porcelain)"
test "$(sha256sum "$TLS2TREES_MANIFEST_JSON" | awk '{print $1}')" = \
  "$(awk '{print $1}' "$TLS2TREES_MANIFEST_SHA256_FILE")"
test "$(sha256sum "$TLS2TREES_FINAL_SELECTION_JSON" | awk '{print $1}')" = "$TLS2TREES_FINAL_SELECTION_SHA256"
module purge
module load miniforge3/25.3.0-python3.12.10
# shellcheck disable=SC1090
source "$TLS2TREES_TREEBENCH_ENV/bin/activate"
export PYTHONNOUSERSITE=1
python methods/tls2trees/scripts/evaluation/summarise_tls2trees_held_out_test.py \
  --output-root "$TLS2TREES_OUTPUT_ROOT" --workflow-run-id "$TLS2TREES_TEST_RUN_ID" \
  --manifest-json "$TLS2TREES_MANIFEST_JSON" \
  --final-selection-json "$TLS2TREES_FINAL_SELECTION_JSON" \
  --final-selection-sha256 "$TLS2TREES_FINAL_SELECTION_SHA256" \
  --output-json "$TLS2TREES_TEST_SUMMARY_JSON" \
  --plot-csv "$TLS2TREES_TEST_PLOT_CSV" \
  --aggregate-csv "$TLS2TREES_TEST_AGGREGATE_CSV"
