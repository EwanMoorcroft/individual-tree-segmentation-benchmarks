#!/bin/bash -l
#SBATCH --job-name=tls2trees_retained_summary
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
test "${TLS2TREES_FINAL_EVALUATION_CONFIRMED:-0}" = "1"
SUMMARY_BENCHMARK_COMMIT="${TLS2TREES_FINAL_EVALUATION_SUMMARY_BENCHMARK_COMMIT:-$TLS2TREES_FINAL_EVALUATION_BENCHMARK_COMMIT}"
test "$(git rev-parse HEAD)" = "$SUMMARY_BENCHMARK_COMMIT"
test -z "$(git status --porcelain)"
test "$(sha256sum "$TLS2TREES_FINAL_EVALUATION_PLAN_JSON" | awk '{print $1}')" = \
  "$TLS2TREES_FINAL_EVALUATION_PLAN_SHA256"
test "$(sha256sum "$TLS2TREES_FINAL_EVALUATION_CLI" | awk '{print $1}')" = \
  "$TLS2TREES_FINAL_EVALUATION_CLI_SHA256"
test "$(sha256sum "$TLS2TREES_FINAL_EVALUATION_EVALUATOR" | awk '{print $1}')" = \
  "$TLS2TREES_FINAL_EVALUATION_EVALUATOR_SHA256"

module purge
module load miniforge3/25.3.0-python3.12.10
# shellcheck disable=SC1090
source "$TLS2TREES_FINAL_EVALUATION_TREEBENCH_ENV/bin/activate"
export PYTHONNOUSERSITE=1

RESUME_ARGS=()
if [[ "${TLS2TREES_FINAL_EVALUATION_RESUME_EXACT_PARTIAL:-0}" = "1" ]]; then
  RESUME_ARGS+=(--resume-exact-partial)
fi

python "$TLS2TREES_FINAL_EVALUATION_CLI" summarise \
  --plan-json "$TLS2TREES_FINAL_EVALUATION_PLAN_JSON" \
  --plan-sha256 "$TLS2TREES_FINAL_EVALUATION_PLAN_SHA256" \
  --output-json "$TLS2TREES_FINAL_EVALUATION_SUMMARY_JSON" \
  --plot-csv "$TLS2TREES_FINAL_EVALUATION_PLOT_CSV" \
  --aggregate-csv "$TLS2TREES_FINAL_EVALUATION_AGGREGATE_CSV" \
  --test-output-json "$TLS2TREES_FINAL_EVALUATION_TEST_SUMMARY_JSON" \
  --test-plot-csv "$TLS2TREES_FINAL_EVALUATION_TEST_PLOT_CSV" \
  --test-aggregate-csv "$TLS2TREES_FINAL_EVALUATION_TEST_AGGREGATE_CSV" \
  "${RESUME_ARGS[@]}"
