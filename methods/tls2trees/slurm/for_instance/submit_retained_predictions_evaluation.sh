#!/usr/bin/env bash

set -euo pipefail

if [[ "${TLS2TREES_FINAL_EVALUATION_CONFIRMED:-0}" != "1" ]]; then
  echo "Refusing immutable TLS2trees retained-prediction evaluation." >&2
  echo "Set TLS2TREES_FINAL_EVALUATION_CONFIRMED=1 after reviewing the frozen inputs." >&2
  exit 2
fi

STAGE2_STATE_FILE="${1:?Usage: submit_retained_predictions_evaluation.sh <completed-stage2-state-file> <completed-held-out-state-file>}"
TEST_STATE_FILE="${2:?Usage: submit_retained_predictions_evaluation.sh <completed-stage2-state-file> <completed-held-out-state-file>}"
test -f "$STAGE2_STATE_FILE"
test -f "$TEST_STATE_FILE"

# shellcheck disable=SC1090
source "$STAGE2_STATE_FILE"
STAGE2_RUN_ID="${TLS2TREES_STAGE2_RUN_ID:?Stage-2 state has no run ID}"
STAGE2_MANIFEST_JSON="${TLS2TREES_STAGE2_MANIFEST_JSON:?Stage-2 state has no manifest}"
STAGE2_SELECTION_JSON="${TLS2TREES_STAGE2_SELECTION_JSON:?Stage-2 state has no selection}"
STAGE2_OUTPUT_ROOT="${TLS2TREES_STAGE2_OUTPUT_ROOT:?Stage-2 state has no output root}"
TREEBENCH_ENV="${TLS2TREES_STAGE2_TREEBENCH_ENV:?Stage-2 state has no treebench environment}"

# shellcheck disable=SC1090
source "$TEST_STATE_FILE"
TEST_RUN_ID="${TLS2TREES_TEST_RUN_ID:?Held-out state has no run ID}"
TEST_MANIFEST_JSON="${TLS2TREES_TEST_MANIFEST_JSON:?Held-out state has no manifest}"
FINAL_SELECTION_JSON="${TLS2TREES_TEST_FINAL_SELECTION_JSON:?Held-out state has no frozen selection}"
FINAL_SELECTION_SHA256="${TLS2TREES_TEST_FINAL_SELECTION_SHA256:?Held-out state has no frozen-selection hash}"
TEST_OUTPUT_ROOT="${TLS2TREES_TEST_OUTPUT_ROOT:?Held-out state has no output root}"
test "$TLS2TREES_TEST_TREEBENCH_ENV" = "$TREEBENCH_ENV"

PROJECT_ROOT="${TLS2TREES_PROJECT_ROOT:-$HOME/scratch/tree-seg-benchmark}"
CLI="methods/tls2trees/scripts/evaluation/evaluate_retained_tls2trees_predictions.py"
EVALUATOR="methods/tls2trees/scripts/evaluation/evaluate_for_instance_tls2trees_plot.py"
TEST_RETENTION_MANIFEST="${TLS2TREES_FINAL_EVALUATION_RETENTION_MANIFEST:-methods/tls2trees/examples/tls2trees_development_tuned_prediction_retention_manifest.json}"

cd "$PROJECT_ROOT"
test -d .git
test -z "$(git status --porcelain)"
BENCHMARK_COMMIT=$(git rev-parse HEAD)
test -x "$TREEBENCH_ENV/bin/python"
for path in "$CLI" "$EVALUATOR" "$STAGE2_MANIFEST_JSON" \
  "$STAGE2_SELECTION_JSON" "$TEST_MANIFEST_JSON" "$FINAL_SELECTION_JSON" \
  "$TEST_RETENTION_MANIFEST"; do
  test -f "$path"
done
TEST_RETENTION_MANIFEST_SHA256=$(sha256sum "$TEST_RETENTION_MANIFEST" | awk '{print $1}')

STAMP="${TLS2TREES_FINAL_EVALUATION_STAMP:-$(date -u +%Y%m%d_%H%M%S)}"
if [[ ! "$STAMP" =~ ^[0-9]{8}_[0-9]{6}$ ]]; then
  echo "TLS2TREES_FINAL_EVALUATION_STAMP must use YYYYMMDD_HHMMSS." >&2
  exit 2
fi
RUN_ID="tls2trees_for-instance_retained_evaluation_$STAMP"
WORKFLOW_ROOT="$PROJECT_ROOT/results/metadata/tls2trees/for_instance/development_tuned/final_evaluation/$RUN_ID"
TABLE_ROOT="$PROJECT_ROOT/results/tables/tls2trees/for_instance/development_tuned/final_evaluation/$RUN_ID"
METRICS_ROOT="$WORKFLOW_ROOT/metrics"
PLAN_JSON="$WORKFLOW_ROOT/evaluation_plan.json"
SUMMARY_JSON="$WORKFLOW_ROOT/evaluation_summary.json"
PLOT_CSV="$TABLE_ROOT/plot_metrics.csv"
AGGREGATE_CSV="$TABLE_ROOT/aggregate_metrics.csv"
TEST_SUMMARY_JSON="$WORKFLOW_ROOT/held_out_test_summary.json"
TEST_PLOT_CSV="$TABLE_ROOT/held_out_test_plot_metrics.csv"
TEST_AGGREGATE_CSV="$TABLE_ROOT/held_out_test_target_summary.csv"
STATE_DIR="$HOME/fastscratch/tls2trees_for_instance_final_evaluation_states"
STATE_FILE="$STATE_DIR/$RUN_ID.env"
for path in "$WORKFLOW_ROOT" "$TABLE_ROOT" "$STATE_FILE"; do
  test ! -e "$path"
done
mkdir -p logs/tls2trees_for_instance "$STATE_DIR" "$WORKFLOW_ROOT" "$TABLE_ROOT"

"$TREEBENCH_ENV/bin/python" "$CLI" build-plan \
  --evaluation-run-id "$RUN_ID" \
  --benchmark-commit "$BENCHMARK_COMMIT" \
  --evaluator "$EVALUATOR" \
  --metrics-root "$METRICS_ROOT" \
  --development-output-root "$STAGE2_OUTPUT_ROOT" \
  --development-workflow-run-id "$STAGE2_RUN_ID" \
  --development-manifest-json "$STAGE2_MANIFEST_JSON" \
  --development-selection-json "$STAGE2_SELECTION_JSON" \
  --test-output-root "$TEST_OUTPUT_ROOT" \
  --test-workflow-run-id "$TEST_RUN_ID" \
  --test-manifest-json "$TEST_MANIFEST_JSON" \
  --final-selection-json "$FINAL_SELECTION_JSON" \
  --final-selection-sha256 "$FINAL_SELECTION_SHA256" \
  --test-retention-manifest-json "$TEST_RETENTION_MANIFEST" \
  --test-retention-manifest-sha256 "$TEST_RETENTION_MANIFEST_SHA256" \
  --output-plan-json "$PLAN_JSON"

PLAN_SHA256=$(sha256sum "$PLAN_JSON" | awk '{print $1}')
CLI_SHA256=$(sha256sum "$CLI" | awk '{print $1}')
EVALUATOR_SHA256=$(sha256sum "$EVALUATOR" | awk '{print $1}')
STAGE2_STATE_SHA256=$(sha256sum "$STAGE2_STATE_FILE" | awk '{print $1}')
TEST_STATE_SHA256=$(sha256sum "$TEST_STATE_FILE" | awk '{print $1}')

EVALUATION_JOB=not_submitted
SUMMARY_JOB=not_submitted
SUBMISSION_STATUS=plan_frozen_preflight_completed
write_state() {
  {
    printf 'TLS2TREES_FINAL_EVALUATION_RUN_ID=%q\n' "$RUN_ID"
    printf 'TLS2TREES_FINAL_EVALUATION_SUBMISSION_STATUS=%q\n' "$SUBMISSION_STATUS"
    printf 'TLS2TREES_FINAL_EVALUATION_JOB=%q\n' "$EVALUATION_JOB"
    printf 'TLS2TREES_FINAL_EVALUATION_SUMMARY_JOB=%q\n' "$SUMMARY_JOB"
    printf 'TLS2TREES_FINAL_EVALUATION_BENCHMARK_COMMIT=%q\n' "$BENCHMARK_COMMIT"
    printf 'TLS2TREES_FINAL_EVALUATION_PLAN_JSON=%q\n' "$PLAN_JSON"
    printf 'TLS2TREES_FINAL_EVALUATION_PLAN_SHA256=%q\n' "$PLAN_SHA256"
    printf 'TLS2TREES_FINAL_EVALUATION_CLI=%q\n' "$CLI"
    printf 'TLS2TREES_FINAL_EVALUATION_CLI_SHA256=%q\n' "$CLI_SHA256"
    printf 'TLS2TREES_FINAL_EVALUATION_EVALUATOR=%q\n' "$EVALUATOR"
    printf 'TLS2TREES_FINAL_EVALUATION_EVALUATOR_SHA256=%q\n' "$EVALUATOR_SHA256"
    printf 'TLS2TREES_FINAL_EVALUATION_TREEBENCH_ENV=%q\n' "$TREEBENCH_ENV"
    printf 'TLS2TREES_FINAL_EVALUATION_STAGE2_STATE=%q\n' "$STAGE2_STATE_FILE"
    printf 'TLS2TREES_FINAL_EVALUATION_STAGE2_STATE_SHA256=%q\n' "$STAGE2_STATE_SHA256"
    printf 'TLS2TREES_FINAL_EVALUATION_TEST_STATE=%q\n' "$TEST_STATE_FILE"
    printf 'TLS2TREES_FINAL_EVALUATION_TEST_STATE_SHA256=%q\n' "$TEST_STATE_SHA256"
    printf 'TLS2TREES_FINAL_EVALUATION_TEST_RETENTION_MANIFEST=%q\n' "$TEST_RETENTION_MANIFEST"
    printf 'TLS2TREES_FINAL_EVALUATION_TEST_RETENTION_MANIFEST_SHA256=%q\n' "$TEST_RETENTION_MANIFEST_SHA256"
    printf 'TLS2TREES_FINAL_EVALUATION_METRICS_ROOT=%q\n' "$METRICS_ROOT"
    printf 'TLS2TREES_FINAL_EVALUATION_SUMMARY_JSON=%q\n' "$SUMMARY_JSON"
    printf 'TLS2TREES_FINAL_EVALUATION_PLOT_CSV=%q\n' "$PLOT_CSV"
    printf 'TLS2TREES_FINAL_EVALUATION_AGGREGATE_CSV=%q\n' "$AGGREGATE_CSV"
    printf 'TLS2TREES_FINAL_EVALUATION_TEST_SUMMARY_JSON=%q\n' "$TEST_SUMMARY_JSON"
    printf 'TLS2TREES_FINAL_EVALUATION_TEST_PLOT_CSV=%q\n' "$TEST_PLOT_CSV"
    printf 'TLS2TREES_FINAL_EVALUATION_TEST_AGGREGATE_CSV=%q\n' "$TEST_AGGREGATE_CSV"
  } > "$STATE_FILE"
}
write_state

COMMON_EXPORTS="ALL,TLS2TREES_FINAL_EVALUATION_CONFIRMED=1,TLS2TREES_FINAL_EVALUATION_INFERENCE_ALLOWED=0,TLS2TREES_PROJECT_ROOT=$PROJECT_ROOT,TLS2TREES_FINAL_EVALUATION_BENCHMARK_COMMIT=$BENCHMARK_COMMIT,TLS2TREES_FINAL_EVALUATION_PLAN_JSON=$PLAN_JSON,TLS2TREES_FINAL_EVALUATION_PLAN_SHA256=$PLAN_SHA256,TLS2TREES_FINAL_EVALUATION_CLI=$CLI,TLS2TREES_FINAL_EVALUATION_CLI_SHA256=$CLI_SHA256,TLS2TREES_FINAL_EVALUATION_EVALUATOR=$EVALUATOR,TLS2TREES_FINAL_EVALUATION_EVALUATOR_SHA256=$EVALUATOR_SHA256,TLS2TREES_FINAL_EVALUATION_TREEBENCH_ENV=$TREEBENCH_ENV,TLS2TREES_FINAL_EVALUATION_SUMMARY_JSON=$SUMMARY_JSON,TLS2TREES_FINAL_EVALUATION_PLOT_CSV=$PLOT_CSV,TLS2TREES_FINAL_EVALUATION_AGGREGATE_CSV=$AGGREGATE_CSV,TLS2TREES_FINAL_EVALUATION_TEST_SUMMARY_JSON=$TEST_SUMMARY_JSON,TLS2TREES_FINAL_EVALUATION_TEST_PLOT_CSV=$TEST_PLOT_CSV,TLS2TREES_FINAL_EVALUATION_TEST_AGGREGATE_CSV=$TEST_AGGREGATE_CSV"
SUBMITTED_JOBS=()
cancel_partial_submission() {
  local status=$?
  if ((${#SUBMITTED_JOBS[@]})); then
    scancel "${SUBMITTED_JOBS[@]}" 2>/dev/null || true
  fi
  SUBMISSION_STATUS=retained_evaluation_submission_failed_jobs_cancelled
  write_state
  echo "Retained-prediction evaluation submission failed; new jobs were cancelled." >&2
  echo "state_file=$STATE_FILE" >&2
  exit "$status"
}
trap cancel_partial_submission ERR

EVALUATION_JOB=$(sbatch --parsable --array="0-105%4" --export="$COMMON_EXPORTS" \
  methods/tls2trees/slurm/for_instance/evaluate_retained_predictions.sbatch)
SUBMITTED_JOBS+=("$EVALUATION_JOB")
SUBMISSION_STATUS=retained_evaluation_array_submitted
write_state
SUMMARY_JOB=$(sbatch --parsable --dependency="afterok:$EVALUATION_JOB" \
  --kill-on-invalid-dep=yes --export="$COMMON_EXPORTS" \
  methods/tls2trees/slurm/for_instance/summarise_retained_predictions.sbatch)
SUBMITTED_JOBS+=("$SUMMARY_JOB")
SUBMISSION_STATUS=retained_evaluation_chain_submitted
write_state
printf '%s\n' "$STATE_FILE" > \
  logs/tls2trees_for_instance/latest_final_evaluation_state_file.txt
trap - ERR

echo "run_id=$RUN_ID"
echo "evaluation_job=$EVALUATION_JOB tasks=106 cpu_concurrency=4"
echo "summary_job=$SUMMARY_JOB"
echo "state_file=$STATE_FILE"
echo "source_stage2_run_id=$STAGE2_RUN_ID"
echo "source_held_out_run_id=$TEST_RUN_ID"
echo "inference_rerun=false"
