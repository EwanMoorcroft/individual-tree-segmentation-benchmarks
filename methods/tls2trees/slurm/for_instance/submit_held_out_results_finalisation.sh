#!/usr/bin/env bash

set -euo pipefail

if [[ "${TLS2TREES_FINALIZE_RESULTS_CONFIRMED:-0}" != "1" ]]; then
  echo "Refusing TLS2trees result finalisation." >&2
  echo "Set TLS2TREES_FINALIZE_RESULTS_CONFIRMED=1 for the completed retained evaluation." >&2
  exit 2
fi

PROJECT_ROOT="${TLS2TREES_PROJECT_ROOT:-$HOME/scratch/tree-seg-benchmark}"
STATE_FILE="${1:-}"
if [[ -z "$STATE_FILE" ]]; then
  POINTER="$PROJECT_ROOT/logs/tls2trees_for_instance/latest_final_evaluation_state_file.txt"
  test -s "$POINTER"
  STATE_FILE=$(tr -d '\r\n' < "$POINTER")
fi
test -f "$STATE_FILE"
# shellcheck disable=SC1090
source "$STATE_FILE"
FINAL_EVALUATION_STATE_FILE="$STATE_FILE"
FINAL_EVALUATION_RUN_ID="${TLS2TREES_FINAL_EVALUATION_RUN_ID:?evaluation state has no run ID}"
FINAL_EVALUATION_SUMMARY_JOB="${TLS2TREES_FINAL_EVALUATION_SUMMARY_JOB:?evaluation state has no summary job}"
FINAL_EVALUATION_BENCHMARK_COMMIT="${TLS2TREES_FINAL_EVALUATION_BENCHMARK_COMMIT:?evaluation state has no benchmark commit}"
FINAL_SUMMARY_JSON="${TLS2TREES_FINAL_EVALUATION_TEST_SUMMARY_JSON:?evaluation state has no held-out summary}"
SOURCE_TEST_STATE="${TLS2TREES_FINAL_EVALUATION_TEST_STATE:?evaluation state has no source held-out state}"
FINAL_EVALUATION_TREEBENCH_ENV="${TLS2TREES_FINAL_EVALUATION_TREEBENCH_ENV:?evaluation state has no treebench environment}"
test -f "$SOURCE_TEST_STATE"
# shellcheck disable=SC1090
source "$SOURCE_TEST_STATE"
test "$TLS2TREES_TEST_TREEBENCH_ENV" = "$FINAL_EVALUATION_TREEBENCH_ENV"

cd "$PROJECT_ROOT"
test -d .git
test -z "$(git status --porcelain)"
BENCHMARK_COMMIT=$(git rev-parse HEAD)
git cat-file -e "${FINAL_EVALUATION_BENCHMARK_COMMIT}^{commit}"
git merge-base --is-ancestor \
  "$FINAL_EVALUATION_BENCHMARK_COMMIT" "$BENCHMARK_COMMIT"
SUMMARY_JOB_STATE=$(sacct -X -n -P -j "$FINAL_EVALUATION_SUMMARY_JOB" \
  --format=JobIDRaw,State | awk -F'|' -v id="$FINAL_EVALUATION_SUMMARY_JOB" \
  '$1 == id {sub(/[+ ].*$/, "", $2); print $2; exit}')
test "$SUMMARY_JOB_STATE" = "COMPLETED"
test -s "$FINAL_SUMMARY_JSON"
test -s "$TLS2TREES_TEST_MANIFEST_JSON"
test -s "$TLS2TREES_TEST_FINAL_SELECTION_JSON"

"$TLS2TREES_TEST_TREEBENCH_ENV/bin/python" -c '
import json,sys
p=json.load(open(sys.argv[1]))
assert p["status"] == "held_out_test_completed"
assert p["valid_metric_count"] == p["expected_metric_count"] == 22
assert p["incomplete_tasks"] == []
assert p["held_out_test_accessed"] is True
assert p["configuration_changed_after_test"] is False
assert p["inference_rerun"] is False
assert p["prediction_adapter_rerun"] is False
assert p["retained_sources_unchanged"] is True
assert p["test_metrics_used_for_configuration_selection"] is False
assert p["evaluator"] == "for_instance_tls2trees_source_row_class3_ignore"
assert p["evaluation_protocol"] == "for_instance_pointwise_class3_ignore"
assert p["evaluation_mask"] == "union_of_reference_target_and_predicted_target_points_excluding_class3_outpoints"
assert p["workflow_run_id"] == sys.argv[2]
assert p["final_selection_sha256"] == sys.argv[3]
' "$FINAL_SUMMARY_JSON" "$TLS2TREES_TEST_RUN_ID" \
  "$TLS2TREES_TEST_FINAL_SELECTION_SHA256"

RESULT_ROOT="$PROJECT_ROOT/results/metadata/tls2trees/for_instance/development_tuned/test/$TLS2TREES_TEST_RUN_ID"
RECEIPT_JSON="$RESULT_ROOT/finalisation_receipt.json"
test -s methods/tls2trees/examples/tls2trees_development_tuned_prediction_retention_manifest.json

JOB_ID=$(sbatch --parsable \
  --export="ALL,TLS2TREES_FINALIZE_RESULTS_CONFIRMED=1,TLS2TREES_PROJECT_ROOT=$PROJECT_ROOT,TLS2TREES_FINALIZE_BENCHMARK_COMMIT=$BENCHMARK_COMMIT,TLS2TREES_FINALIZE_RECEIPT_JSON=$RECEIPT_JSON,TLS2TREES_TEST_TREEBENCH_ENV=$TLS2TREES_TEST_TREEBENCH_ENV,TLS2TREES_TEST_SUMMARY_JSON=$FINAL_SUMMARY_JSON,TLS2TREES_TEST_MANIFEST_JSON=$TLS2TREES_TEST_MANIFEST_JSON,TLS2TREES_TEST_FINAL_SELECTION_JSON=$TLS2TREES_TEST_FINAL_SELECTION_JSON,TLS2TREES_TEST_FINAL_SELECTION_SHA256=$TLS2TREES_TEST_FINAL_SELECTION_SHA256,TLS2TREES_TEST_RUN_ID=$TLS2TREES_TEST_RUN_ID,TLS2TREES_FINAL_EVALUATION_BENCHMARK_COMMIT=$FINAL_EVALUATION_BENCHMARK_COMMIT" \
  methods/tls2trees/slurm/for_instance/finalise_held_out_results.sbatch)

FINALIZE_STATE="$HOME/fastscratch/tls2trees_for_instance_test_states/${TLS2TREES_TEST_RUN_ID}_finalisation.env"
{
  printf 'TLS2TREES_FINALIZE_JOB=%q\n' "$JOB_ID"
  printf 'TLS2TREES_FINALIZE_RUN_ID=%q\n' "$TLS2TREES_TEST_RUN_ID"
  printf 'TLS2TREES_FINALIZE_SOURCE_STATE=%q\n' "$FINAL_EVALUATION_STATE_FILE"
  printf 'TLS2TREES_FINALIZE_EVALUATION_RUN_ID=%q\n' "$FINAL_EVALUATION_RUN_ID"
  printf 'TLS2TREES_FINALIZE_RECEIPT_JSON=%q\n' "$RECEIPT_JSON"
  printf 'TLS2TREES_FINALIZE_BENCHMARK_COMMIT=%q\n' "$BENCHMARK_COMMIT"
} > "$FINALIZE_STATE"
printf '%s\n' "$FINALIZE_STATE" > \
  logs/tls2trees_for_instance/latest_finalisation_state_file.txt

echo "status=tls2trees_result_finalisation_submitted"
echo "job_id=$JOB_ID"
echo "run_id=$TLS2TREES_TEST_RUN_ID"
echo "evaluation_run_id=$FINAL_EVALUATION_RUN_ID"
echo "state_file=$FINALIZE_STATE"
echo "monitor=bash methods/tls2trees/slurm/for_instance/monitor_held_out_results_finalisation.sh"
