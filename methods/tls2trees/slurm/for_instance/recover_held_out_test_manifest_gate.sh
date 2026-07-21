#!/usr/bin/env bash

set -euo pipefail

trap 'status=$?; echo "recovery_preflight_failed_at_line=$LINENO exit_code=$status" >&2' ERR

if [[ "${TLS2TREES_HELD_OUT_TEST_RECOVERY_CONFIRMED:-0}" != "1" ]]; then
  echo "Refusing held-out manifest-gate recovery." >&2
  echo "Set TLS2TREES_HELD_OUT_TEST_RECOVERY_CONFIRMED=1 after auditing the uniform gate failure." >&2
  exit 2
fi
STATE_FILE="${1:?Usage: recover_held_out_test_manifest_gate.sh <failed-test-state-file>}"
test -f "$STATE_FILE"
# shellcheck disable=SC1090
source "$STATE_FILE"

PROJECT_ROOT="${TLS2TREES_PROJECT_ROOT:-$HOME/scratch/tree-seg-benchmark}"
METHOD_ENV="${TLS2TREES_METHOD_ENV:-$HOME/fastscratch/venvs/tls2trees}"
METHOD_ENV_MARKER="$METHOD_ENV/.tls2trees_setup_complete.json"
UPSTREAM_REPO="${TLS2TREES_UPSTREAM_REPO:-$PROJECT_ROOT/external/TLS2trees}"
TREEBENCH_ENV="$TLS2TREES_TEST_TREEBENCH_ENV"
FINAL_SELECTION="$TLS2TREES_TEST_FINAL_SELECTION_JSON"
FINAL_SHA256="$TLS2TREES_TEST_FINAL_SELECTION_SHA256"
MANIFEST_JSON="$TLS2TREES_TEST_MANIFEST_JSON"
MANIFEST_SHA256_FILE="$TLS2TREES_TEST_MANIFEST_SHA256_FILE"
OUTPUT_ROOT="$TLS2TREES_TEST_OUTPUT_ROOT"
RUN_ID="$TLS2TREES_TEST_RUN_ID"
SEMANTIC_CACHE_RUN_ID="$TLS2TREES_TEST_SEMANTIC_CACHE_RUN_ID"
SUMMARY_JSON="$TLS2TREES_TEST_SUMMARY_JSON"
PLOT_CSV="$TLS2TREES_TEST_PLOT_CSV"
AGGREGATE_CSV="$TLS2TREES_TEST_AGGREGATE_CSV"
OLD_SEMANTIC_JOB="$TLS2TREES_TEST_SEMANTIC_JOB"
OLD_CANDIDATE_JOB="$TLS2TREES_TEST_CANDIDATE_JOB"
OLD_SUMMARY_JOB="$TLS2TREES_TEST_SUMMARY_JOB"

cd "$PROJECT_ROOT"
test -z "$(git status --porcelain)"
BENCHMARK_COMMIT=$(git rev-parse HEAD)
test "$BENCHMARK_COMMIT" != "$TLS2TREES_TEST_BENCHMARK_COMMIT"
test "$(sha256sum "$FINAL_SELECTION" | awk '{print $1}')" = "$FINAL_SHA256"
test "$(sha256sum "$MANIFEST_JSON" | awk '{print $1}')" = \
  "$(awk '{print $1}' "$MANIFEST_SHA256_FILE")"
test -f "$SUMMARY_JSON"
"$TREEBENCH_ENV/bin/python" -c '
import json,sys
p=json.load(open(sys.argv[1]))
assert p["status"] == "held_out_test_incomplete"
assert p["valid_metric_count"] == 0
assert p["expected_metric_count"] == 22
assert len(p["incomplete_tasks"]) == 22
assert p["configuration_changed_after_test"] is False
' "$SUMMARY_JSON"

for task in $(seq 0 10); do
  ERR="logs/tls2trees_for_instance/tls2trees_dt_test_semantic_${OLD_SEMANTIC_JOB}_${task}.err"
  test -f "$ERR"
  grep -Fq \
    "Held-out test manifest validation requires allow_held_out_test=True" \
    "$ERR"
done
FAILED_COUNT=$(sacct -X -n -P -j "$OLD_SEMANTIC_JOB" \
  --format=JobID%30,State | awk -F'|' -v id="$OLD_SEMANTIC_JOB" '
    {gsub(/^[[:space:]]+|[[:space:]]+$/, "", $1)}
    $1 ~ ("^" id "_[0-9]+$") {sub(/[+ ].*$/, "", $2); if ($2=="FAILED") n++}
    END {print n+0}')
test "$FAILED_COUNT" = "11"

SEMANTIC_CACHE_ROOT="$OUTPUT_ROOT/tls2trees/for_instance/development_tuned/test/$SEMANTIC_CACHE_RUN_ID"
test ! -e "$SEMANTIC_CACHE_ROOT"
RECOVERY_STAMP=$(date -u +%Y%m%d_%H%M%S)
RECOVERY_ROOT="$(dirname "$SUMMARY_JSON")/recovery/manifest_gate_$RECOVERY_STAMP"
mkdir -p "$RECOVERY_ROOT"
for path in "$SUMMARY_JSON" "$PLOT_CSV" "$AGGREGATE_CSV"; do
  test -f "$path"
  mv "$path" "$RECOVERY_ROOT/"
done

STAGE2_STATE=$(<logs/tls2trees_for_instance/latest_stage2_state_file.txt)
test -f "$STAGE2_STATE"
# shellcheck disable=SC1090
source "$STAGE2_STATE"
STAGE1_CONFIG="$TLS2TREES_STAGE2_STAGE1_CONFIG"
PROBE_SUMMARY_JSON="$TLS2TREES_STAGE2_PROBE_SUMMARY_JSON"
PROBE_SUMMARY_SHA256="$TLS2TREES_STAGE2_PROBE_SUMMARY_SHA256"
STAGE1_CONFIG_SHA256=$(sha256sum "$STAGE1_CONFIG" | awk '{print $1}')
METHOD_ENV_MARKER_SHA256=$(sha256sum "$METHOD_ENV_MARKER" | awk '{print $1}')
MANIFEST_CLI="methods/tls2trees/scripts/data/prepare_for_instance_manifest.py"
CONVERT_CLI="methods/tls2trees/scripts/data/prepare_for_instance_tls2trees_input.py"
SEMANTIC_CLI="methods/tls2trees/scripts/runtime/run_for_instance_tls2trees_semantic.py"
CANDIDATE_CLI="methods/tls2trees/scripts/runtime/run_for_instance_tls2trees_development_candidate.py"
ADAPTER_CLI="methods/tls2trees/scripts/evaluation/adapt_for_instance_tls2trees_predictions.py"
EVALUATE_CLI="methods/tls2trees/scripts/evaluation/evaluate_for_instance_tls2trees_plot.py"
ENV_VALIDATOR="methods/tls2trees/scripts/runtime/validate_tls2trees_environment.py"

RECOVERY_STATE="${STATE_FILE%.env}_manifest_gate_recovery_$RECOVERY_STAMP.env"
test ! -e "$RECOVERY_STATE"
cp "$STATE_FILE" "$RECOVERY_STATE"
INVENTORY_JOB="$TLS2TREES_TEST_INVENTORY_JOB"
SEMANTIC_JOB=not_submitted
CANDIDATE_JOB=not_submitted
SUMMARY_JOB=not_submitted
SUBMISSION_STATUS=audited_manifest_gate_recovery_preflight
update_state() {
  {
    printf 'TLS2TREES_TEST_SUBMISSION_STATUS=%q\n' "$SUBMISSION_STATUS"
    printf 'TLS2TREES_TEST_INVENTORY_JOB=%q\n' "$INVENTORY_JOB"
    printf 'TLS2TREES_TEST_SEMANTIC_JOB=%q\n' "$SEMANTIC_JOB"
    printf 'TLS2TREES_TEST_CANDIDATE_JOB=%q\n' "$CANDIDATE_JOB"
    printf 'TLS2TREES_TEST_SUMMARY_JOB=%q\n' "$SUMMARY_JOB"
    printf 'TLS2TREES_TEST_BENCHMARK_COMMIT=%q\n' "$BENCHMARK_COMMIT"
    printf 'TLS2TREES_TEST_RECOVERY_FROM_SEMANTIC_JOB=%q\n' "$OLD_SEMANTIC_JOB"
    printf 'TLS2TREES_TEST_RECOVERY_FROM_CANDIDATE_JOB=%q\n' "$OLD_CANDIDATE_JOB"
    printf 'TLS2TREES_TEST_RECOVERY_FROM_SUMMARY_JOB=%q\n' "$OLD_SUMMARY_JOB"
    printf 'TLS2TREES_TEST_RECOVERY_ARCHIVE=%q\n' "$RECOVERY_ROOT"
  } >> "$RECOVERY_STATE"
}
update_state
printf '%s\n' "$RECOVERY_STATE" > \
  logs/tls2trees_for_instance/latest_held_out_test_state_file.txt

COMMON_EXPORTS="ALL,TLS2TREES_HELD_OUT_TEST_CONFIRMED=1,TLS2TREES_REQUESTED_VARIANT=development_tuned,TLS2TREES_REQUESTED_SPLIT=test,TLS2TREES_PROJECT_ROOT=$PROJECT_ROOT,TLS2TREES_TREEBENCH_ENV=$TREEBENCH_ENV,TLS2TREES_METHOD_ENV=$METHOD_ENV,TLS2TREES_METHOD_ENV_MARKER=$METHOD_ENV_MARKER,TLS2TREES_METHOD_ENV_MARKER_SHA256=$METHOD_ENV_MARKER_SHA256,TLS2TREES_UPSTREAM_REPO=$UPSTREAM_REPO,TLS2TREES_EXPECTED_BENCHMARK_COMMIT=$BENCHMARK_COMMIT,TLS2TREES_MANIFEST_JSON=$MANIFEST_JSON,TLS2TREES_MANIFEST_SHA256_FILE=$MANIFEST_SHA256_FILE,TLS2TREES_MANIFEST_CLI=$MANIFEST_CLI,TLS2TREES_CONVERT_CLI=$CONVERT_CLI,TLS2TREES_SEMANTIC_CLI=$SEMANTIC_CLI,TLS2TREES_CANDIDATE_CLI=$CANDIDATE_CLI,TLS2TREES_ADAPTER_CLI=$ADAPTER_CLI,TLS2TREES_EVALUATE_CLI=$EVALUATE_CLI,TLS2TREES_ENV_VALIDATOR=$ENV_VALIDATOR,TLS2TREES_STAGE1_CONFIG=$STAGE1_CONFIG,TLS2TREES_STAGE1_CONFIG_SHA256=$STAGE1_CONFIG_SHA256,TLS2TREES_PROBE_SUMMARY_JSON=$PROBE_SUMMARY_JSON,TLS2TREES_PROBE_SUMMARY_SHA256=$PROBE_SUMMARY_SHA256,TLS2TREES_FINAL_SELECTION_JSON=$FINAL_SELECTION,TLS2TREES_FINAL_SELECTION_SHA256=$FINAL_SHA256,TLS2TREES_OUTPUT_ROOT=$OUTPUT_ROOT,TLS2TREES_TEST_RUN_ID=$RUN_ID,TLS2TREES_TEST_SEMANTIC_CACHE_RUN_ID=$SEMANTIC_CACHE_RUN_ID,TLS2TREES_TEST_SUMMARY_JSON=$SUMMARY_JSON,TLS2TREES_TEST_PLOT_CSV=$PLOT_CSV,TLS2TREES_TEST_AGGREGATE_CSV=$AGGREGATE_CSV"

SUBMITTED_JOBS=()
cancel_partial_submission() {
  local status=$?
  if ((${#SUBMITTED_JOBS[@]})); then
    scancel "${SUBMITTED_JOBS[@]}" 2>/dev/null || true
  fi
  SUBMISSION_STATUS=manifest_gate_recovery_submission_failed_jobs_cancelled
  update_state
  exit "$status"
}
trap cancel_partial_submission ERR
SEMANTIC_JOB=$(sbatch --parsable --array="0-10%2" --export="$COMMON_EXPORTS" \
  methods/tls2trees/slurm/for_instance/prepare_semantic_held_out_test.sbatch)
SUBMITTED_JOBS+=("$SEMANTIC_JOB")
SUBMISSION_STATUS=manifest_gate_recovery_semantic_submitted; update_state
CANDIDATE_JOB=$(sbatch --parsable --array="0-21%4" --dependency="afterok:$SEMANTIC_JOB" \
  --kill-on-invalid-dep=yes --export="$COMMON_EXPORTS" \
  methods/tls2trees/slurm/for_instance/evaluate_held_out_test_candidate.sbatch)
SUBMITTED_JOBS+=("$CANDIDATE_JOB")
SUBMISSION_STATUS=manifest_gate_recovery_candidates_submitted; update_state
SUMMARY_JOB=$(sbatch --parsable --dependency="afterany:$CANDIDATE_JOB" \
  --kill-on-invalid-dep=yes --export="$COMMON_EXPORTS" \
  methods/tls2trees/slurm/for_instance/summarise_held_out_test.sbatch)
SUBMITTED_JOBS+=("$SUMMARY_JOB")
SUBMISSION_STATUS=audited_manifest_gate_recovery_chain_submitted; update_state
trap - ERR

echo "status=audited_manifest_gate_recovery_chain_submitted"
echo "run_id=$RUN_ID"
echo "semantic_job=$SEMANTIC_JOB tasks=11 concurrency=2"
echo "candidate_job=$CANDIDATE_JOB tasks=22 concurrency=4"
echo "summary_job=$SUMMARY_JOB"
echo "state_file=$RECOVERY_STATE"
echo "archived_incomplete_summary=$RECOVERY_ROOT"
echo "configuration_changed=false"
