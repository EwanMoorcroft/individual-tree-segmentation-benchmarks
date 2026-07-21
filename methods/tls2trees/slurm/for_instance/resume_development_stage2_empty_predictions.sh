#!/usr/bin/env bash

set -euo pipefail

report_preflight_failure() {
  local status=$?
  echo "Stage 2 recovery preflight failed at line $1 (exit $status)." >&2
  exit "$status"
}
trap 'report_preflight_failure "$LINENO"' ERR

if [[ "${TLS2TREES_STAGE2_RECOVERY_CONFIRMED:-0}" != "1" ]]; then
  echo "Refusing audited Stage 2 empty-prediction recovery." >&2
  echo "Set TLS2TREES_STAGE2_RECOVERY_CONFIRMED=1 after reviewing the four tracebacks." >&2
  exit 2
fi

ORIGINAL_STATE_FILE="${1:?Usage: resume_development_stage2_empty_predictions.sh <stage2-state-file>}"
test -f "$ORIGINAL_STATE_FILE"
# shellcheck disable=SC1090
source "$ORIGINAL_STATE_FILE"

RUN_ID="${TLS2TREES_STAGE2_RUN_ID:?State has no Stage 2 run ID}"
OLD_SEMANTIC_JOB="${TLS2TREES_STAGE2_SEMANTIC_JOB:?State has no semantic job}"
OLD_CANDIDATE_JOB="${TLS2TREES_STAGE2_CANDIDATE_JOB:?State has no candidate job}"
OLD_SUMMARY_JOB="${TLS2TREES_STAGE2_SUMMARY_JOB:?State has no summary job}"
OLD_BENCHMARK_COMMIT="${TLS2TREES_STAGE2_BENCHMARK_COMMIT:?State has no benchmark commit}"
STAGE1_STATE="${TLS2TREES_STAGE2_SOURCE_STAGE1_STATE:?State has no Stage 1 state}"
STAGE1_RUN_ID="${TLS2TREES_STAGE2_SOURCE_STAGE1_RUN_ID:?State has no Stage 1 run ID}"
STAGE1_SUMMARY_JSON="${TLS2TREES_STAGE2_SOURCE_STAGE1_SUMMARY_JSON:?State has no Stage 1 summary}"
STAGE1_SUMMARY_SHA256="${TLS2TREES_STAGE2_SOURCE_STAGE1_SUMMARY_SHA256:?State has no Stage 1 summary hash}"
SELECTION_JSON="${TLS2TREES_STAGE2_SELECTION_JSON:?State has no selection manifest}"
SELECTION_SHA256="${TLS2TREES_STAGE2_SELECTION_SHA256:?State has no selection hash}"
MANIFEST_JSON="${TLS2TREES_STAGE2_MANIFEST_JSON:?State has no development manifest}"
MANIFEST_SHA256="${TLS2TREES_STAGE2_MANIFEST_SHA256:?State has no manifest hash}"
STAGE1_CONFIG="${TLS2TREES_STAGE2_STAGE1_CONFIG:?State has no Stage 1 config}"
STAGE1_CONFIG_SHA256="${TLS2TREES_STAGE2_STAGE1_CONFIG_SHA256:?State has no config hash}"
PROBE_SUMMARY_JSON="${TLS2TREES_STAGE2_PROBE_SUMMARY_JSON:?State has no probe summary}"
PROBE_SUMMARY_SHA256="${TLS2TREES_STAGE2_PROBE_SUMMARY_SHA256:?State has no probe hash}"
SEMANTIC_CACHE_RUN_ID="${TLS2TREES_STAGE2_SEMANTIC_CACHE_RUN_ID:?State has no semantic cache}"
OUTPUT_ROOT="${TLS2TREES_STAGE2_OUTPUT_ROOT:?State has no output root}"
OLD_SUMMARY_JSON="${TLS2TREES_STAGE2_SUMMARY_JSON:?State has no Stage 2 summary}"
OLD_PLOT_CSV="${TLS2TREES_STAGE2_PLOT_CSV:?State has no plot CSV}"
OLD_AGGREGATE_CSV="${TLS2TREES_STAGE2_AGGREGATE_CSV:?State has no aggregate CSV}"
TREEBENCH_ENV="${TLS2TREES_STAGE2_TREEBENCH_ENV:-$HOME/fastscratch/venvs/treebench}"
OLD_EXPECTED_TASK_COUNT="${TLS2TREES_STAGE2_EXPECTED_CANDIDATE_TASKS:-42}"

test "$RUN_ID" = \
  "tls2trees_for-instance_development_tuned_stage2_20260718_202002"
test "$OLD_SEMANTIC_JOB" = "9836432"
case "$OLD_CANDIDATE_JOB" in
  9836433)
    RECOVERY_PHASE="initial_empty_prediction_recovery"
    test "$OLD_SUMMARY_JOB" = "9836434"
    test "$OLD_BENCHMARK_COMMIT" = \
      "13757caafc5e8b7df5bb2878bc41866389f37e46"
    test "$OLD_EXPECTED_TASK_COUNT" = "42"
    EXPECTED_VALID_METRICS=76
    EXPECTED_FAILED_LIST="6,18,27,39"
    EXPECTED_OLD_SUMMARY_SHA256=\
"fb752dd05d64d335d412870a81cae921ab73b8f19aeb961be0c082fa52e76d48"
    ;;
  9838903)
    RECOVERY_PHASE="empty_leaf_tip_recovery"
    test "$OLD_SUMMARY_JOB" = "9838904"
    test "$OLD_BENCHMARK_COMMIT" = \
      "283cdf3f388d86656dc5ec527208c25af02c5ce0"
    test "$OLD_EXPECTED_TASK_COUNT" = "4"
    EXPECTED_VALID_METRICS=80
    EXPECTED_FAILED_LIST="6,27"
    EXPECTED_OLD_SUMMARY_SHA256=""
    ;;
  *)
    echo "Unsupported Stage 2 recovery source job: $OLD_CANDIDATE_JOB" >&2
    exit 2
    ;;
esac

PROJECT_ROOT="${TLS2TREES_PROJECT_ROOT:-$HOME/scratch/tree-seg-benchmark}"
METHOD_ENV="${TLS2TREES_METHOD_ENV:-$HOME/fastscratch/venvs/tls2trees}"
METHOD_ENV_MARKER="$METHOD_ENV/.tls2trees_setup_complete.json"
UPSTREAM_REPO="${TLS2TREES_UPSTREAM_REPO:-$PROJECT_ROOT/external/TLS2trees}"
MANIFEST_CLI="methods/tls2trees/scripts/data/prepare_for_instance_manifest.py"
CANDIDATE_CLI="methods/tls2trees/scripts/runtime/run_for_instance_tls2trees_development_candidate.py"
ADAPTER_CLI="methods/tls2trees/scripts/evaluation/adapt_for_instance_tls2trees_predictions.py"
EVALUATE_CLI="methods/tls2trees/scripts/evaluation/evaluate_for_instance_tls2trees_plot.py"
ENV_VALIDATOR="methods/tls2trees/scripts/runtime/validate_tls2trees_environment.py"

cd "$PROJECT_ROOT"
test -z "$(git status --porcelain)"
BENCHMARK_COMMIT=$(git rev-parse HEAD)
test "$BENCHMARK_COMMIT" != "$OLD_BENCHMARK_COMMIT"
test -x "$TREEBENCH_ENV/bin/python"
test -x "$METHOD_ENV/bin/python"
test -f "$METHOD_ENV_MARKER"
METHOD_ENV_MARKER_SHA256=$(sha256sum "$METHOD_ENV_MARKER" | awk '{print $1}')
test "$(git -C "$UPSTREAM_REPO" rev-parse HEAD)" = \
  "ca12cb73b2c736d80b020e8025f8d975d42e6f01"
test -z "$(git -C "$UPSTREAM_REPO" status --porcelain)"
test "$(sha256sum "$STAGE1_SUMMARY_JSON" | awk '{print $1}')" = "$STAGE1_SUMMARY_SHA256"
test "$(sha256sum "$SELECTION_JSON" | awk '{print $1}')" = "$SELECTION_SHA256"
test "$(sha256sum "$MANIFEST_JSON" | awk '{print $1}')" = "$MANIFEST_SHA256"
test "$(sha256sum "$STAGE1_CONFIG" | awk '{print $1}')" = "$STAGE1_CONFIG_SHA256"
test "$(sha256sum "$PROBE_SUMMARY_JSON" | awk '{print $1}')" = "$PROBE_SUMMARY_SHA256"
test -f "$OLD_SUMMARY_JSON"
if [[ -n "$EXPECTED_OLD_SUMMARY_SHA256" ]]; then
  test "$(sha256sum "$OLD_SUMMARY_JSON" | awk '{print $1}')" = \
    "$EXPECTED_OLD_SUMMARY_SHA256"
fi
"$TREEBENCH_ENV/bin/python" -c '
import json,sys
p=json.load(open(sys.argv[1]))
assert p["status"] == "stage2_incomplete"
assert p["valid_metric_count"] == int(sys.argv[2])
assert p["expected_metric_count"] == 84
assert p["held_out_test_accessed"] is False
assert p["final_configuration_selected"] is False
' "$OLD_SUMMARY_JSON" "$EXPECTED_VALID_METRICS"

SUMMARY_STATE=$(sacct -X -n -P -j "$OLD_SUMMARY_JOB" --format=JobIDRaw,State | \
  awk -F'|' -v id="$OLD_SUMMARY_JOB" '$1 == id {sub(/[+ ].*$/, "", $2); print $2; exit}')
test "$SUMMARY_STATE" = "COMPLETED"

mapfile -t TASK_STATES < <(sacct -X -n -P -j "$OLD_CANDIDATE_JOB" \
  --format=JobID%30,State | awk -F'|' -v id="$OLD_CANDIDATE_JOB" '
  {gsub(/^[[:space:]]+|[[:space:]]+$/, "", $1)}
  $1 ~ ("^" id "_[0-9]+$") {sub(/[+ ].*$/, "", $2); print $1 " " $2}')
if ((${#TASK_STATES[@]} != OLD_EXPECTED_TASK_COUNT)); then
  echo "Expected $OLD_EXPECTED_TASK_COUNT source task states; found ${#TASK_STATES[@]}." >&2
  exit 2
fi
FAILED_TASKS=()
for record in "${TASK_STATES[@]}"; do
  TASK_JOB=${record%% *}
  TASK_STATE=${record#* }
  ARRAY_TASK=${TASK_JOB##*_}
  case "$TASK_STATE" in
    COMPLETED) ;;
    FAILED) FAILED_TASKS+=("$ARRAY_TASK") ;;
    *) echo "Refusing recovery with task $ARRAY_TASK in state $TASK_STATE." >&2; exit 2 ;;
  esac
done
mapfile -t FAILED_TASKS < <(printf '%s\n' "${FAILED_TASKS[@]}" | sort -n)
FAILED_LIST=$(IFS=,; echo "${FAILED_TASKS[*]}")
if [[ "$FAILED_LIST" != "$EXPECTED_FAILED_LIST" ]]; then
  echo "Expected audited failed tasks $EXPECTED_FAILED_LIST; found ${FAILED_LIST:-NONE}." >&2
  exit 2
fi

for ARRAY_TASK in "${FAILED_TASKS[@]}"; do
  SELECTION_INDEX=$((ARRAY_TASK / 21))
  TASK_INDEX=$((ARRAY_TASK % 21))
  CANDIDATE_ID=$("$TREEBENCH_ENV/bin/python" -c '
import json,sys
p=json.load(open(sys.argv[1]))
print(p["selected_candidates"][int(sys.argv[2])]["candidate_id"])
' "$SELECTION_JSON" "$SELECTION_INDEX")
  SAFE_PLOT_ID=$("$TREEBENCH_ENV/bin/python" "$MANIFEST_CLI" resolve \
    --manifest-json "$MANIFEST_JSON" --expected-split development \
    --task-index "$TASK_INDEX" --field safe_plot_id)
  CANDIDATE_RUN_ID="${RUN_ID}__${CANDIDATE_ID}"
  PLOT_ROOT="$OUTPUT_ROOT/tls2trees/for_instance/development_tuned/development/$CANDIDATE_RUN_ID/$SAFE_PLOT_ID"
  METADATA="$PLOT_ROOT/metadata/instance_run.json"
  test -f "$METADATA"
  test ! -e "$PLOT_ROOT/predictions/aligned"
  test ! -e "$PLOT_ROOT/evaluation"
  "$TREEBENCH_ENV/bin/python" -c '
import json,sys
p=json.load(open(sys.argv[1]))
assert p["status"] == "failed"
assert "Instance tile" in str(p.get("error", ""))
assert p["held_out_test_accessed"] is False
' "$METADATA"
  mapfile -t TILE_ERRORS < <(find "$PLOT_ROOT/logs/instance" -maxdepth 1 \
    -type f -name 'tile_*.stderr.log' -print)
  ((${#TILE_ERRORS[@]} > 0))
  case "$ARRAY_TASK" in
    6|27)
      grep -lF 'cannot set a frame with no defined index and a scalar' \
        "${TILE_ERRORS[@]}" >/dev/null
      ;;
    18|39)
      grep -lF 'Cannot restore clstr: groupby.apply did not return a grouped index' \
        "${TILE_ERRORS[@]}" >/dev/null
      ;;
  esac
done

STAMP="${TLS2TREES_STAGE2_RECOVERY_STAMP:-$(date -u +%Y%m%d_%H%M%S)}"
if [[ ! "$STAMP" =~ ^[0-9]{8}_[0-9]{6}$ ]]; then
  echo "TLS2TREES_STAGE2_RECOVERY_STAMP must use YYYYMMDD_HHMMSS." >&2
  exit 2
fi
RECOVERY_SUMMARY_JSON="${OLD_SUMMARY_JSON%.json}_recovery_${STAMP}.json"
RECOVERY_PLOT_CSV="${OLD_PLOT_CSV%.csv}_recovery_${STAMP}.csv"
RECOVERY_AGGREGATE_CSV="${OLD_AGGREGATE_CSV%.csv}_recovery_${STAMP}.csv"
STATE_DIR="$HOME/fastscratch/tls2trees_for_instance_stage2_states"
STATE_FILE="$STATE_DIR/${RUN_ID}_audited_empty_prediction_recovery_${STAMP}.env"
for path in "$RECOVERY_SUMMARY_JSON" "$RECOVERY_PLOT_CSV" \
  "$RECOVERY_AGGREGATE_CSV" "$STATE_FILE"; do
  test ! -e "$path"
done

RECOVERY_JOB="not_submitted"
SUMMARY_JOB="not_submitted"
SUBMISSION_STATUS="audited_empty_prediction_recovery_preflight_completed"
SUBMITTED_JOBS=()
write_state() {
  {
    printf 'TLS2TREES_STAGE2_RUN_ID=%q\n' "$RUN_ID"
    printf 'TLS2TREES_STAGE2_SUBMISSION_STATUS=%q\n' "$SUBMISSION_STATUS"
    printf 'TLS2TREES_STAGE2_SEMANTIC_JOB=%q\n' "$OLD_SEMANTIC_JOB"
    printf 'TLS2TREES_STAGE2_CANDIDATE_JOB=%q\n' "$RECOVERY_JOB"
    printf 'TLS2TREES_STAGE2_EXPECTED_CANDIDATE_TASKS=%q\n' "${#FAILED_TASKS[@]}"
    printf 'TLS2TREES_STAGE2_SUMMARY_JOB=%q\n' "$SUMMARY_JOB"
    printf 'TLS2TREES_STAGE2_BENCHMARK_COMMIT=%q\n' "$BENCHMARK_COMMIT"
    printf 'TLS2TREES_STAGE2_SOURCE_STAGE1_STATE=%q\n' "$STAGE1_STATE"
    printf 'TLS2TREES_STAGE2_SOURCE_STAGE1_RUN_ID=%q\n' "$STAGE1_RUN_ID"
    printf 'TLS2TREES_STAGE2_SOURCE_STAGE1_SUMMARY_JSON=%q\n' "$STAGE1_SUMMARY_JSON"
    printf 'TLS2TREES_STAGE2_SOURCE_STAGE1_SUMMARY_SHA256=%q\n' "$STAGE1_SUMMARY_SHA256"
    printf 'TLS2TREES_STAGE2_SELECTION_JSON=%q\n' "$SELECTION_JSON"
    printf 'TLS2TREES_STAGE2_SELECTION_SHA256=%q\n' "$SELECTION_SHA256"
    printf 'TLS2TREES_STAGE2_MANIFEST_JSON=%q\n' "$MANIFEST_JSON"
    printf 'TLS2TREES_STAGE2_MANIFEST_SHA256=%q\n' "$MANIFEST_SHA256"
    printf 'TLS2TREES_STAGE2_STAGE1_CONFIG=%q\n' "$STAGE1_CONFIG"
    printf 'TLS2TREES_STAGE2_STAGE1_CONFIG_SHA256=%q\n' "$STAGE1_CONFIG_SHA256"
    printf 'TLS2TREES_STAGE2_PROBE_SUMMARY_JSON=%q\n' "$PROBE_SUMMARY_JSON"
    printf 'TLS2TREES_STAGE2_PROBE_SUMMARY_SHA256=%q\n' "$PROBE_SUMMARY_SHA256"
    printf 'TLS2TREES_STAGE2_SEMANTIC_CACHE_RUN_ID=%q\n' "$SEMANTIC_CACHE_RUN_ID"
    printf 'TLS2TREES_STAGE2_OUTPUT_ROOT=%q\n' "$OUTPUT_ROOT"
    printf 'TLS2TREES_STAGE2_SUMMARY_JSON=%q\n' "$RECOVERY_SUMMARY_JSON"
    printf 'TLS2TREES_STAGE2_PLOT_CSV=%q\n' "$RECOVERY_PLOT_CSV"
    printf 'TLS2TREES_STAGE2_AGGREGATE_CSV=%q\n' "$RECOVERY_AGGREGATE_CSV"
    printf 'TLS2TREES_STAGE2_TREEBENCH_ENV=%q\n' "$TREEBENCH_ENV"
    printf 'TLS2TREES_STAGE2_RECOVERY_FROM_CANDIDATE_JOB=%q\n' "$OLD_CANDIDATE_JOB"
    printf 'TLS2TREES_STAGE2_RECOVERY_TASKS=%q\n' "$FAILED_LIST"
    printf 'TLS2TREES_STAGE2_RECOVERY_PHASE=%q\n' "$RECOVERY_PHASE"
  } > "$STATE_FILE"
}
cancel_partial_submission() {
  local status=$?
  if ((${#SUBMITTED_JOBS[@]})); then
    scancel "${SUBMITTED_JOBS[@]}" 2>/dev/null || true
  fi
  SUBMISSION_STATUS="audited_empty_prediction_recovery_submission_failed_jobs_cancelled"
  write_state
  exit "$status"
}
trap cancel_partial_submission ERR

COMMON_EXPORTS="ALL,TLS2TREES_STAGE2_CONFIRMED=1,TLS2TREES_STAGE2_RECOVERY=1,TLS2TREES_REQUESTED_VARIANT=development_tuned,TLS2TREES_REQUESTED_SPLIT=development,TLS2TREES_PROJECT_ROOT=$PROJECT_ROOT,TLS2TREES_TREEBENCH_ENV=$TREEBENCH_ENV,TLS2TREES_METHOD_ENV=$METHOD_ENV,TLS2TREES_METHOD_ENV_MARKER=$METHOD_ENV_MARKER,TLS2TREES_METHOD_ENV_MARKER_SHA256=$METHOD_ENV_MARKER_SHA256,TLS2TREES_UPSTREAM_REPO=$UPSTREAM_REPO,TLS2TREES_EXPECTED_BENCHMARK_COMMIT=$BENCHMARK_COMMIT,TLS2TREES_MANIFEST_JSON=$MANIFEST_JSON,TLS2TREES_MANIFEST_SHA256=$MANIFEST_SHA256,TLS2TREES_MANIFEST_CLI=$MANIFEST_CLI,TLS2TREES_CANDIDATE_CLI=$CANDIDATE_CLI,TLS2TREES_ADAPTER_CLI=$ADAPTER_CLI,TLS2TREES_EVALUATE_CLI=$EVALUATE_CLI,TLS2TREES_ENV_VALIDATOR=$ENV_VALIDATOR,TLS2TREES_STAGE1_CONFIG=$STAGE1_CONFIG,TLS2TREES_STAGE1_CONFIG_SHA256=$STAGE1_CONFIG_SHA256,TLS2TREES_PROBE_SUMMARY_JSON=$PROBE_SUMMARY_JSON,TLS2TREES_PROBE_SUMMARY_SHA256=$PROBE_SUMMARY_SHA256,TLS2TREES_OUTPUT_ROOT=$OUTPUT_ROOT,TLS2TREES_STAGE2_RUN_ID=$RUN_ID,TLS2TREES_STAGE2_SEMANTIC_CACHE_RUN_ID=$SEMANTIC_CACHE_RUN_ID,TLS2TREES_STAGE2_SELECTION_JSON=$SELECTION_JSON,TLS2TREES_STAGE2_SELECTION_SHA256=$SELECTION_SHA256,TLS2TREES_STAGE2_SUMMARY_JSON=$RECOVERY_SUMMARY_JSON,TLS2TREES_STAGE2_PLOT_CSV=$RECOVERY_PLOT_CSV,TLS2TREES_STAGE2_AGGREGATE_CSV=$RECOVERY_AGGREGATE_CSV"

RECOVERY_JOB=$(sbatch --parsable --array="$FAILED_LIST%2" --export="$COMMON_EXPORTS" \
  methods/tls2trees/slurm/for_instance/evaluate_development_stage2_candidate.sbatch)
SUBMITTED_JOBS+=("$RECOVERY_JOB")
SUBMISSION_STATUS="audited_empty_prediction_recovery_array_submitted"
write_state
SUMMARY_JOB=$(sbatch --parsable --dependency="afterany:$RECOVERY_JOB" \
  --kill-on-invalid-dep=yes --export="$COMMON_EXPORTS" \
  methods/tls2trees/slurm/for_instance/summarise_development_stage2.sbatch)
SUBMITTED_JOBS+=("$SUMMARY_JOB")
SUBMISSION_STATUS="audited_empty_prediction_recovery_chain_submitted"
write_state
printf '%s\n' "$STATE_FILE" > logs/tls2trees_for_instance/latest_stage2_state_file.txt
trap - ERR

echo "run_id=$RUN_ID"
echo "recovery_from_candidate_job=$OLD_CANDIDATE_JOB"
echo "recovery_phase=$RECOVERY_PHASE"
echo "recovery_tasks=$FAILED_LIST"
echo "recovery_job=$RECOVERY_JOB summary_job=$SUMMARY_JOB"
echo "state_file=$STATE_FILE"
echo "completed_tasks_and_semantic_cache_reused=true"
echo "expected_metrics=84"
echo "final_configuration_selected=false"
echo "held_out_test_accessed=false"
