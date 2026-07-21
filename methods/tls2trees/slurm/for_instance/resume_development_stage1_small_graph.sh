#!/usr/bin/env bash

set -euo pipefail

report_preflight_failure() {
  local status=$?
  echo "Stage 1 recovery preflight failed at line $1 (exit $status)." >&2
  exit "$status"
}
trap 'report_preflight_failure "$LINENO"' ERR

if [[ "${TLS2TREES_STAGE1_RECOVERY_CONFIRMED:-0}" != "1" ]]; then
  echo "Refusing audited Stage 1 recovery." >&2
  echo "Set TLS2TREES_STAGE1_RECOVERY_CONFIRMED=1 after reviewing all failed task tracebacks." >&2
  exit 2
fi

ORIGINAL_STATE_FILE="${1:?Usage: resume_development_stage1_small_graph.sh <stage1-state-file>}"
test -f "$ORIGINAL_STATE_FILE"
# shellcheck disable=SC1090
source "$ORIGINAL_STATE_FILE"

RUN_ID="${TLS2TREES_STAGE1_RUN_ID:?State has no Stage 1 run ID}"
OLD_CANDIDATE_JOB="${TLS2TREES_STAGE1_CANDIDATE_JOB:?State has no candidate job}"
OLD_SUMMARY_JOB="${TLS2TREES_STAGE1_SUMMARY_JOB:?State has no summary job}"
OLD_BENCHMARK_COMMIT="${TLS2TREES_STAGE1_BENCHMARK_COMMIT:?State has no benchmark commit}"
PROBE_RUN_ID="${TLS2TREES_STAGE1_PROBE_RUN_ID:?State has no probe run ID}"
PROBE_SUMMARY_JSON="${TLS2TREES_STAGE1_PROBE_SUMMARY_JSON:?State has no probe summary}"
PROBE_SUMMARY_SHA256="${TLS2TREES_STAGE1_PROBE_SUMMARY_SHA256:?State has no probe summary hash}"
MANIFEST_JSON="${TLS2TREES_STAGE1_MANIFEST_JSON:?State has no manifest}"
STAGE1_CONFIG="${TLS2TREES_STAGE1_CONFIG:?State has no Stage 1 config}"
STAGE1_CONFIG_SHA256="${TLS2TREES_STAGE1_CONFIG_SHA256:?State has no Stage 1 config hash}"
SEMANTIC_CACHE_RUN_ID="${TLS2TREES_STAGE1_SEMANTIC_CACHE_RUN_ID:?State has no semantic cache}"
OUTPUT_ROOT="${TLS2TREES_STAGE1_OUTPUT_ROOT:?State has no output root}"
OLD_SUMMARY_JSON="${TLS2TREES_STAGE1_SUMMARY_JSON:?State has no summary path}"
OLD_PLOT_CSV="${TLS2TREES_STAGE1_PLOT_CSV:?State has no plot CSV}"
OLD_AGGREGATE_CSV="${TLS2TREES_STAGE1_AGGREGATE_CSV:?State has no aggregate CSV}"
TREEBENCH_ENV="${TLS2TREES_STAGE1_TREEBENCH_ENV:-$HOME/fastscratch/venvs/treebench}"

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
test "$(sha256sum "$PROBE_SUMMARY_JSON" | awk '{print $1}')" = "$PROBE_SUMMARY_SHA256"
test "$(sha256sum "$STAGE1_CONFIG" | awk '{print $1}')" = "$STAGE1_CONFIG_SHA256"
test -f "$OLD_SUMMARY_JSON"
"$TREEBENCH_ENV/bin/python" -c \
  'import json,sys; p=json.load(open(sys.argv[1])); assert p.get("status") == "stage1_incomplete"' \
  "$OLD_SUMMARY_JSON"

SUMMARY_STATE=$(sacct -X -n -P -j "$OLD_SUMMARY_JOB" --format=JobIDRaw,State | \
  awk -F'|' -v id="$OLD_SUMMARY_JOB" '$1 == id {sub(/[+ ].*$/, "", $2); print $2; exit}')
if [[ "$SUMMARY_STATE" != "COMPLETED" ]]; then
  echo "Original Stage 1 summary is not COMPLETED: ${SUMMARY_STATE:-unknown}" >&2
  exit 2
fi

mapfile -t TASK_STATES < <(sacct -X -n -P -j "$OLD_CANDIDATE_JOB" \
  --format=JobID%30,State | awk -F'|' -v id="$OLD_CANDIDATE_JOB" \
  '{gsub(/^[[:space:]]+|[[:space:]]+$/, "", $1)}
   $1 ~ ("^" id "_[0-9]+$") {sub(/[+ ].*$/, "", $2); print $1 " " $2}')
if ((${#TASK_STATES[@]} != 20)); then
  echo "Expected 20 original candidate task states; found ${#TASK_STATES[@]}." >&2
  exit 2
fi
FAILED_TASKS=()
for record in "${TASK_STATES[@]}"; do
  TASK_JOB=${record%% *}
  TASK_STATE=${record#* }
  TASK_INDEX=${TASK_JOB##*_}
  case "$TASK_STATE" in
    COMPLETED) ;;
    FAILED) FAILED_TASKS+=("$TASK_INDEX") ;;
    *) echo "Refusing recovery with task $TASK_INDEX in state $TASK_STATE." >&2; exit 2 ;;
  esac
done
if ((${#FAILED_TASKS[@]} == 0)); then
  echo "No failed Stage 1 candidate tasks require recovery." >&2
  exit 2
fi

SMALL_GRAPH_TASKS=()
ADAPTER_CONFLICT_TASKS=()
for ARRAY_TASK in "${FAILED_TASKS[@]}"; do
  CANDIDATE_INDEX=$((ARRAY_TASK / 5))
  STAGE0_INDEX=$((ARRAY_TASK % 5))
  SAFE_PLOT_ID=$("$TREEBENCH_ENV/bin/python" "$MANIFEST_CLI" resolve-stage0 \
    --manifest-json "$MANIFEST_JSON" --stage0-index "$STAGE0_INDEX" --field safe_plot_id)
  CANDIDATE_ID=$("$TREEBENCH_ENV/bin/python" -c \
    'import sys,yaml; print(yaml.safe_load(open(sys.argv[1]))["candidates"][int(sys.argv[2])]["candidate_id"])' \
    "$STAGE1_CONFIG" "$CANDIDATE_INDEX")
  CANDIDATE_RUN_ID="${RUN_ID}__${CANDIDATE_ID}"
  PLOT_ROOT="$OUTPUT_ROOT/tls2trees/for_instance/development_tuned/development/$CANDIDATE_RUN_ID/$SAFE_PLOT_ID"
  METADATA="$PLOT_ROOT/metadata/instance_run.json"
  test -f "$METADATA"
  test ! -e "$PLOT_ROOT/evaluation"
  INSTANCE_STATUS=$("$TREEBENCH_ENV/bin/python" -c \
    'import json,sys; print(json.load(open(sys.argv[1]))["status"])' "$METADATA")
  case "$INSTANCE_STATUS" in
    failed)
      test ! -e "$PLOT_ROOT/predictions/aligned"
      "$TREEBENCH_ENV/bin/python" -c \
        'import json,sys; p=json.load(open(sys.argv[1])); assert "Instance tile" in str(p.get("error", ""))' \
        "$METADATA"
      if ! find "$PLOT_ROOT/logs/instance" -maxdepth 1 -type f -name 'tile_*.stderr.log' \
        -exec grep -lF 'Expected n_neighbors <= n_samples' {} + | grep -q .; then
        echo "Task $ARRAY_TASK is not the audited small-graph failure." >&2
        exit 2
      fi
      SMALL_GRAPH_TASKS+=("$ARRAY_TASK")
      ;;
    completed)
      PARTIAL_LEAF_OFF_METADATA="$PLOT_ROOT/predictions/aligned/leaf_off/alignment_metadata.json"
      RECOVERY_ARCHIVE_ROOT="$PLOT_ROOT/recovery/adapter_ownership_attempt_1"
      if [[ -f "$PARTIAL_LEAF_OFF_METADATA" ]]; then
        test ! -e "$PLOT_ROOT/predictions/aligned/leaf_on"
        test ! -e "$RECOVERY_ARCHIVE_ROOT"
      else
        test ! -e "$PLOT_ROOT/predictions/aligned"
        PARTIAL_LEAF_OFF_METADATA="$RECOVERY_ARCHIVE_ROOT/aligned/leaf_off/alignment_metadata.json"
      fi
      test -f "$PARTIAL_LEAF_OFF_METADATA"
      "$TREEBENCH_ENV/bin/python" -c \
        'import json,sys; p=json.load(open(sys.argv[1])); d=p["raw_alignment_diagnostics"]; assert p["target"] == "leaf_off"; assert p["status"] == "passed"; assert d["across_tree_conflicting_representative_count"] == 0' \
        "$PARTIAL_LEAF_OFF_METADATA"
      OUTER_ERR="logs/tls2trees_for_instance/tls2trees_dt_stage1_${OLD_CANDIDATE_JOB}_${ARRAY_TASK}.err"
      test -f "$OUTER_ERR"
      if ! grep -qxE \
        'ValueError: Raw prediction ownership is not unique: within_tree_duplicates=0, across_tree_conflicts=[1-9][0-9]*' \
        "$OUTER_ERR"; then
        echo "Task $ARRAY_TASK is not the audited adapter ownership conflict." >&2
        exit 2
      fi
      "$TREEBENCH_ENV/bin/python" -c \
        'import json,sys; p=json.load(open(sys.argv[1])); assert p["prediction_inventory"]["leaf_off"]; assert p["prediction_inventory"]["leaf_on"]' \
        "$METADATA"
      ADAPTER_CONFLICT_TASKS+=("$ARRAY_TASK")
      ;;
    *)
      echo "Task $ARRAY_TASK has unsupported instance status $INSTANCE_STATUS." >&2
      exit 2
      ;;
  esac
done

FAILED_LIST=$(IFS=,; echo "${FAILED_TASKS[*]}")
SMALL_GRAPH_LIST=$(IFS=,; echo "${SMALL_GRAPH_TASKS[*]}")
ADAPTER_CONFLICT_LIST=$(IFS=,; echo "${ADAPTER_CONFLICT_TASKS[*]}")
STAMP="${TLS2TREES_STAGE1_RECOVERY_STAMP:-$(date -u +%Y%m%d_%H%M%S)}"
if [[ ! "$STAMP" =~ ^[0-9]{8}_[0-9]{6}$ ]]; then
  echo "TLS2TREES_STAGE1_RECOVERY_STAMP must use YYYYMMDD_HHMMSS." >&2
  exit 2
fi
RECOVERY_SUMMARY_JSON="${OLD_SUMMARY_JSON%.json}_recovery_${STAMP}.json"
RECOVERY_PLOT_CSV="${OLD_PLOT_CSV%.csv}_recovery_${STAMP}.csv"
RECOVERY_AGGREGATE_CSV="${OLD_AGGREGATE_CSV%.csv}_recovery_${STAMP}.csv"
STATE_DIR="$HOME/fastscratch/tls2trees_for_instance_stage1_states"
STATE_FILE="$STATE_DIR/${RUN_ID}_audited_failure_recovery_${STAMP}.env"
for path in "$RECOVERY_SUMMARY_JSON" "$RECOVERY_PLOT_CSV" "$RECOVERY_AGGREGATE_CSV" "$STATE_FILE"; do
  test ! -e "$path"
done

RECOVERY_JOB="not_submitted"
SUMMARY_JOB="not_submitted"
SUBMISSION_STATUS="audited_failure_recovery_preflight_completed"
SUBMITTED_JOBS=()
write_state() {
  {
    printf 'TLS2TREES_STAGE1_RUN_ID=%q\n' "$RUN_ID"
    printf 'TLS2TREES_STAGE1_SUBMISSION_STATUS=%q\n' "$SUBMISSION_STATUS"
    printf 'TLS2TREES_STAGE1_SEMANTIC_JOB=%q\n' "${TLS2TREES_STAGE1_SEMANTIC_JOB}"
    printf 'TLS2TREES_STAGE1_CANDIDATE_JOB=%q\n' "$RECOVERY_JOB"
    printf 'TLS2TREES_STAGE1_SUMMARY_JOB=%q\n' "$SUMMARY_JOB"
    printf 'TLS2TREES_STAGE1_BENCHMARK_COMMIT=%q\n' "$BENCHMARK_COMMIT"
    printf 'TLS2TREES_STAGE1_PROBE_RUN_ID=%q\n' "$PROBE_RUN_ID"
    printf 'TLS2TREES_STAGE1_PROBE_SUMMARY_JSON=%q\n' "$PROBE_SUMMARY_JSON"
    printf 'TLS2TREES_STAGE1_PROBE_SUMMARY_SHA256=%q\n' "$PROBE_SUMMARY_SHA256"
    printf 'TLS2TREES_STAGE1_MANIFEST_JSON=%q\n' "$MANIFEST_JSON"
    printf 'TLS2TREES_STAGE1_CONFIG=%q\n' "$STAGE1_CONFIG"
    printf 'TLS2TREES_STAGE1_CONFIG_SHA256=%q\n' "$STAGE1_CONFIG_SHA256"
    printf 'TLS2TREES_STAGE1_SEMANTIC_CACHE_RUN_ID=%q\n' "$SEMANTIC_CACHE_RUN_ID"
    printf 'TLS2TREES_STAGE1_OUTPUT_ROOT=%q\n' "$OUTPUT_ROOT"
    printf 'TLS2TREES_STAGE1_SUMMARY_JSON=%q\n' "$RECOVERY_SUMMARY_JSON"
    printf 'TLS2TREES_STAGE1_PLOT_CSV=%q\n' "$RECOVERY_PLOT_CSV"
    printf 'TLS2TREES_STAGE1_AGGREGATE_CSV=%q\n' "$RECOVERY_AGGREGATE_CSV"
    printf 'TLS2TREES_STAGE1_TREEBENCH_ENV=%q\n' "$TREEBENCH_ENV"
    printf 'TLS2TREES_STAGE1_RECOVERY_FROM_JOB=%q\n' "$OLD_CANDIDATE_JOB"
    printf 'TLS2TREES_STAGE1_RECOVERY_TASKS=%q\n' "$FAILED_LIST"
  } > "$STATE_FILE"
}
cancel_partial_submission() {
  local status=$?
  if ((${#SUBMITTED_JOBS[@]})); then scancel "${SUBMITTED_JOBS[@]}" 2>/dev/null || true; fi
  SUBMISSION_STATUS="audited_failure_recovery_submission_failed_jobs_cancelled"
  write_state
  exit "$status"
}
trap cancel_partial_submission ERR

COMMON_EXPORTS="ALL,TLS2TREES_STAGE1_CONFIRMED=1,TLS2TREES_STAGE1_RECOVERY=1,TLS2TREES_PROJECT_ROOT=$PROJECT_ROOT,TLS2TREES_TREEBENCH_ENV=$TREEBENCH_ENV,TLS2TREES_METHOD_ENV=$METHOD_ENV,TLS2TREES_METHOD_ENV_MARKER=$METHOD_ENV_MARKER,TLS2TREES_METHOD_ENV_MARKER_SHA256=$METHOD_ENV_MARKER_SHA256,TLS2TREES_UPSTREAM_REPO=$UPSTREAM_REPO,TLS2TREES_EXPECTED_BENCHMARK_COMMIT=$BENCHMARK_COMMIT,TLS2TREES_MANIFEST_JSON=$MANIFEST_JSON,TLS2TREES_MANIFEST_CLI=$MANIFEST_CLI,TLS2TREES_CANDIDATE_CLI=$CANDIDATE_CLI,TLS2TREES_ADAPTER_CLI=$ADAPTER_CLI,TLS2TREES_EVALUATE_CLI=$EVALUATE_CLI,TLS2TREES_ENV_VALIDATOR=$ENV_VALIDATOR,TLS2TREES_STAGE1_CONFIG=$STAGE1_CONFIG,TLS2TREES_STAGE1_CONFIG_SHA256=$STAGE1_CONFIG_SHA256,TLS2TREES_PROBE_SUMMARY_JSON=$PROBE_SUMMARY_JSON,TLS2TREES_PROBE_SUMMARY_SHA256=$PROBE_SUMMARY_SHA256,TLS2TREES_OUTPUT_ROOT=$OUTPUT_ROOT,TLS2TREES_STAGE1_RUN_ID=$RUN_ID,TLS2TREES_SEMANTIC_CACHE_RUN_ID=$SEMANTIC_CACHE_RUN_ID,TLS2TREES_STAGE1_SUMMARY_JSON=$RECOVERY_SUMMARY_JSON,TLS2TREES_STAGE1_PLOT_CSV=$RECOVERY_PLOT_CSV,TLS2TREES_STAGE1_AGGREGATE_CSV=$RECOVERY_AGGREGATE_CSV"

RECOVERY_JOB=$(sbatch --parsable --array="$FAILED_LIST%2" --export="$COMMON_EXPORTS" \
  methods/tls2trees/slurm/for_instance/evaluate_development_stage1_candidate.sbatch)
SUBMITTED_JOBS+=("$RECOVERY_JOB")
SUBMISSION_STATUS="audited_failure_recovery_array_submitted"
write_state
SUMMARY_JOB=$(sbatch --parsable --dependency="afterany:$RECOVERY_JOB" \
  --kill-on-invalid-dep=yes --export="$COMMON_EXPORTS" \
  methods/tls2trees/slurm/for_instance/summarise_development_stage1.sbatch)
SUBMITTED_JOBS+=("$SUMMARY_JOB")
SUBMISSION_STATUS="audited_failure_recovery_chain_submitted"
write_state
printf '%s\n' "$STATE_FILE" > logs/tls2trees_for_instance/latest_stage1_state_file.txt
trap - ERR

echo "run_id=$RUN_ID"
echo "recovery_from_candidate_job=$OLD_CANDIDATE_JOB"
echo "recovery_tasks=$FAILED_LIST"
echo "small_graph_instance_tasks=$SMALL_GRAPH_LIST"
echo "adapter_ownership_tasks=$ADAPTER_CONFLICT_LIST"
echo "recovery_job=$RECOVERY_JOB summary_job=$SUMMARY_JOB"
echo "state_file=$STATE_FILE"
echo "completed_tasks_and_semantic_cache_reused=true"
echo "held_out_test_accessed=false"
