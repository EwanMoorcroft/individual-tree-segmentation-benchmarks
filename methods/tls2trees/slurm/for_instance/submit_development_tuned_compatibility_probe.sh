#!/usr/bin/env bash

set -euo pipefail

if [[ "${TLS2TREES_DEV_TUNED_PROBE_CONFIRMED:-0}" != "1" ]]; then
  echo "Refusing development-tuned compatibility probe." >&2
  echo "Set TLS2TREES_DEV_TUNED_PROBE_CONFIRMED=1 to run the frozen development-only candidates." >&2
  exit 2
fi

SOURCE_STATE_FILE="${1:?Usage: submit_development_tuned_compatibility_probe.sh <published-smoke-state-file>}"
test -f "$SOURCE_STATE_FILE"
# shellcheck disable=SC1090
source "$SOURCE_STATE_FILE"

PROJECT_ROOT="${TLS2TREES_PROJECT_ROOT:-$HOME/scratch/tree-seg-benchmark}"
TREEBENCH_ENV="${TLS2TREES_TREEBENCH_ENV:-$HOME/fastscratch/venvs/treebench}"
METHOD_ENV="${TLS2TREES_SMOKE_METHOD_ENV:-$HOME/fastscratch/venvs/tls2trees}"
UPSTREAM_REPO="${TLS2TREES_UPSTREAM_REPO:-$PROJECT_ROOT/external/TLS2trees}"
OUTPUT_ROOT="${TLS2TREES_SMOKE_OUTPUT_ROOT:?Source state has no output root}"
SOURCE_RUN_ID="${TLS2TREES_SMOKE_RUN_ID:?Source state has no run ID}"
SOURCE_MANIFEST="${TLS2TREES_SMOKE_MANIFEST_JSON:?Source state has no manifest}"
SOURCE_STAGE0_INDEX="${TLS2TREES_SMOKE_STAGE0_INDEX:?Source state has no Stage 0 index}"
METHOD_ENV_MARKER="$METHOD_ENV/.tls2trees_setup_complete.json"
EXPECTED_MARKER_SHA256="${TLS2TREES_SMOKE_METHOD_ENV_MARKER_SHA256:?Source state has no environment marker hash}"
CANDIDATE_MANIFEST="methods/tls2trees/configs/for_instance_development_tuned_compatibility_probe.yml"
RUNNER="methods/tls2trees/scripts/runtime/run_for_instance_tls2trees_compatibility_probe.py"
SUMMARISER="methods/tls2trees/scripts/evaluation/summarise_tls2trees_compatibility_probe.py"

cd "$PROJECT_ROOT"
test -d .git
test -z "$(git status --porcelain)"
BENCHMARK_COMMIT=$(git rev-parse HEAD)
test -x "$TREEBENCH_ENV/bin/python"
test -x "$METHOD_ENV/bin/python"
test -d "$UPSTREAM_REPO/.git"
test "$(git -C "$UPSTREAM_REPO" rev-parse HEAD)" = \
  "ca12cb73b2c736d80b020e8025f8d975d42e6f01"
test -z "$(git -C "$UPSTREAM_REPO" status --porcelain)"
test -f "$SOURCE_MANIFEST"
test -f "$METHOD_ENV_MARKER"
test "$(sha256sum "$METHOD_ENV_MARKER" | awk '{print $1}')" = "$EXPECTED_MARKER_SHA256"
for path in "$CANDIDATE_MANIFEST" "$RUNNER" "$SUMMARISER"; do
  test -f "$path"
done

MANIFEST_CLI="methods/tls2trees/scripts/data/prepare_for_instance_manifest.py"
SAFE_PLOT_ID=$("$TREEBENCH_ENV/bin/python" "$MANIFEST_CLI" resolve-stage0 \
  --manifest-json "$SOURCE_MANIFEST" \
  --stage0-index "$SOURCE_STAGE0_INDEX" \
  --field safe_plot_id)
SOURCE_PLOT_ROOT="$OUTPUT_ROOT/tls2trees/for_instance/published_default/development/$SOURCE_RUN_ID/$SAFE_PLOT_ID"
SOURCE_SEMANTIC="$SOURCE_PLOT_ROOT/metadata/semantic_run.json"
SOURCE_INSTANCE="$SOURCE_PLOT_ROOT/metadata/instance_run.json"
test -f "$SOURCE_SEMANTIC"
test -f "$SOURCE_INSTANCE"
"$TREEBENCH_ENV/bin/python" -c '
import json,sys
semantic=json.load(open(sys.argv[1]))
instance=json.load(open(sys.argv[2]))
assert semantic.get("status") == "completed"
assert semantic.get("variant") == "published_default"
assert semantic.get("split") == "development"
assert semantic.get("held_out_test_accessed") is False
assert instance.get("status") == "completed_no_predictions"
assert not instance.get("prediction_inventory", {}).get("leaf_off", [])
' "$SOURCE_SEMANTIC" "$SOURCE_INSTANCE"

CANDIDATE_COUNT=$("$TREEBENCH_ENV/bin/python" -c '
import sys,yaml
p=yaml.safe_load(open(sys.argv[1]))
s=p["scope"]
assert s["variant"] == "development_tuned" and s["split"] == "development"
assert s["held_out_test_accessed"] is False
assert s["reference_labels_accessed"] is False
assert s["accuracy_metrics_accessed"] is False
assert s["selection_uses_accuracy_metrics"] is False
assert p["candidate_generation"]["ordering_frozen"] is True
assert [c["candidate_index"] for c in p["candidates"]] == list(range(len(p["candidates"])))
print(len(p["candidates"]))
' "$CANDIDATE_MANIFEST")
if [[ ! "$CANDIDATE_COUNT" =~ ^[1-9][0-9]*$ ]]; then
  echo "Invalid frozen candidate count: $CANDIDATE_COUNT" >&2
  exit 2
fi
CANDIDATE_MANIFEST_SHA256=$(sha256sum "$CANDIDATE_MANIFEST" | awk '{print $1}')

STAMP="${TLS2TREES_PROBE_STAMP:-$(date -u +%Y%m%d_%H%M%S)}"
if [[ ! "$STAMP" =~ ^[0-9]{8}_[0-9]{6}$ ]]; then
  echo "TLS2TREES_PROBE_STAMP must use YYYYMMDD_HHMMSS." >&2
  exit 2
fi
RUN_ID="tls2trees_for-instance_development_tuned_compatibility_probe_$STAMP"
RUN_ROOT="$OUTPUT_ROOT/tls2trees/for_instance/development_tuned/development/$RUN_ID"
WORKFLOW_ROOT="$PROJECT_ROOT/results/metadata/tls2trees/for_instance/development_tuned/compatibility_probe/$RUN_ID"
TABLE_ROOT="$PROJECT_ROOT/results/tables/tls2trees/for_instance/development_tuned/compatibility_probe/$RUN_ID"
SUMMARY_JSON="$WORKFLOW_ROOT/probe_summary.json"
SUMMARY_CSV="$TABLE_ROOT/probe_candidates.csv"
STATE_DIR="$HOME/fastscratch/tls2trees_for_instance_probe_states"
STATE_FILE="$STATE_DIR/$RUN_ID.env"
for path in "$RUN_ROOT" "$WORKFLOW_ROOT" "$TABLE_ROOT" "$STATE_FILE"; do
  if [[ -e "$path" ]]; then
    echo "Refusing existing probe path: $path" >&2
    exit 2
  fi
done
mkdir -p logs/tls2trees_for_instance "$STATE_DIR"
mkdir -p "$RUN_ROOT"

PROBE_JOB="not_submitted"
SUMMARY_JOB="not_submitted"
SUBMISSION_STATUS="preflight_completed"
SUBMITTED_JOBS=()

write_state() {
  {
    printf 'TLS2TREES_PROBE_RUN_ID=%q\n' "$RUN_ID"
    printf 'TLS2TREES_PROBE_SUBMISSION_STATUS=%q\n' "$SUBMISSION_STATUS"
    printf 'TLS2TREES_PROBE_ARRAY_JOB=%q\n' "$PROBE_JOB"
    printf 'TLS2TREES_PROBE_SUMMARY_JOB=%q\n' "$SUMMARY_JOB"
    printf 'TLS2TREES_PROBE_BENCHMARK_COMMIT=%q\n' "$BENCHMARK_COMMIT"
    printf 'TLS2TREES_PROBE_CANDIDATE_COUNT=%q\n' "$CANDIDATE_COUNT"
    printf 'TLS2TREES_PROBE_CANDIDATE_MANIFEST=%q\n' "$CANDIDATE_MANIFEST"
    printf 'TLS2TREES_PROBE_CANDIDATE_MANIFEST_SHA256=%q\n' "$CANDIDATE_MANIFEST_SHA256"
    printf 'TLS2TREES_PROBE_SOURCE_STATE_FILE=%q\n' "$SOURCE_STATE_FILE"
    printf 'TLS2TREES_PROBE_SOURCE_RUN_ID=%q\n' "$SOURCE_RUN_ID"
    printf 'TLS2TREES_PROBE_SOURCE_PLOT_ROOT=%q\n' "$SOURCE_PLOT_ROOT"
    printf 'TLS2TREES_PROBE_OUTPUT_ROOT=%q\n' "$OUTPUT_ROOT"
    printf 'TLS2TREES_PROBE_RUN_ROOT=%q\n' "$RUN_ROOT"
    printf 'TLS2TREES_PROBE_SUMMARY_JSON=%q\n' "$SUMMARY_JSON"
    printf 'TLS2TREES_PROBE_SUMMARY_CSV=%q\n' "$SUMMARY_CSV"
    printf 'TLS2TREES_PROBE_TREEBENCH_ENV=%q\n' "$TREEBENCH_ENV"
    printf 'TLS2TREES_PROBE_METHOD_ENV=%q\n' "$METHOD_ENV"
    printf 'TLS2TREES_PROBE_METHOD_ENV_MARKER_SHA256=%q\n' "$EXPECTED_MARKER_SHA256"
  } > "$STATE_FILE"
}

cancel_partial_submission() {
  local status=$?
  if ((${#SUBMITTED_JOBS[@]})); then
    scancel "${SUBMITTED_JOBS[@]}" 2>/dev/null || true
  fi
  SUBMISSION_STATUS="submission_failed_jobs_cancelled"
  write_state
  echo "Probe submission failed; new jobs were cancelled." >&2
  echo "state_file=$STATE_FILE" >&2
  exit "$status"
}
trap cancel_partial_submission ERR

COMMON_EXPORTS="ALL,TLS2TREES_DEV_TUNED_PROBE_CONFIRMED=1,TLS2TREES_REQUESTED_VARIANT=development_tuned,TLS2TREES_REQUESTED_SPLIT=development,TLS2TREES_PROJECT_ROOT=$PROJECT_ROOT,TLS2TREES_TREEBENCH_ENV=$TREEBENCH_ENV,TLS2TREES_METHOD_ENV=$METHOD_ENV,TLS2TREES_METHOD_ENV_MARKER=$METHOD_ENV_MARKER,TLS2TREES_METHOD_ENV_MARKER_SHA256=$EXPECTED_MARKER_SHA256,TLS2TREES_UPSTREAM_REPO=$UPSTREAM_REPO,TLS2TREES_EXPECTED_BENCHMARK_COMMIT=$BENCHMARK_COMMIT,TLS2TREES_SOURCE_PLOT_ROOT=$SOURCE_PLOT_ROOT,TLS2TREES_OUTPUT_ROOT=$OUTPUT_ROOT,TLS2TREES_PROBE_RUN_ID=$RUN_ID,TLS2TREES_PROBE_RUN_ROOT=$RUN_ROOT,TLS2TREES_CANDIDATE_MANIFEST=$CANDIDATE_MANIFEST,TLS2TREES_CANDIDATE_MANIFEST_SHA256=$CANDIDATE_MANIFEST_SHA256,TLS2TREES_PROBE_SUMMARY_JSON=$SUMMARY_JSON,TLS2TREES_PROBE_SUMMARY_CSV=$SUMMARY_CSV"

PROBE_JOB=$(sbatch --parsable \
  --array="0-$((CANDIDATE_COUNT - 1))%2" \
  --export="$COMMON_EXPORTS" \
  methods/tls2trees/slurm/for_instance/run_development_tuned_compatibility_probe.sbatch)
SUBMITTED_JOBS+=("$PROBE_JOB")
SUBMISSION_STATUS="candidate_array_submitted"
write_state

SUMMARY_JOB=$(sbatch --parsable \
  --dependency="afterany:$PROBE_JOB" \
  --kill-on-invalid-dep=yes \
  --export="$COMMON_EXPORTS" \
  methods/tls2trees/slurm/for_instance/summarise_development_tuned_compatibility_probe.sbatch)
SUBMITTED_JOBS+=("$SUMMARY_JOB")
SUBMISSION_STATUS="probe_and_summary_submitted"
write_state
printf '%s\n' "$STATE_FILE" > logs/tls2trees_for_instance/latest_probe_state_file.txt
trap - ERR

echo "run_id=$RUN_ID"
echo "source_run_id=$SOURCE_RUN_ID"
echo "candidate_array_job=$PROBE_JOB candidate_count=$CANDIDATE_COUNT concurrency=2"
echo "summary_job=$SUMMARY_JOB"
echo "state_file=$STATE_FILE"
echo "reference_labels_accessed=false"
echo "accuracy_metrics_accessed=false"
echo "held_out_test_accessed=false"
echo "No reference labels, accuracy metrics, full-development array, or held-out-test data are used."
