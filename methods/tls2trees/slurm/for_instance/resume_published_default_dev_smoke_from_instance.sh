#!/usr/bin/env bash

set -euo pipefail

if [[ "${TLS2TREES_INSTANCE_RECOVERY_CONFIRMED:-0}" != "1" ]]; then
  echo "Refusing TLS2trees instance recovery." >&2
  echo "Set TLS2TREES_INSTANCE_RECOVERY_CONFIRMED=1 after reviewing the failed tile log." >&2
  exit 2
fi

ORIGINAL_STATE_FILE="${1:?Usage: resume_published_default_dev_smoke_from_instance.sh <state-file>}"
test -f "$ORIGINAL_STATE_FILE"
# shellcheck disable=SC1090
source "$ORIGINAL_STATE_FILE"

PROJECT_ROOT="${TLS2TREES_PROJECT_ROOT:-$HOME/scratch/tree-seg-benchmark}"
TREEBENCH_ENV="${TLS2TREES_TREEBENCH_ENV:-$HOME/fastscratch/venvs/treebench}"
METHOD_ENV="${TLS2TREES_SMOKE_METHOD_ENV:?Original state has no method environment}"
UPSTREAM_REPO="${TLS2TREES_UPSTREAM_REPO:-$PROJECT_ROOT/external/TLS2trees}"
DATASET_ROOT="${TLS2TREES_DATASET_ROOT:-$HOME/data/datasets/for_instance/FORinstance_dataset}"
RUN_ID="${TLS2TREES_SMOKE_RUN_ID:?Original state has no run ID}"
MANIFEST_JSON="${TLS2TREES_SMOKE_MANIFEST_JSON:?Original state has no manifest}"
STAGE0_INDEX="${TLS2TREES_SMOKE_STAGE0_INDEX:?Original state has no Stage 0 index}"
OUTPUT_ROOT="${TLS2TREES_SMOKE_OUTPUT_ROOT:?Original state has no output root}"
WORKFLOW_METADATA_ROOT="${TLS2TREES_SMOKE_WORKFLOW_METADATA_ROOT:?Original state has no metadata root}"
WORKFLOW_TABLE_ROOT="${TLS2TREES_SMOKE_WORKFLOW_TABLE_ROOT:?Original state has no table root}"
RUNTIME_ROOT="$OUTPUT_ROOT/tls2trees/for_instance/published_default/development/$RUN_ID"
OLD_BENCHMARK_COMMIT="${TLS2TREES_SMOKE_BENCHMARK_COMMIT:?Original state has no benchmark commit}"
OLD_INSTANCE_JOB="${TLS2TREES_SMOKE_INSTANCE_JOB:?Original state has no instance job}"
METHOD_ENV_MARKER="$METHOD_ENV/.tls2trees_setup_complete.json"
EXPECTED_MARKER_SHA256="${TLS2TREES_SMOKE_METHOD_ENV_MARKER_SHA256:?Original state has no marker hash}"

cd "$PROJECT_ROOT"
test -d .git
test -z "$(git status --porcelain)"
CURRENT_BENCHMARK_COMMIT=$(git rev-parse HEAD)
test "$CURRENT_BENCHMARK_COMMIT" != "$OLD_BENCHMARK_COMMIT"
test -x "$TREEBENCH_ENV/bin/python"
test -x "$METHOD_ENV/bin/python"
test -d "$UPSTREAM_REPO/.git"
test -d "$DATASET_ROOT"
test -f "$MANIFEST_JSON"
test -f "$METHOD_ENV_MARKER"
test "$(sha256sum "$METHOD_ENV_MARKER" | awk '{print $1}')" = "$EXPECTED_MARKER_SHA256"
test "$(git -C "$UPSTREAM_REPO" rev-parse HEAD)" = \
  "ca12cb73b2c736d80b020e8025f8d975d42e6f01"
test -z "$(git -C "$UPSTREAM_REPO" status --porcelain)"

OLD_INSTANCE_STATE=$(sacct -X -n -P -j "$OLD_INSTANCE_JOB" \
  --format=JobIDRaw,State | awk -F'|' -v id="$OLD_INSTANCE_JOB" \
  '$1 == id {sub(/[+ ].*$/, "", $2); print $2; exit}')
if [[ "$OLD_INSTANCE_STATE" != "FAILED" ]]; then
  echo "Original instance job is not FAILED: ${OLD_INSTANCE_STATE:-unknown}" >&2
  exit 2
fi

MANIFEST_CLI="methods/tls2trees/scripts/data/prepare_for_instance_manifest.py"
CONVERT_CLI="methods/tls2trees/scripts/data/prepare_for_instance_tls2trees_input.py"
SEMANTIC_CLI="methods/tls2trees/scripts/runtime/run_for_instance_tls2trees_semantic.py"
INSTANCE_CLI="methods/tls2trees/scripts/runtime/run_for_instance_tls2trees_instance.py"
ADAPTER_CLI="methods/tls2trees/scripts/evaluation/adapt_for_instance_tls2trees_predictions.py"
EVALUATE_CLI="methods/tls2trees/scripts/evaluation/evaluate_for_instance_tls2trees_plot.py"
GATE_CLI="methods/tls2trees/scripts/evaluation/validate_for_instance_tls2trees_smoke.py"
ENV_VALIDATOR="methods/tls2trees/scripts/runtime/validate_tls2trees_environment.py"

TASK_INDEX=$("$TREEBENCH_ENV/bin/python" "$MANIFEST_CLI" resolve-stage0 \
  --manifest-json "$MANIFEST_JSON" --stage0-index "$STAGE0_INDEX" --field task_index)
SAFE_PLOT_ID=$("$TREEBENCH_ENV/bin/python" "$MANIFEST_CLI" resolve-stage0 \
  --manifest-json "$MANIFEST_JSON" --stage0-index "$STAGE0_INDEX" --field safe_plot_id)
PLOT_ROOT="$RUNTIME_ROOT/$SAFE_PLOT_ID"
FAILED_METADATA="$PLOT_ROOT/metadata/instance_run.json"
FAILED_RAW_ROOT="$PLOT_ROOT/predictions/raw"
test -f "$PLOT_ROOT/metadata/semantic_run.json"
test -f "$FAILED_METADATA"
test -d "$FAILED_RAW_ROOT"
FAILED_TILE_ERROR=$(find "$PLOT_ROOT/logs/instance" -maxdepth 1 -type f \
  -name 'tile_*.stderr.log' -print -quit)
test -n "$FAILED_TILE_ERROR"
grep -Fq "sources must not be empty" "$FAILED_TILE_ERROR"
test ! -e "$PLOT_ROOT/recovery/instance_failed_attempt_1"
test ! -e "$PLOT_ROOT/predictions/aligned"
test ! -e "$PLOT_ROOT/metadata/adapter_run.json"
if find "$FAILED_RAW_ROOT" -type f -print -quit | grep -q .; then
  echo "Refusing recovery because the failed raw root contains files." >&2
  exit 2
fi
"$TREEBENCH_ENV/bin/python" -c \
  'import json,sys; p=json.load(open(sys.argv[1])); assert p.get("status") == "failed"; assert "Instance tile" in str(p.get("error", ""))' \
  "$FAILED_METADATA"

STAMP="${TLS2TREES_RECOVERY_STAMP:-$(date -u +%Y%m%d_%H%M%S)}"
if [[ ! "$STAMP" =~ ^[0-9]{8}_[0-9]{6}$ ]]; then
  echo "TLS2TREES_RECOVERY_STAMP must use YYYYMMDD_HHMMSS." >&2
  exit 2
fi
STATE_DIR="$HOME/fastscratch/tls2trees_for_instance_smoke_states"
STATE_FILE="$STATE_DIR/${RUN_ID}_instance_recovery_${STAMP}.env"
test ! -e "$STATE_FILE"
mkdir -p logs/tls2trees_for_instance "$STATE_DIR"

INVENTORY_JOB="$TLS2TREES_SMOKE_INVENTORY_JOB"
CONVERT_JOB="$TLS2TREES_SMOKE_CONVERT_JOB"
SEMANTIC_JOB="$TLS2TREES_SMOKE_SEMANTIC_JOB"
INSTANCE_JOB="not_submitted"
ADAPTER_JOB="not_submitted"
LEAF_OFF_EVALUATE_JOB="not_submitted"
LEAF_ON_EVALUATE_JOB="not_submitted"
GATE_JOB="not_submitted"
SUMMARY_JOB="not_submitted"
SUBMISSION_STATUS="instance_recovery_preflight_completed"
SUBMITTED_JOBS=()

write_state() {
  {
    printf 'TLS2TREES_SMOKE_RUN_ID=%q\n' "$RUN_ID"
    printf 'TLS2TREES_SMOKE_SUBMISSION_STATUS=%q\n' "$SUBMISSION_STATUS"
    printf 'TLS2TREES_SMOKE_INVENTORY_JOB=%q\n' "$INVENTORY_JOB"
    printf 'TLS2TREES_SMOKE_CONVERT_JOB=%q\n' "$CONVERT_JOB"
    printf 'TLS2TREES_SMOKE_SEMANTIC_JOB=%q\n' "$SEMANTIC_JOB"
    printf 'TLS2TREES_SMOKE_INSTANCE_JOB=%q\n' "$INSTANCE_JOB"
    printf 'TLS2TREES_SMOKE_ADAPTER_JOB=%q\n' "$ADAPTER_JOB"
    printf 'TLS2TREES_SMOKE_LEAF_OFF_EVALUATE_JOB=%q\n' "$LEAF_OFF_EVALUATE_JOB"
    printf 'TLS2TREES_SMOKE_LEAF_ON_EVALUATE_JOB=%q\n' "$LEAF_ON_EVALUATE_JOB"
    printf 'TLS2TREES_SMOKE_GATE_JOB=%q\n' "$GATE_JOB"
    printf 'TLS2TREES_SMOKE_SUMMARY_JOB=%q\n' "$SUMMARY_JOB"
    printf 'TLS2TREES_SMOKE_BENCHMARK_COMMIT=%q\n' "$CURRENT_BENCHMARK_COMMIT"
    printf 'TLS2TREES_SMOKE_RECOVERY_FROM_BENCHMARK_COMMIT=%q\n' "$OLD_BENCHMARK_COMMIT"
    printf 'TLS2TREES_SMOKE_RECOVERY_FROM_INSTANCE_JOB=%q\n' "$OLD_INSTANCE_JOB"
    printf 'TLS2TREES_SMOKE_UPSTREAM_COMMIT=%q\n' "ca12cb73b2c736d80b020e8025f8d975d42e6f01"
    printf 'TLS2TREES_SMOKE_MODEL_SHA256=%q\n' "1a8bb6372394600f7c4b15f76beb98c32cb47ed25f8f729a84117ccfa410e72b"
    printf 'TLS2TREES_SMOKE_METHOD_ENV=%q\n' "$METHOD_ENV"
    printf 'TLS2TREES_SMOKE_METHOD_ENV_MARKER_SHA256=%q\n' "$EXPECTED_MARKER_SHA256"
    printf 'TLS2TREES_SMOKE_MANIFEST_JSON=%q\n' "$MANIFEST_JSON"
    printf 'TLS2TREES_SMOKE_STAGE0_INDEX=%q\n' "$STAGE0_INDEX"
    printf 'TLS2TREES_SMOKE_OUTPUT_ROOT=%q\n' "$OUTPUT_ROOT"
    printf 'TLS2TREES_SMOKE_WORKFLOW_METADATA_ROOT=%q\n' "$WORKFLOW_METADATA_ROOT"
    printf 'TLS2TREES_SMOKE_WORKFLOW_TABLE_ROOT=%q\n' "$WORKFLOW_TABLE_ROOT"
    printf 'TLS2TREES_SMOKE_ORIGINAL_STATE_FILE=%q\n' "$ORIGINAL_STATE_FILE"
  } > "$STATE_FILE"
}

cancel_partial_submission() {
  local status=$?
  if ((${#SUBMITTED_JOBS[@]})); then
    scancel "${SUBMITTED_JOBS[@]}" 2>/dev/null || true
  fi
  SUBMISSION_STATUS="instance_recovery_submission_failed_jobs_cancelled"
  write_state
  echo "Recovery submission failed; new jobs were cancelled." >&2
  echo "state_file=$STATE_FILE" >&2
  exit "$status"
}
trap cancel_partial_submission ERR

COMMON_EXPORTS="ALL,TLS2TREES_DEV_SMOKE_CONFIRMED=1,TLS2TREES_REQUESTED_VARIANT=published_default,TLS2TREES_REQUESTED_SPLIT=development,TLS2TREES_PROJECT_ROOT=$PROJECT_ROOT,TLS2TREES_TREEBENCH_ENV=$TREEBENCH_ENV,TLS2TREES_METHOD_ENV=$METHOD_ENV,TLS2TREES_EXPECTED_METHOD_ENV_MARKER_SHA256=$EXPECTED_MARKER_SHA256,TLS2TREES_UPSTREAM_REPO=$UPSTREAM_REPO,TLS2TREES_DATASET_ROOT=$DATASET_ROOT,TLS2TREES_RUN_ID=$RUN_ID,TLS2TREES_EXPECTED_BENCHMARK_COMMIT=$CURRENT_BENCHMARK_COMMIT,TLS2TREES_RUNTIME_ROOT=$RUNTIME_ROOT,TLS2TREES_WORKFLOW_METADATA_ROOT=$WORKFLOW_METADATA_ROOT,TLS2TREES_WORKFLOW_TABLE_ROOT=$WORKFLOW_TABLE_ROOT,TLS2TREES_MANIFEST_CLI=$MANIFEST_CLI,TLS2TREES_CONVERT_CLI=$CONVERT_CLI,TLS2TREES_SEMANTIC_CLI=$SEMANTIC_CLI,TLS2TREES_INSTANCE_CLI=$INSTANCE_CLI,TLS2TREES_ADAPTER_CLI=$ADAPTER_CLI,TLS2TREES_EVALUATE_CLI=$EVALUATE_CLI,TLS2TREES_GATE_CLI=$GATE_CLI,TLS2TREES_ENV_VALIDATOR=$ENV_VALIDATOR,MANIFEST_JSON=$MANIFEST_JSON,STAGE0_INDEX=$STAGE0_INDEX,RUN_ID=$RUN_ID,OUTPUT_ROOT=$OUTPUT_ROOT,TLS2TREES_REPO=$UPSTREAM_REPO,TLS2TREES_RECOVERY_FROM_BENCHMARK_COMMIT=$OLD_BENCHMARK_COMMIT"

INSTANCE_JOB=$(sbatch --parsable \
  --time=00:30:00 \
  --cpus-per-task=4 \
  --mem=16G \
  --export="$COMMON_EXPORTS,TLS2TREES_INSTANCE_RECOVERY_CONFIRMED=1" \
  methods/tls2trees/slurm/for_instance/instance_published_default_dev_smoke.sbatch)
SUBMITTED_JOBS+=("$INSTANCE_JOB")
SUBMISSION_STATUS="recovery_instance_submitted"
write_state

ADAPTER_JOB=$(sbatch --parsable --dependency="afterok:$INSTANCE_JOB" \
  --kill-on-invalid-dep=yes --export="$COMMON_EXPORTS" \
  methods/tls2trees/slurm/for_instance/adapt_published_default_dev_smoke.sbatch)
SUBMITTED_JOBS+=("$ADAPTER_JOB")

LEAF_OFF_EVALUATE_JOB=$(sbatch --parsable --dependency="afterok:$ADAPTER_JOB" \
  --kill-on-invalid-dep=yes --export="$COMMON_EXPORTS,TLS2TREES_TARGET=leaf_off" \
  methods/tls2trees/slurm/for_instance/evaluate_published_default_dev_smoke.sbatch)
SUBMITTED_JOBS+=("$LEAF_OFF_EVALUATE_JOB")

LEAF_ON_EVALUATE_JOB=$(sbatch --parsable --dependency="afterok:$ADAPTER_JOB" \
  --kill-on-invalid-dep=yes --export="$COMMON_EXPORTS,TLS2TREES_TARGET=leaf_on" \
  methods/tls2trees/slurm/for_instance/evaluate_published_default_dev_smoke.sbatch)
SUBMITTED_JOBS+=("$LEAF_ON_EVALUATE_JOB")

GATE_JOB=$(sbatch --parsable \
  --dependency="afterok:$LEAF_OFF_EVALUATE_JOB:$LEAF_ON_EVALUATE_JOB" \
  --kill-on-invalid-dep=yes --export="$COMMON_EXPORTS" \
  methods/tls2trees/slurm/for_instance/gate_published_default_dev_smoke.sbatch)
SUBMITTED_JOBS+=("$GATE_JOB")

SUMMARY_JOB=$(sbatch --parsable --dependency="afterok:$GATE_JOB" \
  --kill-on-invalid-dep=yes --export="$COMMON_EXPORTS" \
  methods/tls2trees/slurm/for_instance/summarise_published_default_dev_smoke.sbatch)
SUBMITTED_JOBS+=("$SUMMARY_JOB")
SUBMISSION_STATUS="instance_recovery_chain_submitted"
write_state
trap - ERR

echo "run_id=$RUN_ID"
echo "recovery_from_instance_job=$OLD_INSTANCE_JOB"
echo "instance_job=$INSTANCE_JOB adapter_job=$ADAPTER_JOB"
echo "leaf_off_evaluation_job=$LEAF_OFF_EVALUATE_JOB leaf_on_evaluation_job=$LEAF_ON_EVALUATE_JOB"
echo "gate_job=$GATE_JOB summary_job=$SUMMARY_JOB"
echo "state_file=$STATE_FILE"
echo "No semantic, tuning, full-development or held-out-test job was submitted."
