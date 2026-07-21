#!/usr/bin/env bash

set -euo pipefail

if [[ "${TLS2TREES_PUBLISHED_DEFAULT_TEST_CONFIRMED:-0}" != "1" ]]; then
  echo "Refusing TLS2trees published-default held-out submission." >&2
  echo "Set TLS2TREES_PUBLISHED_DEFAULT_TEST_CONFIRMED=1 after reviewing the fixed config." >&2
  exit 2
fi
REVIEWED_CONFIG_SHA256="${TLS2TREES_REVIEWED_PUBLISHED_DEFAULT_CONFIG_SHA256:?Set the reviewed published-default config SHA-256}"
CACHE_STATE_FILE="${1:-}"

PROJECT_ROOT="${TLS2TREES_PROJECT_ROOT:-$HOME/scratch/tree-seg-benchmark}"
TREEBENCH_ENV="${TLS2TREES_TREEBENCH_ENV:-$HOME/fastscratch/venvs/treebench}"
METHOD_ENV="${TLS2TREES_METHOD_ENV:-$HOME/fastscratch/venvs/tls2trees}"
METHOD_ENV_MARKER="$METHOD_ENV/.tls2trees_setup_complete.json"
UPSTREAM_REPO="${TLS2TREES_UPSTREAM_REPO:-$PROJECT_ROOT/external/TLS2trees}"
DATASET_ROOT="${TLS2TREES_FOR_INSTANCE_ROOT:-$HOME/data/datasets/for_instance/FORinstance_dataset}"
SPLIT_METADATA_CSV="$DATASET_ROOT/data_split_metadata.csv"
OUTPUT_ROOT="${TLS2TREES_OUTPUT_ROOT:-$PROJECT_ROOT/data/predictions}"
WORKFLOW_CONFIG="methods/tls2trees/configs/for_instance_published_default_test.yml"
PUBLISHED_CONFIG="methods/tls2trees/configs/for_instance_published_default.yml"
BENCHMARK_CONFIG="methods/tls2trees/configs/for_instance_benchmark.yml"
MANIFEST_CLI="methods/tls2trees/scripts/data/prepare_for_instance_manifest.py"
CONVERT_CLI="methods/tls2trees/scripts/data/prepare_for_instance_tls2trees_input.py"
SEMANTIC_CLI="methods/tls2trees/scripts/runtime/run_for_instance_tls2trees_semantic.py"
INSTANCE_CLI="methods/tls2trees/scripts/runtime/run_for_instance_tls2trees_instance.py"
ADAPTER_CLI="methods/tls2trees/scripts/evaluation/adapt_for_instance_tls2trees_predictions.py"
EVALUATE_CLI="methods/tls2trees/scripts/evaluation/evaluate_for_instance_tls2trees_plot.py"
ENV_VALIDATOR="methods/tls2trees/scripts/runtime/validate_tls2trees_environment.py"
CACHE_HELPER="methods/tls2trees/scripts/runtime/prepare_published_default_semantic_cache.py"
SUMMARY_CLI="methods/tls2trees/scripts/evaluation/summarise_tls2trees_published_default_test.py"

cd "$PROJECT_ROOT"
test -d .git
test -z "$(git status --porcelain)"
BENCHMARK_COMMIT=$(git rev-parse HEAD)
test -x "$TREEBENCH_ENV/bin/python"
test -x "$METHOD_ENV/bin/python"
test -f "$METHOD_ENV_MARKER"
test -f "$SPLIT_METADATA_CSV"
test -d "$UPSTREAM_REPO/.git"
test -z "$(git -C "$UPSTREAM_REPO" status --porcelain)"
UPSTREAM_COMMIT=$(git -C "$UPSTREAM_REPO" rev-parse HEAD)
test "$UPSTREAM_COMMIT" = "ca12cb73b2c736d80b020e8025f8d975d42e6f01"
MODEL="$UPSTREAM_REPO/tls2trees/fsct/model/model.pth"
test -f "$MODEL"
MODEL_SHA256=$(sha256sum "$MODEL" | awk '{print $1}')
test "$MODEL_SHA256" = "1a8bb6372394600f7c4b15f76beb98c32cb47ed25f8f729a84117ccfa410e72b"
export PYTHONNOUSERSITE=1
export LD_LIBRARY_PATH="$METHOD_ENV/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
for path in "$WORKFLOW_CONFIG" "$PUBLISHED_CONFIG" "$BENCHMARK_CONFIG" \
  "$MANIFEST_CLI" "$CONVERT_CLI" "$SEMANTIC_CLI" "$INSTANCE_CLI" \
  "$ADAPTER_CLI" "$EVALUATE_CLI" "$ENV_VALIDATOR" "$CACHE_HELPER" \
  "$SUMMARY_CLI" \
  methods/tls2trees/slurm/for_instance/published_default_test_common.sh \
  methods/tls2trees/slurm/for_instance/prepare_published_default_test_manifest.sbatch \
  methods/tls2trees/slurm/for_instance/prepare_published_default_test_semantic.sbatch \
  methods/tls2trees/slurm/for_instance/evaluate_published_default_test_plot.sbatch \
  methods/tls2trees/slurm/for_instance/summarise_published_default_test.sbatch; do
  test -f "$path"
done
PUBLISHED_CONFIG_SHA256=$(sha256sum "$PUBLISHED_CONFIG" | awk '{print $1}')
test "$PUBLISHED_CONFIG_SHA256" = "$REVIEWED_CONFIG_SHA256"
WORKFLOW_CONFIG_SHA256=$(sha256sum "$WORKFLOW_CONFIG" | awk '{print $1}')
BENCHMARK_CONFIG_SHA256=$(sha256sum "$BENCHMARK_CONFIG" | awk '{print $1}')
METHOD_ENV_MARKER_SHA256=$(sha256sum "$METHOD_ENV_MARKER" | awk '{print $1}')
"$TREEBENCH_ENV/bin/python" - "$WORKFLOW_CONFIG" "$PUBLISHED_CONFIG" <<'PY'
import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd() / "methods/tls2trees/scripts/runtime"))
from published_default_test_common import validate_frozen_configuration

validate_frozen_configuration(sys.argv[1], sys.argv[2])
PY
"$METHOD_ENV/bin/python" "$ENV_VALIDATOR" \
  --tls2trees-repo "$UPSTREAM_REPO" \
  --setup-marker-json "$METHOD_ENV_MARKER" --skip-model-load >/dev/null

CACHE_AVAILABLE=0
CACHE_STATE_SHA256=none
CACHE_MANIFEST_JSON=none
CACHE_MANIFEST_CSV=none
CACHE_MANIFEST_SHA256=none
CACHE_OUTPUT_ROOT=none
CACHE_RUN_ID=none
CACHE_VARIANT=none
if [[ -n "$CACHE_STATE_FILE" ]]; then
  CACHE_STATE_FILE=$(realpath "$CACHE_STATE_FILE")
  test -f "$CACHE_STATE_FILE"
  CACHE_STATE_SHA256=$(sha256sum "$CACHE_STATE_FILE" | awk '{print $1}')
  # shellcheck disable=SC1090
  source "$CACHE_STATE_FILE"
  CACHE_MANIFEST_JSON="${TLS2TREES_TEST_MANIFEST_JSON:?Cache state has no manifest JSON}"
  CACHE_MANIFEST_CSV="${TLS2TREES_TEST_MANIFEST_CSV:?Cache state has no manifest CSV}"
  CACHE_OUTPUT_ROOT="${TLS2TREES_TEST_OUTPUT_ROOT:?Cache state has no output root}"
  CACHE_RUN_ID="${TLS2TREES_TEST_SEMANTIC_CACHE_RUN_ID:?Cache state has no semantic run ID}"
  CACHE_VARIANT=development_tuned
  test -f "$CACHE_MANIFEST_JSON"
  test -f "$CACHE_MANIFEST_CSV"
  CACHE_MANIFEST_SHA256=$(sha256sum "$CACHE_MANIFEST_JSON" | awk '{print $1}')
  "$TREEBENCH_ENV/bin/python" "$MANIFEST_CLI" validate \
    --manifest-json "$CACHE_MANIFEST_JSON" \
    --expected-split test --allow-held-out-test >/dev/null
  CACHE_AVAILABLE=1
else
  CACHE_STATE_FILE=none
fi

LATEST_POINTER="logs/tls2trees_for_instance/latest_published_default_test_state_file.txt"
if [[ -e "$LATEST_POINTER" ]]; then
  echo "Refusing: a published-default held-out state already exists at $LATEST_POINTER" >&2
  echo "This fixed test route is one-time; operational recovery must retain its state." >&2
  exit 2
fi
MIN_FREE_BYTES="${TLS2TREES_PUBLISHED_DEFAULT_TEST_MIN_FREE_BYTES:-85899345920}"
FREE_BYTES=$(df -PB1 "$PROJECT_ROOT" | awk 'NR == 2 {print $4}')
if [[ ! "$MIN_FREE_BYTES" =~ ^[1-9][0-9]*$ ]] || \
  [[ ! "$FREE_BYTES" =~ ^[0-9]+$ ]] || ((FREE_BYTES < MIN_FREE_BYTES)); then
  echo "Need at least $MIN_FREE_BYTES free bytes; found ${FREE_BYTES:-unknown}." >&2
  exit 2
fi

STAMP="${TLS2TREES_PUBLISHED_DEFAULT_TEST_STAMP:-$(date -u +%Y%m%d_%H%M%S)}"
[[ "$STAMP" =~ ^[0-9]{8}_[0-9]{6}$ ]]
RUN_ID="tls2trees_for-instance_published_default_held_out_test_$STAMP"
WORKFLOW_ROOT="$PROJECT_ROOT/results/metadata/tls2trees/for_instance/published_default/test/$RUN_ID"
TABLE_ROOT="$PROJECT_ROOT/results/tables/tls2trees/for_instance/published_default/test/$RUN_ID"
MANIFEST_JSON="$WORKFLOW_ROOT/test_manifest.json"
MANIFEST_CSV="$WORKFLOW_ROOT/test_manifest.csv"
MANIFEST_SHA256_FILE="$WORKFLOW_ROOT/test_manifest.sha256"
SUMMARY_JSON="$WORKFLOW_ROOT/published_default_test_summary.json"
RETENTION_JSON="$WORKFLOW_ROOT/prediction_retention_manifest.json"
PLOT_CSV="$TABLE_ROOT/plot_metrics.csv"
AGGREGATE_CSV="$TABLE_ROOT/target_summary.csv"
RUNTIME_ROOT="$OUTPUT_ROOT/tls2trees/for_instance/published_default/test/$RUN_ID"
STATE_DIR="$HOME/fastscratch/tls2trees_for_instance_published_default_test_states"
STATE_FILE="$STATE_DIR/$RUN_ID.env"
for path in "$WORKFLOW_ROOT" "$TABLE_ROOT" "$RUNTIME_ROOT" "$STATE_FILE"; do
  test ! -e "$path"
done
mkdir -p logs/tls2trees_for_instance "$STATE_DIR" "$WORKFLOW_ROOT"

MANIFEST_JOB=not_submitted
SEMANTIC_JOB=not_submitted
EVALUATE_JOB=not_submitted
SUMMARY_JOB=not_submitted
SUBMISSION_STATUS=preflight_completed
write_state() {
  {
    printf 'TLS2TREES_PD_TEST_RUN_ID=%q\n' "$RUN_ID"
    printf 'TLS2TREES_PD_TEST_SUBMISSION_STATUS=%q\n' "$SUBMISSION_STATUS"
    printf 'TLS2TREES_PD_TEST_MANIFEST_JOB=%q\n' "$MANIFEST_JOB"
    printf 'TLS2TREES_PD_TEST_SEMANTIC_JOB=%q\n' "$SEMANTIC_JOB"
    printf 'TLS2TREES_PD_TEST_EVALUATE_JOB=%q\n' "$EVALUATE_JOB"
    printf 'TLS2TREES_PD_TEST_SUMMARY_JOB=%q\n' "$SUMMARY_JOB"
    printf 'TLS2TREES_PD_TEST_BENCHMARK_COMMIT=%q\n' "$BENCHMARK_COMMIT"
    printf 'TLS2TREES_PD_TEST_UPSTREAM_COMMIT=%q\n' "$UPSTREAM_COMMIT"
    printf 'TLS2TREES_PD_TEST_MODEL_SHA256=%q\n' "$MODEL_SHA256"
    printf 'TLS2TREES_PD_TEST_METHOD_ENV_MARKER_SHA256=%q\n' "$METHOD_ENV_MARKER_SHA256"
    printf 'TLS2TREES_PD_TEST_WORKFLOW_CONFIG=%q\n' "$PROJECT_ROOT/$WORKFLOW_CONFIG"
    printf 'TLS2TREES_PD_TEST_WORKFLOW_CONFIG_SHA256=%q\n' "$WORKFLOW_CONFIG_SHA256"
    printf 'TLS2TREES_PD_TEST_PUBLISHED_CONFIG=%q\n' "$PROJECT_ROOT/$PUBLISHED_CONFIG"
    printf 'TLS2TREES_PD_TEST_PUBLISHED_CONFIG_SHA256=%q\n' "$PUBLISHED_CONFIG_SHA256"
    printf 'TLS2TREES_PD_TEST_BENCHMARK_CONFIG=%q\n' "$PROJECT_ROOT/$BENCHMARK_CONFIG"
    printf 'TLS2TREES_PD_TEST_BENCHMARK_CONFIG_SHA256=%q\n' "$BENCHMARK_CONFIG_SHA256"
    printf 'TLS2TREES_PD_TEST_MANIFEST_JSON=%q\n' "$MANIFEST_JSON"
    printf 'TLS2TREES_PD_TEST_MANIFEST_CSV=%q\n' "$MANIFEST_CSV"
    printf 'TLS2TREES_PD_TEST_MANIFEST_SHA256_FILE=%q\n' "$MANIFEST_SHA256_FILE"
    printf 'TLS2TREES_PD_TEST_OUTPUT_ROOT=%q\n' "$OUTPUT_ROOT"
    printf 'TLS2TREES_PD_TEST_SUMMARY_JSON=%q\n' "$SUMMARY_JSON"
    printf 'TLS2TREES_PD_TEST_RETENTION_JSON=%q\n' "$RETENTION_JSON"
    printf 'TLS2TREES_PD_TEST_PLOT_CSV=%q\n' "$PLOT_CSV"
    printf 'TLS2TREES_PD_TEST_AGGREGATE_CSV=%q\n' "$AGGREGATE_CSV"
    printf 'TLS2TREES_PD_TEST_TREEBENCH_ENV=%q\n' "$TREEBENCH_ENV"
    printf 'TLS2TREES_PD_TEST_CACHE_AVAILABLE=%q\n' "$CACHE_AVAILABLE"
    printf 'TLS2TREES_PD_TEST_CACHE_STATE_FILE=%q\n' "$CACHE_STATE_FILE"
    printf 'TLS2TREES_PD_TEST_CACHE_STATE_SHA256=%q\n' "$CACHE_STATE_SHA256"
    printf 'TLS2TREES_PD_TEST_CACHE_MANIFEST_SHA256=%q\n' "$CACHE_MANIFEST_SHA256"
    printf 'TLS2TREES_PD_TEST_CONFIGURATION_CHANGED_AFTER_TEST=false\n'
  } > "$STATE_FILE"
}
write_state
printf '%s\n' "$STATE_FILE" > "$LATEST_POINTER"

SUBMITTED_JOBS=()
cancel_partial_submission() {
  local status=$?
  if ((${#SUBMITTED_JOBS[@]})); then
    scancel "${SUBMITTED_JOBS[@]}" 2>/dev/null || true
  fi
  SUBMISSION_STATUS=published_default_submission_failed_jobs_cancelled
  write_state
  echo "Submission failed; newly submitted jobs were cancelled." >&2
  echo "state_file=$STATE_FILE" >&2
  exit "$status"
}
trap cancel_partial_submission ERR

COMMON_EXPORTS="ALL,TLS2TREES_PUBLISHED_DEFAULT_TEST_CONFIRMED=1,TLS2TREES_REQUESTED_VARIANT=published_default,TLS2TREES_REQUESTED_SPLIT=test,TLS2TREES_PD_TEST_PROJECT_ROOT=$PROJECT_ROOT,TLS2TREES_PD_TEST_TREEBENCH_ENV=$TREEBENCH_ENV,TLS2TREES_PD_TEST_METHOD_ENV=$METHOD_ENV,TLS2TREES_PD_TEST_METHOD_ENV_MARKER=$METHOD_ENV_MARKER,TLS2TREES_PD_TEST_METHOD_ENV_MARKER_SHA256=$METHOD_ENV_MARKER_SHA256,TLS2TREES_PD_TEST_UPSTREAM_REPO=$UPSTREAM_REPO,TLS2TREES_PD_TEST_DATASET_ROOT=$DATASET_ROOT,TLS2TREES_PD_TEST_SPLIT_METADATA_CSV=$SPLIT_METADATA_CSV,TLS2TREES_PD_TEST_BENCHMARK_COMMIT=$BENCHMARK_COMMIT,TLS2TREES_PD_TEST_UPSTREAM_COMMIT=$UPSTREAM_COMMIT,TLS2TREES_PD_TEST_MODEL_SHA256=$MODEL_SHA256,TLS2TREES_PD_TEST_RUN_ID=$RUN_ID,TLS2TREES_PD_TEST_OUTPUT_ROOT=$OUTPUT_ROOT,TLS2TREES_PD_TEST_MANIFEST_JSON=$MANIFEST_JSON,TLS2TREES_PD_TEST_MANIFEST_CSV=$MANIFEST_CSV,TLS2TREES_PD_TEST_MANIFEST_SHA256_FILE=$MANIFEST_SHA256_FILE,TLS2TREES_PD_TEST_WORKFLOW_CONFIG=$PROJECT_ROOT/$WORKFLOW_CONFIG,TLS2TREES_PD_TEST_WORKFLOW_CONFIG_SHA256=$WORKFLOW_CONFIG_SHA256,TLS2TREES_PD_TEST_PUBLISHED_CONFIG=$PROJECT_ROOT/$PUBLISHED_CONFIG,TLS2TREES_PD_TEST_PUBLISHED_CONFIG_SHA256=$PUBLISHED_CONFIG_SHA256,TLS2TREES_PD_TEST_BENCHMARK_CONFIG=$PROJECT_ROOT/$BENCHMARK_CONFIG,TLS2TREES_PD_TEST_BENCHMARK_CONFIG_SHA256=$BENCHMARK_CONFIG_SHA256,TLS2TREES_PD_TEST_MANIFEST_CLI=$MANIFEST_CLI,TLS2TREES_PD_TEST_CONVERT_CLI=$CONVERT_CLI,TLS2TREES_PD_TEST_SEMANTIC_CLI=$SEMANTIC_CLI,TLS2TREES_PD_TEST_INSTANCE_CLI=$INSTANCE_CLI,TLS2TREES_PD_TEST_ADAPTER_CLI=$ADAPTER_CLI,TLS2TREES_PD_TEST_EVALUATE_CLI=$EVALUATE_CLI,TLS2TREES_PD_TEST_ENV_VALIDATOR=$ENV_VALIDATOR,TLS2TREES_PD_TEST_CACHE_HELPER=$CACHE_HELPER,TLS2TREES_PD_TEST_CACHE_AVAILABLE=$CACHE_AVAILABLE,TLS2TREES_PD_TEST_CACHE_STATE_FILE=$CACHE_STATE_FILE,TLS2TREES_PD_TEST_CACHE_STATE_SHA256=$CACHE_STATE_SHA256,TLS2TREES_PD_TEST_CACHE_MANIFEST_JSON=$CACHE_MANIFEST_JSON,TLS2TREES_PD_TEST_CACHE_MANIFEST_CSV=$CACHE_MANIFEST_CSV,TLS2TREES_PD_TEST_CACHE_MANIFEST_SHA256=$CACHE_MANIFEST_SHA256,TLS2TREES_PD_TEST_CACHE_OUTPUT_ROOT=$CACHE_OUTPUT_ROOT,TLS2TREES_PD_TEST_CACHE_RUN_ID=$CACHE_RUN_ID,TLS2TREES_PD_TEST_CACHE_VARIANT=$CACHE_VARIANT,TLS2TREES_PD_TEST_SUMMARY_JSON=$SUMMARY_JSON,TLS2TREES_PD_TEST_RETENTION_JSON=$RETENTION_JSON,TLS2TREES_PD_TEST_PLOT_CSV=$PLOT_CSV,TLS2TREES_PD_TEST_AGGREGATE_CSV=$AGGREGATE_CSV"

MANIFEST_JOB=$(sbatch --parsable --export="$COMMON_EXPORTS" \
  methods/tls2trees/slurm/for_instance/prepare_published_default_test_manifest.sbatch)
SUBMITTED_JOBS+=("$MANIFEST_JOB")
SUBMISSION_STATUS=manifest_submitted; write_state
SEMANTIC_JOB=$(sbatch --parsable --array="0-10%2" \
  --dependency="afterok:$MANIFEST_JOB" --kill-on-invalid-dep=yes \
  --export="$COMMON_EXPORTS" \
  methods/tls2trees/slurm/for_instance/prepare_published_default_test_semantic.sbatch)
SUBMITTED_JOBS+=("$SEMANTIC_JOB")
SUBMISSION_STATUS=semantic_submitted; write_state
EVALUATE_JOB=$(sbatch --parsable --array="0-10%4" \
  --dependency="afterok:$SEMANTIC_JOB" --kill-on-invalid-dep=yes \
  --export="$COMMON_EXPORTS" \
  methods/tls2trees/slurm/for_instance/evaluate_published_default_test_plot.sbatch)
SUBMITTED_JOBS+=("$EVALUATE_JOB")
SUBMISSION_STATUS=evaluation_submitted; write_state
SUMMARY_JOB=$(sbatch --parsable --dependency="afterok:$EVALUATE_JOB" \
  --kill-on-invalid-dep=yes --export="$COMMON_EXPORTS" \
  methods/tls2trees/slurm/for_instance/summarise_published_default_test.sbatch)
SUBMITTED_JOBS+=("$SUMMARY_JOB")
SUBMISSION_STATUS=published_default_test_chain_submitted; write_state
trap - ERR

echo "run_id=$RUN_ID"
echo "manifest_job=$MANIFEST_JOB expected_plots=11"
echo "semantic_job=$SEMANTIC_JOB expected_plots=11 concurrency=2"
echo "evaluate_job=$EVALUATE_JOB expected_plots=11 expected_metrics=22 concurrency=4"
echo "summary_job=$SUMMARY_JOB"
echo "semantic_cache_available=$CACHE_AVAILABLE"
echo "state_file=$STATE_FILE"
echo "published_config_sha256=$PUBLISHED_CONFIG_SHA256"
echo "configuration_changed_after_test=false"
