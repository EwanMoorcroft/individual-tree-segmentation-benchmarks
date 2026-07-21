#!/usr/bin/env bash

set -euo pipefail

if [[ "${TLS2TREES_HELD_OUT_TEST_CONFIRMED:-0}" != "1" ]]; then
  echo "Refusing one-time TLS2trees held-out test submission." >&2
  echo "Set TLS2TREES_HELD_OUT_TEST_CONFIRMED=1 only after reviewing the frozen configuration." >&2
  exit 2
fi
REVIEWED_SHA256="${TLS2TREES_REVIEWED_FINAL_SELECTION_SHA256:?Set the reviewed final-selection SHA-256}"
FINAL_SELECTION="${1:?Usage: submit_held_out_test.sh <reviewed-final-selection.json>}"
test -f "$FINAL_SELECTION"

PROJECT_ROOT="${TLS2TREES_PROJECT_ROOT:-$HOME/scratch/tree-seg-benchmark}"
TREEBENCH_ENV="${TLS2TREES_TREEBENCH_ENV:-$HOME/fastscratch/venvs/treebench}"
METHOD_ENV="${TLS2TREES_METHOD_ENV:-$HOME/fastscratch/venvs/tls2trees}"
METHOD_ENV_MARKER="$METHOD_ENV/.tls2trees_setup_complete.json"
UPSTREAM_REPO="${TLS2TREES_UPSTREAM_REPO:-$PROJECT_ROOT/external/TLS2trees}"
DATASET_ROOT="${TLS2TREES_FOR_INSTANCE_ROOT:-$HOME/data/datasets/for_instance/FORinstance_dataset}"
SPLIT_METADATA_CSV="$DATASET_ROOT/data_split_metadata.csv"
STAGE2_STATE=$(<"$PROJECT_ROOT/logs/tls2trees_for_instance/latest_stage2_state_file.txt")
test -f "$STAGE2_STATE"
# shellcheck disable=SC1090
source "$STAGE2_STATE"

OUTPUT_ROOT="${TLS2TREES_STAGE2_OUTPUT_ROOT:?Stage 2 state has no output root}"
STAGE1_CONFIG="${TLS2TREES_STAGE2_STAGE1_CONFIG:?Stage 2 state has no Stage 1 config}"
PROBE_SUMMARY_JSON="${TLS2TREES_STAGE2_PROBE_SUMMARY_JSON:?Stage 2 state has no probe summary}"
PROBE_SUMMARY_SHA256="${TLS2TREES_STAGE2_PROBE_SUMMARY_SHA256:?Stage 2 state has no probe hash}"
MANIFEST_CLI="methods/tls2trees/scripts/data/prepare_for_instance_manifest.py"
CONVERT_CLI="methods/tls2trees/scripts/data/prepare_for_instance_tls2trees_input.py"
SEMANTIC_CLI="methods/tls2trees/scripts/runtime/run_for_instance_tls2trees_semantic.py"
CANDIDATE_CLI="methods/tls2trees/scripts/runtime/run_for_instance_tls2trees_development_candidate.py"
ADAPTER_CLI="methods/tls2trees/scripts/evaluation/adapt_for_instance_tls2trees_predictions.py"
EVALUATE_CLI="methods/tls2trees/scripts/evaluation/evaluate_for_instance_tls2trees_plot.py"
ENV_VALIDATOR="methods/tls2trees/scripts/runtime/validate_tls2trees_environment.py"

cd "$PROJECT_ROOT"
test -d .git
test -z "$(git status --porcelain)"
BENCHMARK_COMMIT=$(git rev-parse HEAD)
test -x "$TREEBENCH_ENV/bin/python"
test -x "$METHOD_ENV/bin/python"
test -f "$METHOD_ENV_MARKER"
test -f "$SPLIT_METADATA_CSV"
test "$(sha256sum "$FINAL_SELECTION" | awk '{print $1}')" = "$REVIEWED_SHA256"
FINAL_SELECTION=$(realpath "$FINAL_SELECTION")
test "$(git -C "$UPSTREAM_REPO" rev-parse HEAD)" = "ca12cb73b2c736d80b020e8025f8d975d42e6f01"
test -z "$(git -C "$UPSTREAM_REPO" status --porcelain)"
for path in "$STAGE1_CONFIG" "$PROBE_SUMMARY_JSON" "$MANIFEST_CLI" \
  "$CONVERT_CLI" "$SEMANTIC_CLI" "$CANDIDATE_CLI" "$ADAPTER_CLI" \
  "$EVALUATE_CLI" "$ENV_VALIDATOR" \
  methods/tls2trees/scripts/evaluation/summarise_tls2trees_held_out_test.py \
  methods/tls2trees/slurm/for_instance/prepare_held_out_test_manifest.sbatch \
  methods/tls2trees/slurm/for_instance/prepare_semantic_held_out_test.sbatch \
  methods/tls2trees/slurm/for_instance/evaluate_held_out_test_candidate.sbatch \
  methods/tls2trees/slurm/for_instance/summarise_held_out_test.sbatch; do
  test -f "$path"
done
STAGE1_CONFIG_SHA256=$(sha256sum "$STAGE1_CONFIG" | awk '{print $1}')
METHOD_ENV_MARKER_SHA256=$(sha256sum "$METHOD_ENV_MARKER" | awk '{print $1}')
"$TREEBENCH_ENV/bin/python" -c '
import json,sys
p=json.load(open(sys.argv[1]))
assert p["status"] == "development_tuned_configuration_frozen"
assert p["variant"] == "development_tuned"
assert p["selection_split"] == "development"
assert p["held_out_test_accessed"] is False
assert p["held_out_test_runnable"] is False
assert p["final_configuration_selected"] is True
assert p["review_required_before_held_out_test"] is True
assert p["selected_by_target"]["leaf_off"]["candidate_id"] == "p04_min_points_50_lower_band"
assert p["selected_by_target"]["leaf_on"]["candidate_id"] == "p02_min_points_50"
assert p["source_stage1_config_sha256"] == sys.argv[2]
assert p["source_stage2_run_id"] == sys.argv[3]
' "$FINAL_SELECTION" "$STAGE1_CONFIG_SHA256" "$TLS2TREES_STAGE2_RUN_ID"
test "$(sha256sum "$PROBE_SUMMARY_JSON" | awk '{print $1}')" = "$PROBE_SUMMARY_SHA256"

LATEST_POINTER="logs/tls2trees_for_instance/latest_held_out_test_state_file.txt"
if [[ -e "$LATEST_POINTER" ]]; then
  echo "Refusing: a held-out test state already exists at $LATEST_POINTER" >&2
  echo "This workflow is intentionally one-time and cannot be resubmitted for tuning." >&2
  exit 2
fi
MIN_FREE_BYTES="${TLS2TREES_TEST_MIN_FREE_BYTES:-161061273600}"
FREE_BYTES=$(df -PB1 "$PROJECT_ROOT" | awk 'NR == 2 {print $4}')
if [[ ! "$FREE_BYTES" =~ ^[0-9]+$ ]] || ((FREE_BYTES < MIN_FREE_BYTES)); then
  echo "Need at least $MIN_FREE_BYTES free bytes; found ${FREE_BYTES:-unknown}." >&2
  exit 2
fi

STAMP="${TLS2TREES_TEST_STAMP:-$(date -u +%Y%m%d_%H%M%S)}"
[[ "$STAMP" =~ ^[0-9]{8}_[0-9]{6}$ ]]
RUN_ID="tls2trees_for-instance_development_tuned_held_out_test_$STAMP"
SEMANTIC_CACHE_RUN_ID="${RUN_ID}__semantic_cache"
WORKFLOW_ROOT="$PROJECT_ROOT/results/metadata/tls2trees/for_instance/development_tuned/test/$RUN_ID"
TABLE_ROOT="$PROJECT_ROOT/results/tables/tls2trees/for_instance/development_tuned/test/$RUN_ID"
MANIFEST_JSON="$WORKFLOW_ROOT/test_manifest.json"
MANIFEST_CSV="$WORKFLOW_ROOT/test_manifest.csv"
MANIFEST_SHA256_FILE="$WORKFLOW_ROOT/test_manifest.sha256"
SUMMARY_JSON="$WORKFLOW_ROOT/held_out_test_summary.json"
PLOT_CSV="$TABLE_ROOT/plot_metrics.csv"
AGGREGATE_CSV="$TABLE_ROOT/target_summary.csv"
STATE_DIR="$HOME/fastscratch/tls2trees_for_instance_test_states"
STATE_FILE="$STATE_DIR/$RUN_ID.env"
SEMANTIC_CACHE_ROOT="$OUTPUT_ROOT/tls2trees/for_instance/development_tuned/test/$SEMANTIC_CACHE_RUN_ID"
for path in "$WORKFLOW_ROOT" "$TABLE_ROOT" "$STATE_FILE" "$SEMANTIC_CACHE_ROOT"; do
  test ! -e "$path"
done
mkdir -p logs/tls2trees_for_instance "$STATE_DIR" "$WORKFLOW_ROOT"

INVENTORY_JOB=not_submitted
SEMANTIC_JOB=not_submitted
CANDIDATE_JOB=not_submitted
SUMMARY_JOB=not_submitted
SUBMISSION_STATUS=preflight_completed
write_state() {
  {
    printf 'TLS2TREES_TEST_RUN_ID=%q\n' "$RUN_ID"
    printf 'TLS2TREES_TEST_SUBMISSION_STATUS=%q\n' "$SUBMISSION_STATUS"
    printf 'TLS2TREES_TEST_INVENTORY_JOB=%q\n' "$INVENTORY_JOB"
    printf 'TLS2TREES_TEST_SEMANTIC_JOB=%q\n' "$SEMANTIC_JOB"
    printf 'TLS2TREES_TEST_CANDIDATE_JOB=%q\n' "$CANDIDATE_JOB"
    printf 'TLS2TREES_TEST_SUMMARY_JOB=%q\n' "$SUMMARY_JOB"
    printf 'TLS2TREES_TEST_BENCHMARK_COMMIT=%q\n' "$BENCHMARK_COMMIT"
    printf 'TLS2TREES_TEST_FINAL_SELECTION_JSON=%q\n' "$FINAL_SELECTION"
    printf 'TLS2TREES_TEST_FINAL_SELECTION_SHA256=%q\n' "$REVIEWED_SHA256"
    printf 'TLS2TREES_TEST_MANIFEST_JSON=%q\n' "$MANIFEST_JSON"
    printf 'TLS2TREES_TEST_MANIFEST_CSV=%q\n' "$MANIFEST_CSV"
    printf 'TLS2TREES_TEST_MANIFEST_SHA256_FILE=%q\n' "$MANIFEST_SHA256_FILE"
    printf 'TLS2TREES_TEST_SEMANTIC_CACHE_RUN_ID=%q\n' "$SEMANTIC_CACHE_RUN_ID"
    printf 'TLS2TREES_TEST_OUTPUT_ROOT=%q\n' "$OUTPUT_ROOT"
    printf 'TLS2TREES_TEST_SUMMARY_JSON=%q\n' "$SUMMARY_JSON"
    printf 'TLS2TREES_TEST_PLOT_CSV=%q\n' "$PLOT_CSV"
    printf 'TLS2TREES_TEST_AGGREGATE_CSV=%q\n' "$AGGREGATE_CSV"
    printf 'TLS2TREES_TEST_TREEBENCH_ENV=%q\n' "$TREEBENCH_ENV"
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
  SUBMISSION_STATUS=held_out_test_submission_failed_jobs_cancelled
  write_state
  echo "Held-out submission failed; newly submitted jobs were cancelled." >&2
  echo "The state pointer is retained to prevent an accidental second test run." >&2
  echo "state_file=$STATE_FILE" >&2
  exit "$status"
}
trap cancel_partial_submission ERR

COMMON_EXPORTS="ALL,TLS2TREES_HELD_OUT_TEST_CONFIRMED=1,TLS2TREES_REQUESTED_VARIANT=development_tuned,TLS2TREES_REQUESTED_SPLIT=test,TLS2TREES_PROJECT_ROOT=$PROJECT_ROOT,TLS2TREES_TREEBENCH_ENV=$TREEBENCH_ENV,TLS2TREES_METHOD_ENV=$METHOD_ENV,TLS2TREES_METHOD_ENV_MARKER=$METHOD_ENV_MARKER,TLS2TREES_METHOD_ENV_MARKER_SHA256=$METHOD_ENV_MARKER_SHA256,TLS2TREES_UPSTREAM_REPO=$UPSTREAM_REPO,TLS2TREES_EXPECTED_BENCHMARK_COMMIT=$BENCHMARK_COMMIT,TLS2TREES_DATASET_ROOT=$DATASET_ROOT,TLS2TREES_SPLIT_METADATA_CSV=$SPLIT_METADATA_CSV,TLS2TREES_MANIFEST_JSON=$MANIFEST_JSON,TLS2TREES_MANIFEST_CSV=$MANIFEST_CSV,TLS2TREES_MANIFEST_SHA256_FILE=$MANIFEST_SHA256_FILE,TLS2TREES_MANIFEST_CLI=$MANIFEST_CLI,TLS2TREES_CONVERT_CLI=$CONVERT_CLI,TLS2TREES_SEMANTIC_CLI=$SEMANTIC_CLI,TLS2TREES_CANDIDATE_CLI=$CANDIDATE_CLI,TLS2TREES_ADAPTER_CLI=$ADAPTER_CLI,TLS2TREES_EVALUATE_CLI=$EVALUATE_CLI,TLS2TREES_ENV_VALIDATOR=$ENV_VALIDATOR,TLS2TREES_STAGE1_CONFIG=$STAGE1_CONFIG,TLS2TREES_STAGE1_CONFIG_SHA256=$STAGE1_CONFIG_SHA256,TLS2TREES_PROBE_SUMMARY_JSON=$PROBE_SUMMARY_JSON,TLS2TREES_PROBE_SUMMARY_SHA256=$PROBE_SUMMARY_SHA256,TLS2TREES_FINAL_SELECTION_JSON=$FINAL_SELECTION,TLS2TREES_FINAL_SELECTION_SHA256=$REVIEWED_SHA256,TLS2TREES_OUTPUT_ROOT=$OUTPUT_ROOT,TLS2TREES_TEST_RUN_ID=$RUN_ID,TLS2TREES_TEST_SEMANTIC_CACHE_RUN_ID=$SEMANTIC_CACHE_RUN_ID,TLS2TREES_TEST_SUMMARY_JSON=$SUMMARY_JSON,TLS2TREES_TEST_PLOT_CSV=$PLOT_CSV,TLS2TREES_TEST_AGGREGATE_CSV=$AGGREGATE_CSV"

INVENTORY_JOB=$(sbatch --parsable --export="$COMMON_EXPORTS" \
  methods/tls2trees/slurm/for_instance/prepare_held_out_test_manifest.sbatch)
SUBMITTED_JOBS+=("$INVENTORY_JOB")
SUBMISSION_STATUS=inventory_submitted; write_state
SEMANTIC_JOB=$(sbatch --parsable --array="0-10%2" --dependency="afterok:$INVENTORY_JOB" \
  --kill-on-invalid-dep=yes --export="$COMMON_EXPORTS" \
  methods/tls2trees/slurm/for_instance/prepare_semantic_held_out_test.sbatch)
SUBMITTED_JOBS+=("$SEMANTIC_JOB")
SUBMISSION_STATUS=semantic_submitted; write_state
CANDIDATE_JOB=$(sbatch --parsable --array="0-21%4" --dependency="afterok:$SEMANTIC_JOB" \
  --kill-on-invalid-dep=yes --export="$COMMON_EXPORTS" \
  methods/tls2trees/slurm/for_instance/evaluate_held_out_test_candidate.sbatch)
SUBMITTED_JOBS+=("$CANDIDATE_JOB")
SUBMISSION_STATUS=candidates_submitted; write_state
SUMMARY_JOB=$(sbatch --parsable --dependency="afterany:$CANDIDATE_JOB" \
  --kill-on-invalid-dep=yes --export="$COMMON_EXPORTS" \
  methods/tls2trees/slurm/for_instance/summarise_held_out_test.sbatch)
SUBMITTED_JOBS+=("$SUMMARY_JOB")
SUBMISSION_STATUS=held_out_test_chain_submitted; write_state
trap - ERR

echo "run_id=$RUN_ID"
echo "inventory_job=$INVENTORY_JOB expected_plots=11"
echo "semantic_job=$SEMANTIC_JOB gpu_tasks=11 concurrency=2"
echo "candidate_job=$CANDIDATE_JOB target_plot_tasks=22 concurrency=4"
echo "summary_job=$SUMMARY_JOB expected_metrics=22"
echo "state_file=$STATE_FILE"
echo "final_selection_sha256=$REVIEWED_SHA256"
echo "held_out_test_accessed=on_inventory_start"
echo "configuration_changes_after_submission=forbidden"
