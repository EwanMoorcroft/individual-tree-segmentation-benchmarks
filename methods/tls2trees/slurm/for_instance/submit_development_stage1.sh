#!/usr/bin/env bash

set -euo pipefail

if [[ "${TLS2TREES_STAGE1_CONFIRMED:-0}" != "1" ]]; then
  echo "Refusing TLS2trees development Stage 1 submission." >&2
  echo "Set TLS2TREES_STAGE1_CONFIRMED=1 after reviewing the completed probe summary." >&2
  exit 2
fi

PROBE_STATE_FILE="${1:?Usage: submit_development_stage1.sh <completed-probe-state-file>}"
test -f "$PROBE_STATE_FILE"
# shellcheck disable=SC1090
source "$PROBE_STATE_FILE"
PROBE_SUMMARY_JSON="${TLS2TREES_PROBE_SUMMARY_JSON:?Probe state has no summary}"
PROBE_RUN_ID="${TLS2TREES_PROBE_RUN_ID:?Probe state has no run ID}"
PROBE_SOURCE_STATE="${TLS2TREES_PROBE_SOURCE_STATE_FILE:?Probe state has no source state}"
METHOD_ENV="${TLS2TREES_PROBE_METHOD_ENV:?Probe state has no method environment}"
EXPECTED_MARKER_SHA256="${TLS2TREES_PROBE_METHOD_ENV_MARKER_SHA256:?Probe state has no marker hash}"
OUTPUT_ROOT="${TLS2TREES_PROBE_OUTPUT_ROOT:?Probe state has no output root}"

test -f "$PROBE_SOURCE_STATE"
# shellcheck disable=SC1090
source "$PROBE_SOURCE_STATE"
MANIFEST_JSON="${TLS2TREES_SMOKE_MANIFEST_JSON:?Source state has no development manifest}"

PROJECT_ROOT="${TLS2TREES_PROJECT_ROOT:-$HOME/scratch/tree-seg-benchmark}"
TREEBENCH_ENV="${TLS2TREES_TREEBENCH_ENV:-$HOME/fastscratch/venvs/treebench}"
UPSTREAM_REPO="${TLS2TREES_UPSTREAM_REPO:-$PROJECT_ROOT/external/TLS2trees}"
METHOD_ENV_MARKER="$METHOD_ENV/.tls2trees_setup_complete.json"
STAGE1_CONFIG="methods/tls2trees/configs/for_instance_development_tuned_stage1.yml"
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
test "$(sha256sum "$METHOD_ENV_MARKER" | awk '{print $1}')" = "$EXPECTED_MARKER_SHA256"
test "$(git -C "$UPSTREAM_REPO" rev-parse HEAD)" = \
  "ca12cb73b2c736d80b020e8025f8d975d42e6f01"
test -z "$(git -C "$UPSTREAM_REPO" status --porcelain)"
for path in "$MANIFEST_JSON" "$PROBE_SUMMARY_JSON" "$STAGE1_CONFIG" "$MANIFEST_CLI" \
  "$CONVERT_CLI" "$SEMANTIC_CLI" "$CANDIDATE_CLI" "$ADAPTER_CLI" \
  "$EVALUATE_CLI" "$ENV_VALIDATOR"; do
  test -f "$path"
done

PROBE_SUMMARY_SHA256=$(sha256sum "$PROBE_SUMMARY_JSON" | awk '{print $1}')
STAGE1_CONFIG_SHA256=$(sha256sum "$STAGE1_CONFIG" | awk '{print $1}')
"$TREEBENCH_ENV/bin/python" -c '
import json,sys,yaml
probe=json.load(open(sys.argv[1]))
config=yaml.safe_load(open(sys.argv[2]))
required=config["probe_promotion"]["required_viable_candidate_ids"]
assert probe["status"] == "viable_candidates_found"
assert probe["viable_candidate_ids"] == required
assert probe["incomplete_candidate_ids"] == []
assert probe["held_out_test_accessed"] is False
assert [c["candidate_id"] for c in config["candidates"]] == required
assert all(c["parameters"]["add_leaves"] is True for c in config["candidates"])
assert config["scope"]["held_out_test_accessed"] is False
assert config["scope"]["final_configuration_selected"] is False
' "$PROBE_SUMMARY_JSON" "$STAGE1_CONFIG"
"$TREEBENCH_ENV/bin/python" "$MANIFEST_CLI" validate \
  --manifest-json "$MANIFEST_JSON" --expected-split development >/dev/null

MIN_FREE_BYTES="${TLS2TREES_STAGE1_MIN_FREE_BYTES:-107374182400}"
FREE_BYTES=$(df -PB1 "$PROJECT_ROOT" | awk 'NR == 2 {print $4}')
if [[ ! "$FREE_BYTES" =~ ^[0-9]+$ ]] || ((FREE_BYTES < MIN_FREE_BYTES)); then
  echo "Need at least $MIN_FREE_BYTES free bytes; found ${FREE_BYTES:-unknown}." >&2
  exit 2
fi

STAMP="${TLS2TREES_STAGE1_STAMP:-$(date -u +%Y%m%d_%H%M%S)}"
if [[ ! "$STAMP" =~ ^[0-9]{8}_[0-9]{6}$ ]]; then
  echo "TLS2TREES_STAGE1_STAMP must use YYYYMMDD_HHMMSS." >&2
  exit 2
fi
RUN_ID="tls2trees_for-instance_development_tuned_stage1_$STAMP"
SEMANTIC_CACHE_RUN_ID="${RUN_ID}__semantic_cache"
WORKFLOW_ROOT="$PROJECT_ROOT/results/metadata/tls2trees/for_instance/development_tuned/stage1/$RUN_ID"
TABLE_ROOT="$PROJECT_ROOT/results/tables/tls2trees/for_instance/development_tuned/stage1/$RUN_ID"
SUMMARY_JSON="$WORKFLOW_ROOT/stage1_summary.json"
PLOT_CSV="$TABLE_ROOT/plot_metrics.csv"
AGGREGATE_CSV="$TABLE_ROOT/candidate_target_summary.csv"
STATE_DIR="$HOME/fastscratch/tls2trees_for_instance_stage1_states"
STATE_FILE="$STATE_DIR/$RUN_ID.env"
SEMANTIC_CACHE_ROOT="$OUTPUT_ROOT/tls2trees/for_instance/development_tuned/development/$SEMANTIC_CACHE_RUN_ID"
for path in "$SEMANTIC_CACHE_ROOT" "$WORKFLOW_ROOT" "$TABLE_ROOT" "$STATE_FILE"; do
  test ! -e "$path"
done
mkdir -p logs/tls2trees_for_instance "$STATE_DIR"

SEMANTIC_JOB="not_submitted"
CANDIDATE_JOB="not_submitted"
SUMMARY_JOB="not_submitted"
SUBMISSION_STATUS="preflight_completed"
SUBMITTED_JOBS=()
write_state() {
  {
    printf 'TLS2TREES_STAGE1_RUN_ID=%q\n' "$RUN_ID"
    printf 'TLS2TREES_STAGE1_SUBMISSION_STATUS=%q\n' "$SUBMISSION_STATUS"
    printf 'TLS2TREES_STAGE1_SEMANTIC_JOB=%q\n' "$SEMANTIC_JOB"
    printf 'TLS2TREES_STAGE1_CANDIDATE_JOB=%q\n' "$CANDIDATE_JOB"
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
    printf 'TLS2TREES_STAGE1_SUMMARY_JSON=%q\n' "$SUMMARY_JSON"
    printf 'TLS2TREES_STAGE1_PLOT_CSV=%q\n' "$PLOT_CSV"
    printf 'TLS2TREES_STAGE1_AGGREGATE_CSV=%q\n' "$AGGREGATE_CSV"
    printf 'TLS2TREES_STAGE1_TREEBENCH_ENV=%q\n' "$TREEBENCH_ENV"
  } > "$STATE_FILE"
}
cancel_partial_submission() {
  local status=$?
  if ((${#SUBMITTED_JOBS[@]})); then scancel "${SUBMITTED_JOBS[@]}" 2>/dev/null || true; fi
  SUBMISSION_STATUS="submission_failed_jobs_cancelled"
  write_state
  echo "Stage 1 submission failed; new jobs were cancelled." >&2
  echo "state_file=$STATE_FILE" >&2
  exit "$status"
}
trap cancel_partial_submission ERR

COMMON_EXPORTS="ALL,TLS2TREES_STAGE1_CONFIRMED=1,TLS2TREES_REQUESTED_VARIANT=development_tuned,TLS2TREES_REQUESTED_SPLIT=development,TLS2TREES_PROJECT_ROOT=$PROJECT_ROOT,TLS2TREES_TREEBENCH_ENV=$TREEBENCH_ENV,TLS2TREES_METHOD_ENV=$METHOD_ENV,TLS2TREES_METHOD_ENV_MARKER=$METHOD_ENV_MARKER,TLS2TREES_METHOD_ENV_MARKER_SHA256=$EXPECTED_MARKER_SHA256,TLS2TREES_UPSTREAM_REPO=$UPSTREAM_REPO,TLS2TREES_EXPECTED_BENCHMARK_COMMIT=$BENCHMARK_COMMIT,TLS2TREES_MANIFEST_JSON=$MANIFEST_JSON,TLS2TREES_MANIFEST_CLI=$MANIFEST_CLI,TLS2TREES_CONVERT_CLI=$CONVERT_CLI,TLS2TREES_SEMANTIC_CLI=$SEMANTIC_CLI,TLS2TREES_CANDIDATE_CLI=$CANDIDATE_CLI,TLS2TREES_ADAPTER_CLI=$ADAPTER_CLI,TLS2TREES_EVALUATE_CLI=$EVALUATE_CLI,TLS2TREES_ENV_VALIDATOR=$ENV_VALIDATOR,TLS2TREES_STAGE1_CONFIG=$STAGE1_CONFIG,TLS2TREES_STAGE1_CONFIG_SHA256=$STAGE1_CONFIG_SHA256,TLS2TREES_PROBE_SUMMARY_JSON=$PROBE_SUMMARY_JSON,TLS2TREES_PROBE_SUMMARY_SHA256=$PROBE_SUMMARY_SHA256,TLS2TREES_OUTPUT_ROOT=$OUTPUT_ROOT,TLS2TREES_STAGE1_RUN_ID=$RUN_ID,TLS2TREES_SEMANTIC_CACHE_RUN_ID=$SEMANTIC_CACHE_RUN_ID,TLS2TREES_STAGE1_SUMMARY_JSON=$SUMMARY_JSON,TLS2TREES_STAGE1_PLOT_CSV=$PLOT_CSV,TLS2TREES_STAGE1_AGGREGATE_CSV=$AGGREGATE_CSV"

SEMANTIC_JOB=$(sbatch --parsable --array="0-4%2" --export="$COMMON_EXPORTS" \
  methods/tls2trees/slurm/for_instance/prepare_semantic_development_stage1.sbatch)
SUBMITTED_JOBS+=("$SEMANTIC_JOB")
SUBMISSION_STATUS="semantic_array_submitted"
write_state
CANDIDATE_JOB=$(sbatch --parsable --array="0-19%4" \
  --dependency="afterok:$SEMANTIC_JOB" --kill-on-invalid-dep=yes \
  --export="$COMMON_EXPORTS" \
  methods/tls2trees/slurm/for_instance/evaluate_development_stage1_candidate.sbatch)
SUBMITTED_JOBS+=("$CANDIDATE_JOB")
SUBMISSION_STATUS="candidate_evaluation_array_submitted"
write_state
SUMMARY_JOB=$(sbatch --parsable --dependency="afterany:$CANDIDATE_JOB" \
  --kill-on-invalid-dep=yes --export="$COMMON_EXPORTS" \
  methods/tls2trees/slurm/for_instance/summarise_development_stage1.sbatch)
SUBMITTED_JOBS+=("$SUMMARY_JOB")
SUBMISSION_STATUS="stage1_chain_submitted"
write_state
printf '%s\n' "$STATE_FILE" > logs/tls2trees_for_instance/latest_stage1_state_file.txt
trap - ERR

echo "run_id=$RUN_ID"
echo "semantic_job=$SEMANTIC_JOB stage0_plots=5 gpu_concurrency=2"
echo "candidate_job=$CANDIDATE_JOB tasks=20 cpu_concurrency=4"
echo "summary_job=$SUMMARY_JOB"
echo "state_file=$STATE_FILE"
echo "development_accuracy_metrics_authorised=true"
echo "final_configuration_selected=false"
echo "held_out_test_accessed=false"
