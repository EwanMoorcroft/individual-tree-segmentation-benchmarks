#!/usr/bin/env bash

set -euo pipefail

if [[ "${TLS2TREES_LEAF_SCREEN_CONFIRMED:-0}" != "1" ]]; then
  echo "Refusing TLS2trees development leaf-screen submission." >&2
  echo "Set TLS2TREES_LEAF_SCREEN_CONFIRMED=1 after reviewing this fixed 3x3 grid." >&2
  exit 2
fi

SOURCE_STATE_FILE="${1:?Usage: submit_development_leaf_screen.sh <completed-stage1-state-file>}"
test -f "$SOURCE_STATE_FILE"
# shellcheck disable=SC1090
source "$SOURCE_STATE_FILE"

SOURCE_RUN_ID="${TLS2TREES_STAGE1_RUN_ID:?Stage 1 state has no run ID}"
DEVELOPMENT_EVIDENCE_JSON="${TLS2TREES_STAGE1_SUMMARY_JSON:?Stage 1 state has no summary}"
SOURCE_STAGE1_CONFIG="${TLS2TREES_STAGE1_CONFIG:?Stage 1 state has no config}"
MANIFEST_JSON="${TLS2TREES_STAGE1_MANIFEST_JSON:?Stage 1 state has no manifest}"
SOURCE_SEMANTIC_CACHE_RUN_ID="${TLS2TREES_STAGE1_SEMANTIC_CACHE_RUN_ID:?Stage 1 state has no semantic cache}"
OUTPUT_ROOT="${TLS2TREES_STAGE1_OUTPUT_ROOT:?Stage 1 state has no output root}"
TREEBENCH_ENV="${TLS2TREES_STAGE1_TREEBENCH_ENV:-$HOME/fastscratch/venvs/treebench}"

PROJECT_ROOT="${TLS2TREES_PROJECT_ROOT:-$HOME/scratch/tree-seg-benchmark}"
METHOD_ENV="${TLS2TREES_METHOD_ENV:-$HOME/fastscratch/venvs/tls2trees}"
METHOD_ENV_MARKER="$METHOD_ENV/.tls2trees_setup_complete.json"
UPSTREAM_REPO="${TLS2TREES_UPSTREAM_REPO:-$PROJECT_ROOT/external/TLS2trees}"
LEAF_SCREEN_CONFIG="methods/tls2trees/configs/for_instance_development_tuned_leaf_screen.yml"
SEARCH_SPACE_CONFIG="methods/tls2trees/configs/for_instance_search_space.yml"
MANIFEST_CLI="methods/tls2trees/scripts/data/prepare_for_instance_manifest.py"
CANDIDATE_CLI="methods/tls2trees/scripts/runtime/run_for_instance_tls2trees_development_candidate.py"
ADAPTER_CLI="methods/tls2trees/scripts/evaluation/adapt_for_instance_tls2trees_predictions.py"
EVALUATE_CLI="methods/tls2trees/scripts/evaluation/evaluate_for_instance_tls2trees_plot.py"
SUMMARY_CLI="methods/tls2trees/scripts/evaluation/summarise_tls2trees_development_leaf_screen.py"
ENV_VALIDATOR="methods/tls2trees/scripts/runtime/validate_tls2trees_environment.py"

cd "$PROJECT_ROOT"
test -d .git
test -z "$(git status --porcelain)"
BENCHMARK_COMMIT=$(git rev-parse HEAD)
test -x "$TREEBENCH_ENV/bin/python"
test -x "$METHOD_ENV/bin/python"
test -f "$METHOD_ENV_MARKER"
METHOD_ENV_MARKER_SHA256=$(sha256sum "$METHOD_ENV_MARKER" | awk '{print $1}')
test "$(git -C "$UPSTREAM_REPO" rev-parse HEAD)" = \
  "ca12cb73b2c736d80b020e8025f8d975d42e6f01"
test -z "$(git -C "$UPSTREAM_REPO" status --porcelain)"
for path in "$DEVELOPMENT_EVIDENCE_JSON" "$SOURCE_STAGE1_CONFIG" \
  "$MANIFEST_JSON" "$LEAF_SCREEN_CONFIG" "$SEARCH_SPACE_CONFIG" \
  "$MANIFEST_CLI" "$CANDIDATE_CLI" "$ADAPTER_CLI" "$EVALUATE_CLI" \
  "$SUMMARY_CLI" "$ENV_VALIDATOR"; do
  test -f "$path"
done

"$TREEBENCH_ENV/bin/python" "$MANIFEST_CLI" validate \
  --manifest-json "$MANIFEST_JSON" --expected-split development >/dev/null
MANIFEST_SHA256=$(sha256sum "$MANIFEST_JSON" | awk '{print $1}')
DEVELOPMENT_EVIDENCE_SHA256=$(sha256sum "$DEVELOPMENT_EVIDENCE_JSON" | awk '{print $1}')
SOURCE_STAGE1_CONFIG_SHA256=$(sha256sum "$SOURCE_STAGE1_CONFIG" | awk '{print $1}')
LEAF_SCREEN_CONFIG_SHA256=$(sha256sum "$LEAF_SCREEN_CONFIG" | awk '{print $1}')

"$TREEBENCH_ENV/bin/python" -c '
import itertools,json,sys,yaml
summary=json.load(open(sys.argv[1]))
source=yaml.safe_load(open(sys.argv[2]))
leaf=yaml.safe_load(open(sys.argv[3]))
search=yaml.safe_load(open(sys.argv[4]))
assert summary["status"] == "stage1_completed"
assert summary["split"] == "development"
assert summary["valid_metric_count"] == summary["expected_metric_count"] == 40
assert summary["held_out_test_accessed"] is False
assert summary["final_configuration_selected"] is False
assert summary["workflow_run_id"] == sys.argv[5]
assert summary["stage1_config_sha256"] == sys.argv[6]
p02=next(c for c in source["candidates"] if c["candidate_id"] == "p02_min_points_50")
assert summary["candidate_parameters"]["p02_min_points_50"] == p02["parameters"]
assert leaf["dataset"]["allowed_split"] == "development"
assert leaf["scope"]["targets"] == ["leaf_on"]
assert leaf["scope"]["held_out_test_accessed"] is False
assert leaf["scope"]["selection_uses_held_out_test_metrics"] is False
assert leaf["development_evidence"]["accuracy_used_to_construct_grid"] is False
voxel=search["searched_instance_parameters"]["add_leaves_voxel_length_m"]["values"]
edge=search["searched_instance_parameters"]["add_leaves_edge_length_m"]["values"]
assert leaf["leaf_attachment_grid"]["voxel_length_m"] == voxel
assert leaf["leaf_attachment_grid"]["edge_length_m"] == edge
expected=list(itertools.product(voxel,edge))
observed=[(c["parameters"]["add_leaves_voxel_length"],c["parameters"]["add_leaves_edge_length"]) for c in leaf["candidates"]]
assert observed == expected
leaf_keys={"add_leaves_voxel_length","add_leaves_edge_length"}
fixed={k:v for k,v in p02["parameters"].items() if k not in leaf_keys}
assert all({k:v for k,v in c["parameters"].items() if k not in leaf_keys} == fixed for c in leaf["candidates"])
assert leaf["run_gate"]["candidate_plot_task_count"] == 45
assert leaf["run_gate"]["semantic_jobs_submitted"] is False
assert leaf["run_gate"]["held_out_test_runnable"] is False
' "$DEVELOPMENT_EVIDENCE_JSON" "$SOURCE_STAGE1_CONFIG" \
  "$LEAF_SCREEN_CONFIG" "$SEARCH_SPACE_CONFIG" "$SOURCE_RUN_ID" \
  "$SOURCE_STAGE1_CONFIG_SHA256"

for STAGE0_INDEX in 0 1 2 3 4; do
  SAFE_PLOT_ID=$("$TREEBENCH_ENV/bin/python" "$MANIFEST_CLI" resolve-stage0 \
    --manifest-json "$MANIFEST_JSON" --stage0-index "$STAGE0_INDEX" --field safe_plot_id)
  SEMANTIC_METADATA="$OUTPUT_ROOT/tls2trees/for_instance/development_tuned/development/$SOURCE_SEMANTIC_CACHE_RUN_ID/$SAFE_PLOT_ID/metadata/semantic_run.json"
  test -f "$SEMANTIC_METADATA"
  "$TREEBENCH_ENV/bin/python" -c '
import json,sys
p=json.load(open(sys.argv[1]))
assert p["status"] == "completed"
assert p["split"] == "development"
assert p["variant"] == "development_tuned"
assert p["held_out_test_accessed"] is False
' "$SEMANTIC_METADATA"
done

MIN_FREE_BYTES="${TLS2TREES_LEAF_SCREEN_MIN_FREE_BYTES:-161061273600}"
FREE_BYTES=$(df -PB1 "$PROJECT_ROOT" | awk 'NR == 2 {print $4}')
if [[ ! "$FREE_BYTES" =~ ^[0-9]+$ ]] || ((FREE_BYTES < MIN_FREE_BYTES)); then
  echo "Need at least $MIN_FREE_BYTES free bytes; found ${FREE_BYTES:-unknown}." >&2
  exit 2
fi

STAMP="${TLS2TREES_LEAF_SCREEN_STAMP:-$(date -u +%Y%m%d_%H%M%S)}"
if [[ ! "$STAMP" =~ ^[0-9]{8}_[0-9]{6}$ ]]; then
  echo "TLS2TREES_LEAF_SCREEN_STAMP must use YYYYMMDD_HHMMSS." >&2
  exit 2
fi
RUN_ID="tls2trees_for-instance_development_tuned_leaf_screen_$STAMP"
WORKFLOW_ROOT="$PROJECT_ROOT/results/metadata/tls2trees/for_instance/development_tuned/leaf_screen/$RUN_ID"
TABLE_ROOT="$PROJECT_ROOT/results/tables/tls2trees/for_instance/development_tuned/leaf_screen/$RUN_ID"
SUMMARY_JSON="$WORKFLOW_ROOT/leaf_screen_summary.json"
PLOT_CSV="$TABLE_ROOT/plot_metrics.csv"
AGGREGATE_CSV="$TABLE_ROOT/candidate_summary.csv"
STATE_DIR="$HOME/fastscratch/tls2trees_for_instance_leaf_screen_states"
STATE_FILE="$STATE_DIR/$RUN_ID.env"
for path in "$WORKFLOW_ROOT" "$TABLE_ROOT" "$STATE_FILE"; do
  test ! -e "$path"
done
mkdir -p logs/tls2trees_for_instance "$STATE_DIR"

CANDIDATE_JOB="not_submitted"
SUMMARY_JOB="not_submitted"
SUBMISSION_STATUS="development_only_preflight_completed"
SUBMITTED_JOBS=()
write_state() {
  {
    printf 'TLS2TREES_LEAF_SCREEN_RUN_ID=%q\n' "$RUN_ID"
    printf 'TLS2TREES_LEAF_SCREEN_SUBMISSION_STATUS=%q\n' "$SUBMISSION_STATUS"
    printf 'TLS2TREES_LEAF_SCREEN_CANDIDATE_JOB=%q\n' "$CANDIDATE_JOB"
    printf 'TLS2TREES_LEAF_SCREEN_SUMMARY_JOB=%q\n' "$SUMMARY_JOB"
    printf 'TLS2TREES_LEAF_SCREEN_BENCHMARK_COMMIT=%q\n' "$BENCHMARK_COMMIT"
    printf 'TLS2TREES_LEAF_SCREEN_SOURCE_STATE=%q\n' "$SOURCE_STATE_FILE"
    printf 'TLS2TREES_LEAF_SCREEN_SOURCE_RUN_ID=%q\n' "$SOURCE_RUN_ID"
    printf 'TLS2TREES_LEAF_SCREEN_SOURCE_SEMANTIC_CACHE_RUN_ID=%q\n' "$SOURCE_SEMANTIC_CACHE_RUN_ID"
    printf 'TLS2TREES_LEAF_SCREEN_DEVELOPMENT_EVIDENCE_JSON=%q\n' "$DEVELOPMENT_EVIDENCE_JSON"
    printf 'TLS2TREES_LEAF_SCREEN_DEVELOPMENT_EVIDENCE_SHA256=%q\n' "$DEVELOPMENT_EVIDENCE_SHA256"
    printf 'TLS2TREES_LEAF_SCREEN_MANIFEST_JSON=%q\n' "$MANIFEST_JSON"
    printf 'TLS2TREES_LEAF_SCREEN_MANIFEST_SHA256=%q\n' "$MANIFEST_SHA256"
    printf 'TLS2TREES_LEAF_SCREEN_CONFIG=%q\n' "$LEAF_SCREEN_CONFIG"
    printf 'TLS2TREES_LEAF_SCREEN_CONFIG_SHA256=%q\n' "$LEAF_SCREEN_CONFIG_SHA256"
    printf 'TLS2TREES_LEAF_SCREEN_OUTPUT_ROOT=%q\n' "$OUTPUT_ROOT"
    printf 'TLS2TREES_LEAF_SCREEN_SUMMARY_JSON=%q\n' "$SUMMARY_JSON"
    printf 'TLS2TREES_LEAF_SCREEN_PLOT_CSV=%q\n' "$PLOT_CSV"
    printf 'TLS2TREES_LEAF_SCREEN_AGGREGATE_CSV=%q\n' "$AGGREGATE_CSV"
    printf 'TLS2TREES_LEAF_SCREEN_TREEBENCH_ENV=%q\n' "$TREEBENCH_ENV"
  } > "$STATE_FILE"
}
cancel_partial_submission() {
  local status=$?
  if ((${#SUBMITTED_JOBS[@]})); then
    scancel "${SUBMITTED_JOBS[@]}" 2>/dev/null || true
  fi
  SUBMISSION_STATUS="leaf_screen_submission_failed_jobs_cancelled"
  write_state
  echo "Leaf-screen submission failed; new jobs were cancelled." >&2
  echo "state_file=$STATE_FILE" >&2
  exit "$status"
}
trap cancel_partial_submission ERR

COMMON_EXPORTS="ALL,TLS2TREES_LEAF_SCREEN_CONFIRMED=1,TLS2TREES_REQUESTED_VARIANT=development_tuned,TLS2TREES_REQUESTED_SPLIT=development,TLS2TREES_PROJECT_ROOT=$PROJECT_ROOT,TLS2TREES_TREEBENCH_ENV=$TREEBENCH_ENV,TLS2TREES_METHOD_ENV=$METHOD_ENV,TLS2TREES_METHOD_ENV_MARKER=$METHOD_ENV_MARKER,TLS2TREES_METHOD_ENV_MARKER_SHA256=$METHOD_ENV_MARKER_SHA256,TLS2TREES_UPSTREAM_REPO=$UPSTREAM_REPO,TLS2TREES_EXPECTED_BENCHMARK_COMMIT=$BENCHMARK_COMMIT,TLS2TREES_MANIFEST_JSON=$MANIFEST_JSON,TLS2TREES_MANIFEST_SHA256=$MANIFEST_SHA256,TLS2TREES_MANIFEST_CLI=$MANIFEST_CLI,TLS2TREES_CANDIDATE_CLI=$CANDIDATE_CLI,TLS2TREES_ADAPTER_CLI=$ADAPTER_CLI,TLS2TREES_EVALUATE_CLI=$EVALUATE_CLI,TLS2TREES_ENV_VALIDATOR=$ENV_VALIDATOR,TLS2TREES_LEAF_SCREEN_CONFIG=$LEAF_SCREEN_CONFIG,TLS2TREES_LEAF_SCREEN_CONFIG_SHA256=$LEAF_SCREEN_CONFIG_SHA256,TLS2TREES_DEVELOPMENT_EVIDENCE_JSON=$DEVELOPMENT_EVIDENCE_JSON,TLS2TREES_DEVELOPMENT_EVIDENCE_SHA256=$DEVELOPMENT_EVIDENCE_SHA256,TLS2TREES_OUTPUT_ROOT=$OUTPUT_ROOT,TLS2TREES_SOURCE_SEMANTIC_CACHE_RUN_ID=$SOURCE_SEMANTIC_CACHE_RUN_ID,TLS2TREES_LEAF_SCREEN_RUN_ID=$RUN_ID,TLS2TREES_LEAF_SCREEN_SUMMARY_JSON=$SUMMARY_JSON,TLS2TREES_LEAF_SCREEN_PLOT_CSV=$PLOT_CSV,TLS2TREES_LEAF_SCREEN_AGGREGATE_CSV=$AGGREGATE_CSV"

CANDIDATE_JOB=$(sbatch --parsable --array="0-44%4" \
  --export="$COMMON_EXPORTS" \
  methods/tls2trees/slurm/for_instance/evaluate_development_leaf_screen_candidate.sbatch)
SUBMITTED_JOBS+=("$CANDIDATE_JOB")
SUBMISSION_STATUS="leaf_screen_candidate_array_submitted"
write_state
SUMMARY_JOB=$(sbatch --parsable --dependency="afterok:$CANDIDATE_JOB" \
  --kill-on-invalid-dep=yes --export="$COMMON_EXPORTS" \
  methods/tls2trees/slurm/for_instance/summarise_development_leaf_screen.sbatch)
SUBMITTED_JOBS+=("$SUMMARY_JOB")
SUBMISSION_STATUS="development_leaf_screen_chain_submitted"
write_state
printf '%s\n' "$STATE_FILE" > \
  logs/tls2trees_for_instance/latest_leaf_screen_state_file.txt
trap - ERR

echo "run_id=$RUN_ID"
echo "source_stage1_run_id=$SOURCE_RUN_ID"
echo "semantic_cache_run_id=$SOURCE_SEMANTIC_CACHE_RUN_ID"
echo "semantic_jobs_submitted=false"
echo "candidate_job=$CANDIDATE_JOB tasks=45 cpu_concurrency=4"
echo "summary_job=$SUMMARY_JOB"
echo "state_file=$STATE_FILE"
echo "target=leaf_on"
echo "expected_metrics=45"
echo "final_configuration_selected=false"
echo "held_out_test_accessed=false"
