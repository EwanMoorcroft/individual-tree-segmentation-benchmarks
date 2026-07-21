#!/usr/bin/env bash

set -euo pipefail

if [[ "${TLS2TREES_DEV_SMOKE_CONFIRMED:-0}" != "1" ]]; then
  echo "Refusing TLS2trees development-smoke submission." >&2
  echo "Set TLS2TREES_DEV_SMOKE_CONFIRMED=1 after reviewing the runbook." >&2
  exit 2
fi
if [[ "${TLS2TREES_REQUESTED_VARIANT:-published_default}" != "published_default" ]]; then
  echo "Only variant=published_default is available in this smoke route." >&2
  exit 2
fi
if [[ "${TLS2TREES_REQUESTED_SPLIT:-development}" != "development" ]]; then
  echo "Only split=development is available in this smoke route." >&2
  exit 2
fi

PROJECT_ROOT="${TLS2TREES_PROJECT_ROOT:-$HOME/scratch/tree-seg-benchmark}"
TREEBENCH_ENV="${TLS2TREES_TREEBENCH_ENV:-$HOME/fastscratch/venvs/treebench}"
METHOD_ENV="${TLS2TREES_METHOD_ENV:-$HOME/fastscratch/venvs/tls2trees}"
UPSTREAM_REPO="${TLS2TREES_UPSTREAM_REPO:-$PROJECT_ROOT/external/TLS2trees}"
DATASET_ROOT="${TLS2TREES_DATASET_ROOT:-$HOME/data/datasets/for_instance/FORinstance_dataset}"
EXPECTED_UPSTREAM_COMMIT="ca12cb73b2c736d80b020e8025f8d975d42e6f01"
EXPECTED_MODEL_SHA256="1a8bb6372394600f7c4b15f76beb98c32cb47ed25f8f729a84117ccfa410e72b"
export PYTHONNOUSERSITE=1
export LD_LIBRARY_PATH="$METHOD_ENV/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

cd "$PROJECT_ROOT"

test -d .git
test -x "$TREEBENCH_ENV/bin/python"
test -d "$UPSTREAM_REPO/.git"
test -d "$DATASET_ROOT"
test -f "$DATASET_ROOT/data_split_metadata.csv"

BENCHMARK_COMMIT=$(git rev-parse HEAD)
if [[ -n "$(git status --porcelain)" ]]; then
  echo "Refusing a dirty benchmark checkout; commit or remove changes first." >&2
  exit 2
fi
ACTUAL_UPSTREAM_COMMIT=$(git -C "$UPSTREAM_REPO" rev-parse HEAD)
if [[ "$ACTUAL_UPSTREAM_COMMIT" != "$EXPECTED_UPSTREAM_COMMIT" ]]; then
  echo "TLS2trees commit mismatch: expected $EXPECTED_UPSTREAM_COMMIT, found $ACTUAL_UPSTREAM_COMMIT." >&2
  exit 2
fi
if [[ -n "$(git -C "$UPSTREAM_REPO" status --porcelain)" ]]; then
  echo "Refusing a dirty upstream TLS2trees checkout." >&2
  exit 2
fi
MODEL="$UPSTREAM_REPO/tls2trees/fsct/model/model.pth"
test -f "$MODEL"
MODEL_SHA256=$(sha256sum "$MODEL" | awk '{print $1}')
if [[ "$MODEL_SHA256" != "$EXPECTED_MODEL_SHA256" ]]; then
  echo "Bundled FSCT model SHA-256 mismatch." >&2
  exit 2
fi

MANIFEST_CLI="${TLS2TREES_MANIFEST_CLI:-methods/tls2trees/scripts/data/prepare_for_instance_manifest.py}"
CONVERT_CLI="${TLS2TREES_CONVERT_CLI:-methods/tls2trees/scripts/data/prepare_for_instance_tls2trees_input.py}"
SEMANTIC_CLI="${TLS2TREES_SEMANTIC_CLI:-methods/tls2trees/scripts/runtime/run_for_instance_tls2trees_semantic.py}"
INSTANCE_CLI="${TLS2TREES_INSTANCE_CLI:-methods/tls2trees/scripts/runtime/run_for_instance_tls2trees_instance.py}"
ADAPTER_CLI="${TLS2TREES_ADAPTER_CLI:-methods/tls2trees/scripts/evaluation/adapt_for_instance_tls2trees_predictions.py}"
EVALUATE_CLI="${TLS2TREES_EVALUATE_CLI:-methods/tls2trees/scripts/evaluation/evaluate_for_instance_tls2trees_plot.py}"
GATE_CLI="${TLS2TREES_GATE_CLI:-methods/tls2trees/scripts/evaluation/validate_for_instance_tls2trees_smoke.py}"
ENV_VALIDATOR="${TLS2TREES_ENV_VALIDATOR:-methods/tls2trees/scripts/runtime/validate_tls2trees_environment.py}"
METHOD_ENV_MARKER="$METHOD_ENV/.tls2trees_setup_complete.json"
for cli in \
  "$MANIFEST_CLI" \
  "$CONVERT_CLI" \
  "$SEMANTIC_CLI" \
  "$INSTANCE_CLI" \
  "$ADAPTER_CLI" \
  "$EVALUATE_CLI" \
  "$GATE_CLI" \
  "$ENV_VALIDATOR"; do
  if [[ ! -f "$cli" ]]; then
    echo "Required smoke CLI is missing: $cli" >&2
    echo "No Slurm job was submitted." >&2
    exit 2
  fi
done
if [[ ! -f "$METHOD_ENV_MARKER" ]]; then
  echo "Validated TLS2trees environment marker is missing: $METHOD_ENV_MARKER" >&2
  echo "Run setup_tls2trees_environment.sbatch before submitting the smoke." >&2
  exit 2
fi
if [[ ! -x "$METHOD_ENV/bin/python" ]]; then
  echo "Validated TLS2trees Python is missing: $METHOD_ENV/bin/python" >&2
  echo "Run setup_tls2trees_environment.sbatch before submitting the smoke." >&2
  exit 2
fi
METHOD_ENV_MARKER_SHA256=$(sha256sum "$METHOD_ENV_MARKER" | awk '{print $1}')

"$TREEBENCH_ENV/bin/python" "$MANIFEST_CLI" --help >/dev/null
"$TREEBENCH_ENV/bin/python" "$CONVERT_CLI" --help >/dev/null
"$METHOD_ENV/bin/python" "$SEMANTIC_CLI" --help >/dev/null
"$METHOD_ENV/bin/python" "$INSTANCE_CLI" --help >/dev/null
"$TREEBENCH_ENV/bin/python" "$ADAPTER_CLI" --help >/dev/null
"$TREEBENCH_ENV/bin/python" "$EVALUATE_CLI" --help >/dev/null
"$TREEBENCH_ENV/bin/python" "$GATE_CLI" --help >/dev/null
"$METHOD_ENV/bin/python" -c "import laspy, networkx, numpy, pandas, scipy, sklearn, torch, torch_geometric, yaml"
"$TREEBENCH_ENV/bin/python" -c "import laspy, numpy, scipy, yaml"
"$METHOD_ENV/bin/python" "$ENV_VALIDATOR" \
  --tls2trees-repo "$UPSTREAM_REPO" \
  --setup-marker-json "$METHOD_ENV_MARKER" \
  --skip-model-load >/dev/null
TLS2TREES_REPO="$UPSTREAM_REPO" \
PYTHONPATH="$UPSTREAM_REPO:$UPSTREAM_REPO/tls2trees" \
  "$METHOD_ENV/bin/python" \
  methods/tls2trees/scripts/runtime/patches/semantic_patched.py --help >/dev/null
TLS2TREES_REPO="$UPSTREAM_REPO" \
PYTHONPATH="$UPSTREAM_REPO:$UPSTREAM_REPO/tls2trees" \
  "$METHOD_ENV/bin/python" \
  methods/tls2trees/scripts/runtime/patches/instance_patched.py --help >/dev/null
"$TREEBENCH_ENV/bin/python" -c 'import pathlib,sys,yaml; p=pathlib.Path(sys.argv[1]); c=yaml.safe_load(p.read_text()); gate=c.get("run_gate",{}); assert gate.get("runnable") is True, "published-default config run_gate.runnable is not true"' \
  methods/tls2trees/configs/for_instance_published_default.yml

MIN_FREE_BYTES="${TLS2TREES_SMOKE_MIN_FREE_BYTES:-53687091200}"
if [[ ! "$MIN_FREE_BYTES" =~ ^[1-9][0-9]*$ ]]; then
  echo "TLS2TREES_SMOKE_MIN_FREE_BYTES must be a positive integer." >&2
  exit 2
fi
FREE_BYTES=$(df -PB1 "$PROJECT_ROOT" | awk 'NR == 2 {print $4}')
if [[ ! "$FREE_BYTES" =~ ^[0-9]+$ ]] || ((FREE_BYTES < MIN_FREE_BYTES)); then
  echo "Need at least $MIN_FREE_BYTES free bytes under $PROJECT_ROOT; found ${FREE_BYTES:-unknown}." >&2
  exit 2
fi

STAMP="${TLS2TREES_SMOKE_STAMP:-$(date -u +%Y%m%d_%H%M%S)}"
if [[ ! "$STAMP" =~ ^[0-9]{8}_[0-9]{6}$ ]]; then
  echo "TLS2TREES_SMOKE_STAMP must use YYYYMMDD_HHMMSS." >&2
  exit 2
fi
RUN_ID="tls2trees_for-instance_published_default_development_smoke_$STAMP"
STAGE0_INDEX=0
OUTPUT_ROOT="$PROJECT_ROOT/data/predictions"
RUNTIME_ROOT="$OUTPUT_ROOT/tls2trees/for_instance/published_default/development/$RUN_ID"
WORKFLOW_METADATA_ROOT="$PROJECT_ROOT/results/metadata/tls2trees/for_instance/published_default/workflow/development/$RUN_ID"
WORKFLOW_TABLE_ROOT="$PROJECT_ROOT/results/tables/tls2trees/for_instance/published_default/workflow/development/$RUN_ID"
MANIFEST_JSON="$WORKFLOW_METADATA_ROOT/smoke_manifest.json"
STATE_DIR="$HOME/fastscratch/tls2trees_for_instance_smoke_states"
STATE_FILE="$STATE_DIR/${RUN_ID}.env"

for target in \
  "$RUNTIME_ROOT" \
  "$WORKFLOW_METADATA_ROOT" \
  "$WORKFLOW_TABLE_ROOT" \
  "$STATE_FILE"; do
  if [[ -e "$target" ]]; then
    echo "Refusing existing smoke-run path: $target" >&2
    exit 2
  fi
done

mkdir -p logs/tls2trees_for_instance "$STATE_DIR"

INVENTORY_JOB="not_submitted"
CONVERT_JOB="not_submitted"
SEMANTIC_JOB="not_submitted"
INSTANCE_JOB="not_submitted"
ADAPTER_JOB="not_submitted"
LEAF_OFF_EVALUATE_JOB="not_submitted"
LEAF_ON_EVALUATE_JOB="not_submitted"
GATE_JOB="not_submitted"
SUMMARY_JOB="not_submitted"
SUBMISSION_STATUS="preflight_completed"
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
    printf 'TLS2TREES_SMOKE_BENCHMARK_COMMIT=%q\n' "$BENCHMARK_COMMIT"
    printf 'TLS2TREES_SMOKE_UPSTREAM_COMMIT=%q\n' "$ACTUAL_UPSTREAM_COMMIT"
    printf 'TLS2TREES_SMOKE_MODEL_SHA256=%q\n' "$MODEL_SHA256"
    printf 'TLS2TREES_SMOKE_METHOD_ENV=%q\n' "$METHOD_ENV"
    printf 'TLS2TREES_SMOKE_METHOD_ENV_MARKER_SHA256=%q\n' "$METHOD_ENV_MARKER_SHA256"
    printf 'TLS2TREES_SMOKE_MANIFEST_JSON=%q\n' "$MANIFEST_JSON"
    printf 'TLS2TREES_SMOKE_STAGE0_INDEX=%q\n' "$STAGE0_INDEX"
    printf 'TLS2TREES_SMOKE_OUTPUT_ROOT=%q\n' "$OUTPUT_ROOT"
    printf 'TLS2TREES_SMOKE_WORKFLOW_METADATA_ROOT=%q\n' "$WORKFLOW_METADATA_ROOT"
    printf 'TLS2TREES_SMOKE_WORKFLOW_TABLE_ROOT=%q\n' "$WORKFLOW_TABLE_ROOT"
  } > "$STATE_FILE"
}

cancel_partial_submission() {
  local status=$?
  if ((${#SUBMITTED_JOBS[@]})); then
    scancel "${SUBMITTED_JOBS[@]}" 2>/dev/null || true
  fi
  SUBMISSION_STATUS="submission_failed_jobs_cancelled"
  write_state
  echo "Submission failed; jobs created by this attempt were cancelled." >&2
  echo "state_file=$STATE_FILE" >&2
  exit "$status"
}
trap cancel_partial_submission ERR

COMMON_EXPORTS="ALL,TLS2TREES_DEV_SMOKE_CONFIRMED=1,TLS2TREES_REQUESTED_VARIANT=published_default,TLS2TREES_REQUESTED_SPLIT=development,TLS2TREES_PROJECT_ROOT=$PROJECT_ROOT,TLS2TREES_TREEBENCH_ENV=$TREEBENCH_ENV,TLS2TREES_METHOD_ENV=$METHOD_ENV,TLS2TREES_EXPECTED_METHOD_ENV_MARKER_SHA256=$METHOD_ENV_MARKER_SHA256,TLS2TREES_UPSTREAM_REPO=$UPSTREAM_REPO,TLS2TREES_DATASET_ROOT=$DATASET_ROOT,TLS2TREES_RUN_ID=$RUN_ID,TLS2TREES_EXPECTED_BENCHMARK_COMMIT=$BENCHMARK_COMMIT,TLS2TREES_RUNTIME_ROOT=$RUNTIME_ROOT,TLS2TREES_WORKFLOW_METADATA_ROOT=$WORKFLOW_METADATA_ROOT,TLS2TREES_WORKFLOW_TABLE_ROOT=$WORKFLOW_TABLE_ROOT,TLS2TREES_MANIFEST_CLI=$MANIFEST_CLI,TLS2TREES_CONVERT_CLI=$CONVERT_CLI,TLS2TREES_SEMANTIC_CLI=$SEMANTIC_CLI,TLS2TREES_INSTANCE_CLI=$INSTANCE_CLI,TLS2TREES_ADAPTER_CLI=$ADAPTER_CLI,TLS2TREES_EVALUATE_CLI=$EVALUATE_CLI,TLS2TREES_GATE_CLI=$GATE_CLI,TLS2TREES_ENV_VALIDATOR=$ENV_VALIDATOR,MANIFEST_JSON=$MANIFEST_JSON,STAGE0_INDEX=$STAGE0_INDEX,RUN_ID=$RUN_ID,OUTPUT_ROOT=$OUTPUT_ROOT,TLS2TREES_REPO=$UPSTREAM_REPO"

INVENTORY_JOB=$(sbatch --parsable \
  --export="$COMMON_EXPORTS" \
  methods/tls2trees/slurm/for_instance/inventory_published_default_dev_smoke.sbatch)
SUBMITTED_JOBS+=("$INVENTORY_JOB")
SUBMISSION_STATUS="inventory_submitted"
write_state

CONVERT_JOB=$(sbatch --parsable \
  --dependency="afterok:$INVENTORY_JOB" \
  --kill-on-invalid-dep=yes \
  --export="$COMMON_EXPORTS" \
  methods/tls2trees/slurm/for_instance/convert_published_default_dev_smoke.sbatch)
SUBMITTED_JOBS+=("$CONVERT_JOB")
SUBMISSION_STATUS="conversion_submitted"
write_state

SEMANTIC_JOB=$(sbatch --parsable \
  --dependency="afterok:$CONVERT_JOB" \
  --kill-on-invalid-dep=yes \
  --export="$COMMON_EXPORTS" \
  methods/tls2trees/slurm/for_instance/semantic_published_default_dev_smoke.sbatch)
SUBMITTED_JOBS+=("$SEMANTIC_JOB")
SUBMISSION_STATUS="semantic_submitted"
write_state

INSTANCE_JOB=$(sbatch --parsable \
  --dependency="afterok:$SEMANTIC_JOB" \
  --kill-on-invalid-dep=yes \
  --export="$COMMON_EXPORTS" \
  methods/tls2trees/slurm/for_instance/instance_published_default_dev_smoke.sbatch)
SUBMITTED_JOBS+=("$INSTANCE_JOB")
SUBMISSION_STATUS="instance_submitted"
write_state

ADAPTER_JOB=$(sbatch --parsable \
  --dependency="afterok:$INSTANCE_JOB" \
  --kill-on-invalid-dep=yes \
  --export="$COMMON_EXPORTS" \
  methods/tls2trees/slurm/for_instance/adapt_published_default_dev_smoke.sbatch)
SUBMITTED_JOBS+=("$ADAPTER_JOB")
SUBMISSION_STATUS="prediction_adapter_submitted"
write_state

LEAF_OFF_EVALUATE_JOB=$(sbatch --parsable \
  --dependency="afterok:$ADAPTER_JOB" \
  --kill-on-invalid-dep=yes \
  --export="$COMMON_EXPORTS,TLS2TREES_TARGET=leaf_off" \
  methods/tls2trees/slurm/for_instance/evaluate_published_default_dev_smoke.sbatch)
SUBMITTED_JOBS+=("$LEAF_OFF_EVALUATE_JOB")
SUBMISSION_STATUS="leaf_off_evaluation_submitted"
write_state

LEAF_ON_EVALUATE_JOB=$(sbatch --parsable \
  --dependency="afterok:$ADAPTER_JOB" \
  --kill-on-invalid-dep=yes \
  --export="$COMMON_EXPORTS,TLS2TREES_TARGET=leaf_on" \
  methods/tls2trees/slurm/for_instance/evaluate_published_default_dev_smoke.sbatch)
SUBMITTED_JOBS+=("$LEAF_ON_EVALUATE_JOB")
SUBMISSION_STATUS="target_evaluations_submitted"
write_state

GATE_JOB=$(sbatch --parsable \
  --dependency="afterok:$LEAF_OFF_EVALUATE_JOB:$LEAF_ON_EVALUATE_JOB" \
  --kill-on-invalid-dep=yes \
  --export="$COMMON_EXPORTS" \
  methods/tls2trees/slurm/for_instance/gate_published_default_dev_smoke.sbatch)
SUBMITTED_JOBS+=("$GATE_JOB")
SUBMISSION_STATUS="gate_submitted"
write_state

SUMMARY_JOB=$(sbatch --parsable \
  --dependency="afterok:$GATE_JOB" \
  --kill-on-invalid-dep=yes \
  --export="$COMMON_EXPORTS" \
  methods/tls2trees/slurm/for_instance/summarise_published_default_dev_smoke.sbatch)
SUBMITTED_JOBS+=("$SUMMARY_JOB")
SUBMISSION_STATUS="chain_submitted"
write_state
trap - ERR

echo "run_id=$RUN_ID"
echo "inventory_job=$INVENTORY_JOB conversion_job=$CONVERT_JOB"
echo "semantic_job=$SEMANTIC_JOB instance_job=$INSTANCE_JOB adapter_job=$ADAPTER_JOB"
echo "leaf_off_evaluation_job=$LEAF_OFF_EVALUATE_JOB leaf_on_evaluation_job=$LEAF_ON_EVALUATE_JOB"
echo "gate_job=$GATE_JOB summary_job=$SUMMARY_JOB"
echo "state_file=$STATE_FILE"
echo "manifest_json=$MANIFEST_JSON"
echo "output_root=$OUTPUT_ROOT"
echo "No tuning, full-development array or held-out-test job was submitted."
