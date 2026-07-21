#!/usr/bin/env bash

# Shared guards and paths for the development-only TLS2trees smoke workflow.
# This file is sourced by Slurm scripts; it is not a submission entry point.

TLS2TREES_EXPECTED_UPSTREAM_COMMIT="ca12cb73b2c736d80b020e8025f8d975d42e6f01"
TLS2TREES_EXPECTED_MODEL_SHA256="1a8bb6372394600f7c4b15f76beb98c32cb47ed25f8f729a84117ccfa410e72b"
TLS2TREES_BENCHMARK_CONFIG="methods/tls2trees/configs/for_instance_benchmark.yml"
TLS2TREES_METHOD_CONFIG="methods/tls2trees/configs/for_instance_published_default.yml"
TLS2TREES_VARIANT="published_default"
TLS2TREES_SPLIT="development"

tls2trees_die() {
  echo "TLS2trees smoke error: $*" >&2
  return 2
}

tls2trees_require_file() {
  local path="${1:?path is required}"
  local label="${2:?label is required}"
  [[ -f "$path" ]] || tls2trees_die "missing $label: $path"
}

tls2trees_require_dir() {
  local path="${1:?path is required}"
  local label="${2:?label is required}"
  [[ -d "$path" ]] || tls2trees_die "missing $label: $path"
}

tls2trees_require_executable() {
  local path="${1:?path is required}"
  local label="${2:?label is required}"
  [[ -x "$path" ]] || tls2trees_die "missing executable $label: $path"
}

tls2trees_load_contract() {
  TLS2TREES_PROJECT_ROOT="${TLS2TREES_PROJECT_ROOT:-$HOME/scratch/tree-seg-benchmark}"
  TLS2TREES_TREEBENCH_ENV="${TLS2TREES_TREEBENCH_ENV:-$HOME/fastscratch/venvs/treebench}"
  TLS2TREES_METHOD_ENV="${TLS2TREES_METHOD_ENV:-$HOME/fastscratch/venvs/tls2trees}"
  TLS2TREES_METHOD_ENV_MARKER="$TLS2TREES_METHOD_ENV/.tls2trees_setup_complete.json"
  TLS2TREES_EXPECTED_METHOD_ENV_MARKER_SHA256="${TLS2TREES_EXPECTED_METHOD_ENV_MARKER_SHA256:?TLS2TREES_EXPECTED_METHOD_ENV_MARKER_SHA256 is required}"
  TLS2TREES_UPSTREAM_REPO="${TLS2TREES_UPSTREAM_REPO:-$TLS2TREES_PROJECT_ROOT/external/TLS2trees}"
  TLS2TREES_DATASET_ROOT="${TLS2TREES_DATASET_ROOT:-$HOME/data/datasets/for_instance/FORinstance_dataset}"
  TLS2TREES_RUN_ID="${TLS2TREES_RUN_ID:?TLS2TREES_RUN_ID is required}"
  TLS2TREES_EXPECTED_BENCHMARK_COMMIT="${TLS2TREES_EXPECTED_BENCHMARK_COMMIT:?TLS2TREES_EXPECTED_BENCHMARK_COMMIT is required}"
  TLS2TREES_RUNTIME_ROOT="${TLS2TREES_RUNTIME_ROOT:?TLS2TREES_RUNTIME_ROOT is required}"
  TLS2TREES_WORKFLOW_METADATA_ROOT="${TLS2TREES_WORKFLOW_METADATA_ROOT:?TLS2TREES_WORKFLOW_METADATA_ROOT is required}"
  TLS2TREES_WORKFLOW_TABLE_ROOT="${TLS2TREES_WORKFLOW_TABLE_ROOT:?TLS2TREES_WORKFLOW_TABLE_ROOT is required}"
  MANIFEST_JSON="${MANIFEST_JSON:?MANIFEST_JSON is required}"
  STAGE0_INDEX="${STAGE0_INDEX:?STAGE0_INDEX is required}"
  TASK_INDEX="${TASK_INDEX:-}"
  RUN_ID="${RUN_ID:?RUN_ID is required}"
  OUTPUT_ROOT="${OUTPUT_ROOT:?OUTPUT_ROOT is required}"
  TLS2TREES_REPO="${TLS2TREES_REPO:?TLS2TREES_REPO is required}"

  TLS2TREES_MANIFEST_CLI="${TLS2TREES_MANIFEST_CLI:-methods/tls2trees/scripts/data/prepare_for_instance_manifest.py}"
  TLS2TREES_CONVERT_CLI="${TLS2TREES_CONVERT_CLI:-methods/tls2trees/scripts/data/prepare_for_instance_tls2trees_input.py}"
  TLS2TREES_SEMANTIC_CLI="${TLS2TREES_SEMANTIC_CLI:-methods/tls2trees/scripts/runtime/run_for_instance_tls2trees_semantic.py}"
  TLS2TREES_INSTANCE_CLI="${TLS2TREES_INSTANCE_CLI:-methods/tls2trees/scripts/runtime/run_for_instance_tls2trees_instance.py}"
  TLS2TREES_ADAPTER_CLI="${TLS2TREES_ADAPTER_CLI:-methods/tls2trees/scripts/evaluation/adapt_for_instance_tls2trees_predictions.py}"
  TLS2TREES_EVALUATE_CLI="${TLS2TREES_EVALUATE_CLI:-methods/tls2trees/scripts/evaluation/evaluate_for_instance_tls2trees_plot.py}"
  TLS2TREES_GATE_CLI="${TLS2TREES_GATE_CLI:-methods/tls2trees/scripts/evaluation/validate_for_instance_tls2trees_smoke.py}"
  TLS2TREES_ENV_VALIDATOR="${TLS2TREES_ENV_VALIDATOR:-methods/tls2trees/scripts/runtime/validate_tls2trees_environment.py}"
  TLS2TREES_SMOKE_MANIFEST="$MANIFEST_JSON"
  TLS2TREES_INVENTORY_METADATA="$TLS2TREES_WORKFLOW_METADATA_ROOT/inventory.json"
  TLS2TREES_GATE_METADATA="$TLS2TREES_WORKFLOW_METADATA_ROOT/gate.json"
  TLS2TREES_RUN_SUMMARY="$TLS2TREES_WORKFLOW_TABLE_ROOT/run_summary.json"
}

tls2trees_validate_contract() {
  [[ "${TLS2TREES_DEV_SMOKE_CONFIRMED:-0}" == "1" ]] || \
    tls2trees_die "TLS2TREES_DEV_SMOKE_CONFIRMED must equal 1"
  [[ "${TLS2TREES_REQUESTED_VARIANT:-published_default}" == "$TLS2TREES_VARIANT" ]] || \
    tls2trees_die "only variant=published_default is permitted"
  [[ "${TLS2TREES_REQUESTED_SPLIT:-development}" == "$TLS2TREES_SPLIT" ]] || \
    tls2trees_die "only split=development is permitted"
  [[ "$TLS2TREES_RUN_ID" =~ ^tls2trees_for-instance_published_default_development_smoke_[0-9]{8}_[0-9]{6}$ ]] || \
    tls2trees_die "unsafe or non-smoke run ID: $TLS2TREES_RUN_ID"
  [[ "$RUN_ID" == "$TLS2TREES_RUN_ID" ]] || tls2trees_die "RUN_ID alias mismatch"
  [[ "$STAGE0_INDEX" =~ ^[0-4]$ ]] || tls2trees_die "STAGE0_INDEX must be in 0..4"
  if [[ -n "$TASK_INDEX" && ! "$TASK_INDEX" =~ ^[0-9]+$ ]]; then
    tls2trees_die "TASK_INDEX must be a non-negative integer"
  fi
  [[ "$OUTPUT_ROOT/tls2trees/for_instance/$TLS2TREES_VARIANT/$TLS2TREES_SPLIT/$RUN_ID" == "$TLS2TREES_RUNTIME_ROOT" ]] || \
    tls2trees_die "runtime root does not match the immutable route"
  [[ "$TLS2TREES_REPO" == "$TLS2TREES_UPSTREAM_REPO" ]] || tls2trees_die "TLS2TREES_REPO alias mismatch"

  tls2trees_require_dir "$TLS2TREES_PROJECT_ROOT/.git" "benchmark Git checkout"
  tls2trees_require_executable "$TLS2TREES_TREEBENCH_ENV/bin/python" "treebench Python"
  tls2trees_require_executable "$TLS2TREES_METHOD_ENV/bin/python" "TLS2trees Python"
  tls2trees_require_file "$TLS2TREES_METHOD_ENV_MARKER" "validated TLS2trees environment marker"
  tls2trees_require_file "$TLS2TREES_PROJECT_ROOT/$TLS2TREES_ENV_VALIDATOR" "TLS2trees environment validator"
  tls2trees_require_dir "$TLS2TREES_UPSTREAM_REPO/.git" "TLS2trees Git checkout"
  tls2trees_require_dir "$TLS2TREES_DATASET_ROOT" "FOR-instance dataset root"
  tls2trees_require_file "$TLS2TREES_DATASET_ROOT/data_split_metadata.csv" "dataset split metadata"
  tls2trees_require_file "$TLS2TREES_PROJECT_ROOT/$TLS2TREES_BENCHMARK_CONFIG" "benchmark config"
  tls2trees_require_file "$TLS2TREES_PROJECT_ROOT/$TLS2TREES_METHOD_CONFIG" "published-default config"

  local actual_benchmark_commit
  actual_benchmark_commit=$(git -C "$TLS2TREES_PROJECT_ROOT" rev-parse HEAD)
  [[ "$actual_benchmark_commit" == "$TLS2TREES_EXPECTED_BENCHMARK_COMMIT" ]] || \
    tls2trees_die "benchmark checkout changed after submission"
  [[ -z "$(git -C "$TLS2TREES_PROJECT_ROOT" status --porcelain)" ]] || \
    tls2trees_die "benchmark checkout is dirty"

  local actual_upstream_commit
  actual_upstream_commit=$(git -C "$TLS2TREES_UPSTREAM_REPO" rev-parse HEAD)
  [[ "$actual_upstream_commit" == "$TLS2TREES_EXPECTED_UPSTREAM_COMMIT" ]] || \
    tls2trees_die "upstream commit mismatch: expected $TLS2TREES_EXPECTED_UPSTREAM_COMMIT, found $actual_upstream_commit"
  [[ -z "$(git -C "$TLS2TREES_UPSTREAM_REPO" status --porcelain)" ]] || \
    tls2trees_die "upstream TLS2trees checkout is dirty"

  local model="$TLS2TREES_UPSTREAM_REPO/tls2trees/fsct/model/model.pth"
  tls2trees_require_file "$model" "bundled FSCT model"
  local actual_model_sha256
  actual_model_sha256=$(sha256sum "$model" | awk '{print $1}')
  [[ "$actual_model_sha256" == "$TLS2TREES_EXPECTED_MODEL_SHA256" ]] || \
    tls2trees_die "bundled model SHA-256 mismatch"

  local actual_method_env_marker_sha256
  actual_method_env_marker_sha256=$(sha256sum "$TLS2TREES_METHOD_ENV_MARKER" | awk '{print $1}')
  [[ "$actual_method_env_marker_sha256" == "$TLS2TREES_EXPECTED_METHOD_ENV_MARKER_SHA256" ]] || \
    tls2trees_die "TLS2trees environment marker changed after submission"
}

tls2trees_load_treebench_env() {
  module purge
  module load miniforge3/25.3.0-python3.12.10
  # shellcheck disable=SC1090
  source "$TLS2TREES_TREEBENCH_ENV/bin/activate"
  export PYTHONNOUSERSITE=1
}

tls2trees_load_method_env() {
  module purge
  module load miniforge3/25.3.0-python3.12.10
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate "$TLS2TREES_METHOD_ENV"
  export PYTHONNOUSERSITE=1
  export LD_LIBRARY_PATH="$TLS2TREES_METHOD_ENV/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
}

tls2trees_resolve_stage0_context() {
  tls2trees_require_file "$MANIFEST_JSON" "development manifest"
  tls2trees_require_file "$TLS2TREES_PROJECT_ROOT/$TLS2TREES_MANIFEST_CLI" "manifest CLI"
  local resolver=(
    "$TLS2TREES_TREEBENCH_ENV/bin/python"
    "$TLS2TREES_PROJECT_ROOT/$TLS2TREES_MANIFEST_CLI"
    resolve-stage0
    --manifest-json "$MANIFEST_JSON"
    --stage0-index "$STAGE0_INDEX"
  )
  TASK_INDEX=$("${resolver[@]}" --field task_index)
  SAFE_PLOT_ID=$("${resolver[@]}" --field safe_plot_id)
  RELATIVE_PATH=$("${resolver[@]}" --field relative_path)
  INPUT_LAS=$("${resolver[@]}" --field input_las)
  [[ "$TASK_INDEX" =~ ^[0-9]+$ ]] || tls2trees_die "manifest resolver returned an invalid task index"
  [[ "$SAFE_PLOT_ID" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] || tls2trees_die "manifest resolver returned an unsafe plot ID"
  [[ "$RELATIVE_PATH" != /* && "$RELATIVE_PATH" != *".."* ]] || tls2trees_die "manifest resolver returned an unsafe relative path"
  PLOT_ROOT="$OUTPUT_ROOT/tls2trees/for_instance/$TLS2TREES_VARIANT/$TLS2TREES_SPLIT/$RUN_ID/$SAFE_PLOT_ID"
  export TASK_INDEX SAFE_PLOT_ID RELATIVE_PATH INPUT_LAS PLOT_ROOT
}

tls2trees_stage_banner() {
  local stage="${1:?stage is required}"
  echo "stage=$stage"
  echo "run_id=$TLS2TREES_RUN_ID"
  echo "variant=$TLS2TREES_VARIANT"
  echo "split=$TLS2TREES_SPLIT"
  echo "stage0_index=$STAGE0_INDEX"
  echo "task_index=${TASK_INDEX:-unresolved}"
  echo "relative_path=${RELATIVE_PATH:-unresolved}"
  echo "benchmark_commit=$TLS2TREES_EXPECTED_BENCHMARK_COMMIT"
  echo "upstream_commit=$TLS2TREES_EXPECTED_UPSTREAM_COMMIT"
  echo "method_env=$TLS2TREES_METHOD_ENV"
  echo "method_env_marker_sha256=$TLS2TREES_EXPECTED_METHOD_ENV_MARKER_SHA256"
  echo "job_id=${SLURM_JOB_ID:-manual}"
  echo "hostname=$(hostname)"
  echo "started_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
}
