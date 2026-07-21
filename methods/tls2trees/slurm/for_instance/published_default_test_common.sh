#!/usr/bin/env bash

# Shared fail-closed contract for the immutable published-default test chain.

pd_test_die() {
  echo "TLS2trees published-default test error: $*" >&2
  return 2
}

pd_test_require_file() {
  [[ -f "${1:?path required}" ]] || pd_test_die "missing ${2:?label required}: $1"
}

pd_test_load_contract() {
  TLS2TREES_PD_TEST_PROJECT_ROOT="${TLS2TREES_PD_TEST_PROJECT_ROOT:?required}"
  TLS2TREES_PD_TEST_TREEBENCH_ENV="${TLS2TREES_PD_TEST_TREEBENCH_ENV:?required}"
  TLS2TREES_PD_TEST_METHOD_ENV="${TLS2TREES_PD_TEST_METHOD_ENV:?required}"
  TLS2TREES_PD_TEST_METHOD_ENV_MARKER="${TLS2TREES_PD_TEST_METHOD_ENV_MARKER:?required}"
  TLS2TREES_PD_TEST_METHOD_ENV_MARKER_SHA256="${TLS2TREES_PD_TEST_METHOD_ENV_MARKER_SHA256:?required}"
  TLS2TREES_PD_TEST_UPSTREAM_REPO="${TLS2TREES_PD_TEST_UPSTREAM_REPO:?required}"
  TLS2TREES_PD_TEST_DATASET_ROOT="${TLS2TREES_PD_TEST_DATASET_ROOT:?required}"
  TLS2TREES_PD_TEST_SPLIT_METADATA_CSV="${TLS2TREES_PD_TEST_SPLIT_METADATA_CSV:?required}"
  TLS2TREES_PD_TEST_BENCHMARK_COMMIT="${TLS2TREES_PD_TEST_BENCHMARK_COMMIT:?required}"
  TLS2TREES_PD_TEST_UPSTREAM_COMMIT="${TLS2TREES_PD_TEST_UPSTREAM_COMMIT:?required}"
  TLS2TREES_PD_TEST_MODEL_SHA256="${TLS2TREES_PD_TEST_MODEL_SHA256:?required}"
  TLS2TREES_PD_TEST_RUN_ID="${TLS2TREES_PD_TEST_RUN_ID:?required}"
  TLS2TREES_PD_TEST_OUTPUT_ROOT="${TLS2TREES_PD_TEST_OUTPUT_ROOT:?required}"
  TLS2TREES_PD_TEST_MANIFEST_JSON="${TLS2TREES_PD_TEST_MANIFEST_JSON:?required}"
  TLS2TREES_PD_TEST_MANIFEST_CSV="${TLS2TREES_PD_TEST_MANIFEST_CSV:?required}"
  TLS2TREES_PD_TEST_MANIFEST_SHA256_FILE="${TLS2TREES_PD_TEST_MANIFEST_SHA256_FILE:?required}"
  TLS2TREES_PD_TEST_WORKFLOW_CONFIG="${TLS2TREES_PD_TEST_WORKFLOW_CONFIG:?required}"
  TLS2TREES_PD_TEST_WORKFLOW_CONFIG_SHA256="${TLS2TREES_PD_TEST_WORKFLOW_CONFIG_SHA256:?required}"
  TLS2TREES_PD_TEST_PUBLISHED_CONFIG="${TLS2TREES_PD_TEST_PUBLISHED_CONFIG:?required}"
  TLS2TREES_PD_TEST_PUBLISHED_CONFIG_SHA256="${TLS2TREES_PD_TEST_PUBLISHED_CONFIG_SHA256:?required}"
  TLS2TREES_PD_TEST_BENCHMARK_CONFIG="${TLS2TREES_PD_TEST_BENCHMARK_CONFIG:?required}"
  TLS2TREES_PD_TEST_BENCHMARK_CONFIG_SHA256="${TLS2TREES_PD_TEST_BENCHMARK_CONFIG_SHA256:?required}"
  TLS2TREES_PD_TEST_MANIFEST_CLI="${TLS2TREES_PD_TEST_MANIFEST_CLI:?required}"
  TLS2TREES_PD_TEST_CONVERT_CLI="${TLS2TREES_PD_TEST_CONVERT_CLI:?required}"
  TLS2TREES_PD_TEST_SEMANTIC_CLI="${TLS2TREES_PD_TEST_SEMANTIC_CLI:?required}"
  TLS2TREES_PD_TEST_INSTANCE_CLI="${TLS2TREES_PD_TEST_INSTANCE_CLI:?required}"
  TLS2TREES_PD_TEST_ADAPTER_CLI="${TLS2TREES_PD_TEST_ADAPTER_CLI:?required}"
  TLS2TREES_PD_TEST_EVALUATE_CLI="${TLS2TREES_PD_TEST_EVALUATE_CLI:?required}"
  TLS2TREES_PD_TEST_ENV_VALIDATOR="${TLS2TREES_PD_TEST_ENV_VALIDATOR:?required}"
  TLS2TREES_PD_TEST_CACHE_HELPER="${TLS2TREES_PD_TEST_CACHE_HELPER:?required}"
  TLS2TREES_PD_TEST_CACHE_AVAILABLE="${TLS2TREES_PD_TEST_CACHE_AVAILABLE:?required}"
  [[ "${TLS2TREES_PUBLISHED_DEFAULT_TEST_CONFIRMED:-0}" == "1" ]] || \
    pd_test_die "explicit held-out confirmation is missing"
  [[ "${TLS2TREES_REQUESTED_VARIANT:-}" == "published_default" ]] || \
    pd_test_die "variant must be published_default"
  [[ "${TLS2TREES_REQUESTED_SPLIT:-}" == "test" ]] || \
    pd_test_die "split must be test"
  [[ "$TLS2TREES_PD_TEST_RUN_ID" =~ ^tls2trees_for-instance_published_default_held_out_test_[0-9]{8}_[0-9]{6}$ ]] || \
    pd_test_die "unsafe run ID"

  cd "$TLS2TREES_PD_TEST_PROJECT_ROOT"
  [[ "$(git rev-parse HEAD)" == "$TLS2TREES_PD_TEST_BENCHMARK_COMMIT" ]] || \
    pd_test_die "benchmark commit changed after submission"
  [[ -z "$(git status --porcelain)" ]] || pd_test_die "benchmark checkout is dirty"
  [[ "$(git -C "$TLS2TREES_PD_TEST_UPSTREAM_REPO" rev-parse HEAD)" == "$TLS2TREES_PD_TEST_UPSTREAM_COMMIT" ]] || \
    pd_test_die "upstream commit changed"
  [[ -z "$(git -C "$TLS2TREES_PD_TEST_UPSTREAM_REPO" status --porcelain)" ]] || \
    pd_test_die "upstream checkout is dirty"
  pd_test_require_file "$TLS2TREES_PD_TEST_METHOD_ENV_MARKER" "environment marker"
  [[ "$(sha256sum "$TLS2TREES_PD_TEST_METHOD_ENV_MARKER" | awk '{print $1}')" == "$TLS2TREES_PD_TEST_METHOD_ENV_MARKER_SHA256" ]] || \
    pd_test_die "environment marker changed"
  [[ "$(sha256sum "$TLS2TREES_PD_TEST_WORKFLOW_CONFIG" | awk '{print $1}')" == "$TLS2TREES_PD_TEST_WORKFLOW_CONFIG_SHA256" ]] || \
    pd_test_die "workflow config changed"
  [[ "$(sha256sum "$TLS2TREES_PD_TEST_PUBLISHED_CONFIG" | awk '{print $1}')" == "$TLS2TREES_PD_TEST_PUBLISHED_CONFIG_SHA256" ]] || \
    pd_test_die "published config changed"
  [[ "$(sha256sum "$TLS2TREES_PD_TEST_BENCHMARK_CONFIG" | awk '{print $1}')" == "$TLS2TREES_PD_TEST_BENCHMARK_CONFIG_SHA256" ]] || \
    pd_test_die "benchmark config changed"
  local model="$TLS2TREES_PD_TEST_UPSTREAM_REPO/tls2trees/fsct/model/model.pth"
  pd_test_require_file "$model" "bundled model"
  [[ "$(sha256sum "$model" | awk '{print $1}')" == "$TLS2TREES_PD_TEST_MODEL_SHA256" ]] || \
    pd_test_die "bundled model changed"
}

pd_test_require_manifest() {
  pd_test_require_file "$TLS2TREES_PD_TEST_MANIFEST_JSON" "test manifest"
  pd_test_require_file "$TLS2TREES_PD_TEST_MANIFEST_SHA256_FILE" "manifest checksum"
  [[ "$(sha256sum "$TLS2TREES_PD_TEST_MANIFEST_JSON" | awk '{print $1}')" == "$(awk '{print $1}' "$TLS2TREES_PD_TEST_MANIFEST_SHA256_FILE")" ]] || \
    pd_test_die "test manifest changed"
}

pd_test_load_treebench() {
  module purge
  module load miniforge3/25.3.0-python3.12.10
  # shellcheck disable=SC1090
  source "$TLS2TREES_PD_TEST_TREEBENCH_ENV/bin/activate"
  export PYTHONNOUSERSITE=1
}

pd_test_load_method() {
  module purge
  module load miniforge3/25.3.0-python3.12.10
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate "$TLS2TREES_PD_TEST_METHOD_ENV"
  export PYTHONNOUSERSITE=1
  export LD_LIBRARY_PATH="$TLS2TREES_PD_TEST_METHOD_ENV/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
}

pd_test_resolve_plot() {
  local task_index="${1:?task index required}"
  local resolver=(
    "$TLS2TREES_PD_TEST_TREEBENCH_ENV/bin/python"
    "$TLS2TREES_PD_TEST_MANIFEST_CLI"
    resolve
    --manifest-json "$TLS2TREES_PD_TEST_MANIFEST_JSON"
    --expected-split test
    --allow-held-out-test
    --task-index "$task_index"
  )
  TLS2TREES_PD_TEST_SAFE_PLOT_ID=$("${resolver[@]}" --field safe_plot_id)
  TLS2TREES_PD_TEST_RELATIVE_PATH=$("${resolver[@]}" --field relative_path)
  TLS2TREES_PD_TEST_INPUT_LAS=$("${resolver[@]}" --field input_las)
  [[ "$TLS2TREES_PD_TEST_SAFE_PLOT_ID" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] || \
    pd_test_die "manifest returned an unsafe plot ID"
  export TLS2TREES_PD_TEST_SAFE_PLOT_ID TLS2TREES_PD_TEST_RELATIVE_PATH TLS2TREES_PD_TEST_INPUT_LAS
}
