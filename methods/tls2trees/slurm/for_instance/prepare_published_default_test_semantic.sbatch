#!/bin/bash -l
#SBATCH --job-name=tls2trees_pd_test_semantic
#SBATCH --partition=gpu-l40s-low
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=192G
#SBATCH --no-requeue
#SBATCH --output=logs/tls2trees_for_instance/%x_%A_%a.out
#SBATCH --error=logs/tls2trees_for_instance/%x_%A_%a.err

set -euo pipefail
cd "${TLS2TREES_PD_TEST_PROJECT_ROOT:-$HOME/scratch/tree-seg-benchmark}"
# shellcheck disable=SC1091
source methods/tls2trees/slurm/for_instance/published_default_test_common.sh
pd_test_load_contract
pd_test_require_manifest
TASK_INDEX="$SLURM_ARRAY_TASK_ID"
pd_test_load_treebench
pd_test_resolve_plot "$TASK_INDEX"

if [[ "$TLS2TREES_PD_TEST_CACHE_AVAILABLE" == "1" ]]; then
  set +e
  python "$TLS2TREES_PD_TEST_CACHE_HELPER" \
    --manifest-json "$TLS2TREES_PD_TEST_MANIFEST_JSON" \
    --source-manifest-json "$TLS2TREES_PD_TEST_CACHE_MANIFEST_JSON" \
    --task-index "$TASK_INDEX" \
    --source-output-root "$TLS2TREES_PD_TEST_CACHE_OUTPUT_ROOT" \
    --source-run-id "$TLS2TREES_PD_TEST_CACHE_RUN_ID" \
    --source-variant "$TLS2TREES_PD_TEST_CACHE_VARIANT" \
    --output-root "$TLS2TREES_PD_TEST_OUTPUT_ROOT" \
    --run-id "$TLS2TREES_PD_TEST_RUN_ID" \
    --workflow-config "$TLS2TREES_PD_TEST_WORKFLOW_CONFIG" \
    --published-config "$TLS2TREES_PD_TEST_PUBLISHED_CONFIG" \
    --tls2trees-repo "$TLS2TREES_PD_TEST_UPSTREAM_REPO" \
    --source-state-file "$TLS2TREES_PD_TEST_CACHE_STATE_FILE" \
    --source-state-sha256 "$TLS2TREES_PD_TEST_CACHE_STATE_SHA256"
  CACHE_RC=$?
  set -e
  if [[ "$CACHE_RC" == "0" ]]; then
    echo "status=published_default_test_semantic_completed"
    echo "task_index=$TASK_INDEX"
    echo "safe_plot_id=$TLS2TREES_PD_TEST_SAFE_PLOT_ID"
    echo "semantic_cache_reused=true"
    echo "held_out_test_accessed=true"
    exit 0
  fi
  [[ "$CACHE_RC" == "3" ]] || pd_test_die "cache helper failed with status $CACHE_RC"
  echo "semantic_cache_reused=false"
  echo "semantic_fallback=dedicated_gpu_inference"
fi

python "$TLS2TREES_PD_TEST_CONVERT_CLI" \
  --manifest-json "$TLS2TREES_PD_TEST_MANIFEST_JSON" \
  --task-index "$TASK_INDEX" \
  --output-root "$TLS2TREES_PD_TEST_OUTPUT_ROOT" \
  --run-id "$TLS2TREES_PD_TEST_RUN_ID" \
  --variant published_default --split test --allow-held-out-test

pd_test_load_method
test "$(sha256sum "$TLS2TREES_PD_TEST_METHOD_ENV_MARKER" | awk '{print $1}')" = \
  "$TLS2TREES_PD_TEST_METHOD_ENV_MARKER_SHA256"
python "$TLS2TREES_PD_TEST_ENV_VALIDATOR" \
  --tls2trees-repo "$TLS2TREES_PD_TEST_UPSTREAM_REPO" \
  --setup-marker-json "$TLS2TREES_PD_TEST_METHOD_ENV_MARKER" \
  --require-cuda
python "$TLS2TREES_PD_TEST_SEMANTIC_CLI" \
  --manifest-json "$TLS2TREES_PD_TEST_MANIFEST_JSON" \
  --task-index "$TASK_INDEX" \
  --output-root "$TLS2TREES_PD_TEST_OUTPUT_ROOT" \
  --run-id "$TLS2TREES_PD_TEST_RUN_ID" \
  --tls2trees-repo "$TLS2TREES_PD_TEST_UPSTREAM_REPO" \
  --config "$TLS2TREES_PD_TEST_PUBLISHED_CONFIG" \
  --variant published_default --split test --allow-held-out-test

echo "status=published_default_test_semantic_completed"
echo "task_index=$TASK_INDEX"
echo "safe_plot_id=$TLS2TREES_PD_TEST_SAFE_PLOT_ID"
echo "semantic_cache_reused=false"
echo "held_out_test_accessed=true"
