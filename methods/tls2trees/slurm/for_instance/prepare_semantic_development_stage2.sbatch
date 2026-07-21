#!/bin/bash -l
#SBATCH --job-name=tls2trees_dt_s2_semantic
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

cd "${TLS2TREES_PROJECT_ROOT:-$HOME/scratch/tree-seg-benchmark}"
test "${TLS2TREES_STAGE2_CONFIRMED:-0}" = "1"
test "${TLS2TREES_REQUESTED_VARIANT:-}" = "development_tuned"
test "${TLS2TREES_REQUESTED_SPLIT:-}" = "development"
test "$(git rev-parse HEAD)" = "$TLS2TREES_EXPECTED_BENCHMARK_COMMIT"
test -z "$(git status --porcelain)"
test "$(sha256sum "$TLS2TREES_MANIFEST_JSON" | awk '{print $1}')" = \
  "$TLS2TREES_MANIFEST_SHA256"

TASK_INDEX="$SLURM_ARRAY_TASK_ID"
SAFE_PLOT_ID=$("$TLS2TREES_TREEBENCH_ENV/bin/python" "$TLS2TREES_MANIFEST_CLI" resolve \
  --manifest-json "$TLS2TREES_MANIFEST_JSON" --expected-split development \
  --task-index "$TASK_INDEX" --field safe_plot_id)
PLOT_ROOT="$TLS2TREES_OUTPUT_ROOT/tls2trees/for_instance/development_tuned/development/$TLS2TREES_STAGE2_SEMANTIC_CACHE_RUN_ID/$SAFE_PLOT_ID"
test ! -e "$PLOT_ROOT"

module purge
module load miniforge3/25.3.0-python3.12.10
# shellcheck disable=SC1090
source "$TLS2TREES_TREEBENCH_ENV/bin/activate"
export PYTHONNOUSERSITE=1
python "$TLS2TREES_CONVERT_CLI" \
  --manifest-json "$TLS2TREES_MANIFEST_JSON" \
  --task-index "$TASK_INDEX" \
  --output-root "$TLS2TREES_OUTPUT_ROOT" \
  --run-id "$TLS2TREES_STAGE2_SEMANTIC_CACHE_RUN_ID" \
  --variant development_tuned \
  --split development

module purge
module load miniforge3/25.3.0-python3.12.10
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$TLS2TREES_METHOD_ENV"
export PYTHONNOUSERSITE=1
export LD_LIBRARY_PATH="$TLS2TREES_METHOD_ENV/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
test "$(sha256sum "$TLS2TREES_METHOD_ENV_MARKER" | awk '{print $1}')" = \
  "$TLS2TREES_METHOD_ENV_MARKER_SHA256"
python "$TLS2TREES_ENV_VALIDATOR" \
  --tls2trees-repo "$TLS2TREES_UPSTREAM_REPO" \
  --setup-marker-json "$TLS2TREES_METHOD_ENV_MARKER" \
  --require-cuda
python "$TLS2TREES_SEMANTIC_CLI" \
  --manifest-json "$TLS2TREES_MANIFEST_JSON" \
  --task-index "$TASK_INDEX" \
  --output-root "$TLS2TREES_OUTPUT_ROOT" \
  --run-id "$TLS2TREES_STAGE2_SEMANTIC_CACHE_RUN_ID" \
  --tls2trees-repo "$TLS2TREES_UPSTREAM_REPO" \
  --config methods/tls2trees/configs/for_instance_published_default.yml \
  --variant development_tuned \
  --split development

echo "status=stage2_semantic_cache_completed"
echo "task_index=$TASK_INDEX"
echo "safe_plot_id=$SAFE_PLOT_ID"
echo "held_out_test_accessed=false"
