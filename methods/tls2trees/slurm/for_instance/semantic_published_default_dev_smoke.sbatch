#!/bin/bash -l
#SBATCH --job-name=tls2trees_smoke_semantic
#SBATCH --partition=gpu-l40s-low
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --no-requeue
#SBATCH --output=logs/tls2trees_for_instance/%x_%j.out
#SBATCH --error=logs/tls2trees_for_instance/%x_%j.err

set -euo pipefail

cd "${TLS2TREES_PROJECT_ROOT:-$HOME/scratch/tree-seg-benchmark}"
# shellcheck disable=SC1091
source methods/tls2trees/slurm/for_instance/published_default_dev_smoke_common.sh
tls2trees_load_contract
tls2trees_validate_contract
tls2trees_load_method_env
tls2trees_resolve_stage0_context
tls2trees_stage_banner semantic

tls2trees_require_file "$TLS2TREES_SEMANTIC_CLI" "semantic runner"
tls2trees_require_file "$PLOT_ROOT/converted/conversion_metadata.json" "conversion metadata"
if [[ -e "$PLOT_ROOT/semantic" || -e "$PLOT_ROOT/metadata/semantic_run.json" ]]; then
  tls2trees_die "refusing existing semantic output under $PLOT_ROOT"
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
python "$TLS2TREES_ENV_VALIDATOR" \
  --tls2trees-repo "$TLS2TREES_REPO" \
  --setup-marker-json "$TLS2TREES_METHOD_ENV_MARKER" \
  --require-cuda
python "$TLS2TREES_SEMANTIC_CLI" \
  --manifest-json "$MANIFEST_JSON" \
  --task-index "$TASK_INDEX" \
  --output-root "$OUTPUT_ROOT" \
  --run-id "$RUN_ID" \
  --tls2trees-repo "$TLS2TREES_REPO" \
  --config "$TLS2TREES_METHOD_CONFIG" \
  --variant published_default \
  --split development

test -f "$PLOT_ROOT/metadata/semantic_run.json"
echo "plot_root=$PLOT_ROOT"
echo "gpu_request=1"
echo "expected_runtime=30 minutes to 4 hours; calibrate from this smoke"
echo "success_criteria=all converted tiles receive non-empty x/y/z/n_z/label semantic outputs and completed metadata"
echo "failure_indicators=CUDA or PyTorch-Geometric import error, OOM, missing tile output, model hash mismatch"
echo "held_out_test_accessed=false"
echo "finished_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
