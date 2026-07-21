#!/bin/bash -l
#SBATCH --job-name=tls2trees_dt_probe
#SBATCH --partition=nodes
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --no-requeue
#SBATCH --output=logs/tls2trees_for_instance/%x_%A_%a.out
#SBATCH --error=logs/tls2trees_for_instance/%x_%A_%a.err

set -euo pipefail

PROJECT_ROOT="${TLS2TREES_PROJECT_ROOT:-$HOME/scratch/tree-seg-benchmark}"
cd "$PROJECT_ROOT"

test "${TLS2TREES_DEV_TUNED_PROBE_CONFIRMED:-0}" = "1"
test "${TLS2TREES_REQUESTED_VARIANT:-}" = "development_tuned"
test "${TLS2TREES_REQUESTED_SPLIT:-}" = "development"
test "$(git rev-parse HEAD)" = "$TLS2TREES_EXPECTED_BENCHMARK_COMMIT"
test -z "$(git status --porcelain)"
test "$(git -C "$TLS2TREES_UPSTREAM_REPO" rev-parse HEAD)" = \
  "ca12cb73b2c736d80b020e8025f8d975d42e6f01"
test -z "$(git -C "$TLS2TREES_UPSTREAM_REPO" status --porcelain)"
test "$(sha256sum "$TLS2TREES_CANDIDATE_MANIFEST" | awk '{print $1}')" = \
  "$TLS2TREES_CANDIDATE_MANIFEST_SHA256"
test "$(sha256sum "$TLS2TREES_METHOD_ENV_MARKER" | awk '{print $1}')" = \
  "$TLS2TREES_METHOD_ENV_MARKER_SHA256"
test -f "$TLS2TREES_SOURCE_PLOT_ROOT/metadata/semantic_run.json"

module purge
module load miniforge3/25.3.0-python3.12.10
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$TLS2TREES_METHOD_ENV"
export PYTHONNOUSERSITE=1
export LD_LIBRARY_PATH="$TLS2TREES_METHOD_ENV/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

python methods/tls2trees/scripts/runtime/validate_tls2trees_environment.py \
  --tls2trees-repo "$TLS2TREES_UPSTREAM_REPO" \
  --setup-marker-json "$TLS2TREES_METHOD_ENV_MARKER" \
  --skip-model-load

echo "stage=development_tuned_compatibility_probe"
echo "run_id=$TLS2TREES_PROBE_RUN_ID"
echo "candidate_index=$SLURM_ARRAY_TASK_ID"
echo "source_plot_root=$TLS2TREES_SOURCE_PLOT_ROOT"
echo "benchmark_commit=$TLS2TREES_EXPECTED_BENCHMARK_COMMIT"
echo "accuracy_metrics_accessed=false"
echo "held_out_test_accessed=false"

python methods/tls2trees/scripts/runtime/run_for_instance_tls2trees_compatibility_probe.py \
  --source-plot-root "$TLS2TREES_SOURCE_PLOT_ROOT" \
  --output-root "$TLS2TREES_OUTPUT_ROOT" \
  --run-id "$TLS2TREES_PROBE_RUN_ID" \
  --candidate-manifest "$TLS2TREES_CANDIDATE_MANIFEST" \
  --candidate-index "$SLURM_ARRAY_TASK_ID" \
  --tls2trees-repo "$TLS2TREES_UPSTREAM_REPO"

echo "finished_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
