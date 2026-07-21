#!/bin/bash -l
#SBATCH --job-name=tls2trees_retained_eval
#SBATCH --partition=nodes
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --no-requeue
#SBATCH --output=logs/tls2trees_for_instance/%x_%A_%a.out
#SBATCH --error=logs/tls2trees_for_instance/%x_%A_%a.err

set -euo pipefail

cd "${TLS2TREES_PROJECT_ROOT:-$HOME/scratch/tree-seg-benchmark}"
test "${TLS2TREES_FINAL_EVALUATION_CONFIRMED:-0}" = "1"
test "${TLS2TREES_FINAL_EVALUATION_INFERENCE_ALLOWED:-0}" = "0"
test "$(git rev-parse HEAD)" = "$TLS2TREES_FINAL_EVALUATION_BENCHMARK_COMMIT"
test -z "$(git status --porcelain)"
test "$(sha256sum "$TLS2TREES_FINAL_EVALUATION_PLAN_JSON" | awk '{print $1}')" = \
  "$TLS2TREES_FINAL_EVALUATION_PLAN_SHA256"
test "$(sha256sum "$TLS2TREES_FINAL_EVALUATION_CLI" | awk '{print $1}')" = \
  "$TLS2TREES_FINAL_EVALUATION_CLI_SHA256"
test "$(sha256sum "$TLS2TREES_FINAL_EVALUATION_EVALUATOR" | awk '{print $1}')" = \
  "$TLS2TREES_FINAL_EVALUATION_EVALUATOR_SHA256"

module purge
module load miniforge3/25.3.0-python3.12.10
# shellcheck disable=SC1090
source "$TLS2TREES_FINAL_EVALUATION_TREEBENCH_ENV/bin/activate"
export PYTHONNOUSERSITE=1

python "$TLS2TREES_FINAL_EVALUATION_CLI" run-task \
  --plan-json "$TLS2TREES_FINAL_EVALUATION_PLAN_JSON" \
  --plan-sha256 "$TLS2TREES_FINAL_EVALUATION_PLAN_SHA256" \
  --task-index "$SLURM_ARRAY_TASK_ID"

echo "status=retained_prediction_evaluation_task_completed"
echo "task_index=$SLURM_ARRAY_TASK_ID"
echo "inference_rerun=false"
echo "prediction_adapter_rerun=false"
