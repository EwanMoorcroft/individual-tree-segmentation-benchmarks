#!/bin/bash -l
#SBATCH --job-name=treelearn_dev
#SBATCH --partition=gpu-l40s-low
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=12
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --output=logs/treelearn_for_instance/treelearn_dev_%A_%a.out
#SBATCH --error=logs/treelearn_for_instance/treelearn_dev_%A_%a.err

set -euo pipefail

cd "$HOME/scratch/tree-seg-benchmark"
mkdir -p logs/treelearn_for_instance

if [[ "${TREELEARN_FULL_DEV_CONFIRMED:-0}" != "1" \
      || "${TREELEARN_DEV_ALIGNMENT_REVIEW_CONFIRMED:-0}" != "1" ]]; then
  echo "Refusing TreeLearn full development task without both confirmations."
  exit 2
fi

module purge
module load miniforge3/25.3.0-python3.12.10
source "$(conda info --base)/etc/profile.d/conda.sh"

TREELEARN_ENV="${TREELEARN_ENV:-$HOME/fastscratch/venvs/treelearn}"
TREELEARN_REPO="${TREELEARN_REPO:-$HOME/fastscratch/external/TreeLearn}"
TREELEARN_CHECKPOINT="${TREELEARN_CHECKPOINT:-$HOME/fastscratch/treelearn_checkpoints/model_weights_20241213.pth}"
TREELEARN_DATASET_ROOT="${TREELEARN_DATASET_ROOT:-$HOME/data/datasets/for_instance/FORinstance_dataset}"
TREELEARN_DEV_RUN_ID="${TREELEARN_DEV_RUN_ID:?TREELEARN_DEV_RUN_ID is required}"
TREELEARN_DEV_MANIFEST="${TREELEARN_DEV_MANIFEST:?TREELEARN_DEV_MANIFEST is required}"
TREELEARN_EXPECTED_BENCHMARK_COMMIT="${TREELEARN_EXPECTED_BENCHMARK_COMMIT:?TREELEARN_EXPECTED_BENCHMARK_COMMIT is required}"
TREELEARN_EXPECTED_CHECKPOINT_MD5="${TREELEARN_EXPECTED_CHECKPOINT_MD5:?TREELEARN_EXPECTED_CHECKPOINT_MD5 is required}"
TREELEARN_EXPECTED_CHECKPOINT_SHA256="${TREELEARN_EXPECTED_CHECKPOINT_SHA256:?TREELEARN_EXPECTED_CHECKPOINT_SHA256 is required}"
TASK_INDEX="${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}"

test -x "$TREELEARN_ENV/bin/python"
test -f "$TREELEARN_ENV/.treelearn_setup_complete"
test -x "$HOME/fastscratch/venvs/treebench/bin/python"
test -d "$TREELEARN_REPO/.git"
test -f "$TREELEARN_CHECKPOINT"
test -f "$TREELEARN_DEV_MANIFEST"
test "$(git rev-parse HEAD)" = "$TREELEARN_EXPECTED_BENCHMARK_COMMIT"
test -z "$(git status --porcelain)"
test "$(git -C "$TREELEARN_REPO" rev-parse HEAD)" = \
  "fd240ce7caa4c444fe3418aca454dc578bc557d4"
test -z "$(git -C "$TREELEARN_REPO" status --porcelain)"

ACTUAL_MD5=$(md5sum "$TREELEARN_CHECKPOINT" | cut -d ' ' -f 1)
ACTUAL_SHA256=$(sha256sum "$TREELEARN_CHECKPOINT" | cut -d ' ' -f 1)
test "$ACTUAL_MD5" = "$TREELEARN_EXPECTED_CHECKPOINT_MD5"
test "$ACTUAL_SHA256" = "$TREELEARN_EXPECTED_CHECKPOINT_SHA256"

conda activate "$TREELEARN_ENV"
python methods/treelearn/scripts/validate_treelearn_environment.py \
  --treelearn-repo "$TREELEARN_REPO" \
  --require-cuda

echo "run_id=$TREELEARN_DEV_RUN_ID"
echo "task_index=$TASK_INDEX"
echo "job_id=${SLURM_JOB_ID:-manual}"
echo "array_job_id=${SLURM_ARRAY_JOB_ID:-manual}"
echo "split=dev"
echo "held_out_test_accessed=false"

set +e
python methods/treelearn/scripts/run_for_instance_development_task.py \
  --config methods/treelearn/configs/for_instance_development.yml \
  --manifest "$TREELEARN_DEV_MANIFEST" \
  --task-index "$TASK_INDEX" \
  --run-id "$TREELEARN_DEV_RUN_ID" \
  --dataset-root "$TREELEARN_DATASET_ROOT" \
  --treelearn-repo "$TREELEARN_REPO" \
  --checkpoint "$TREELEARN_CHECKPOINT"
INFERENCE_RC=$?
set -e

"$HOME/fastscratch/venvs/treebench/bin/python" \
  methods/treelearn/scripts/evaluate_for_instance_development_task.py \
  --config methods/treelearn/configs/for_instance_development.yml \
  --manifest "$TREELEARN_DEV_MANIFEST" \
  --task-index "$TASK_INDEX" \
  --run-id "$TREELEARN_DEV_RUN_ID"

echo "inference_return_code=$INFERENCE_RC"
echo "status=treelearn-development-task-accounted"
