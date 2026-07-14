#!/bin/bash -l
#SBATCH --job-name=treelearn_pre_test
#SBATCH --partition=gpu-l40s-low
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=12
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --output=logs/treelearn_for_instance/treelearn_pre_test_%A_%a.out
#SBATCH --error=logs/treelearn_for_instance/treelearn_pre_test_%A_%a.err
set -euo pipefail
cd "$HOME/scratch/tree-seg-benchmark"
test "${TREELEARN_PRETRAINED_TEST_CONFIRMED:-0}" = 1
test "$(git rev-parse HEAD)" = "$TREELEARN_TEST_BENCHMARK_COMMIT"
test -z "$(git status --porcelain)"
test "$(git -C "$TREELEARN_REPO" rev-parse HEAD)" = fd240ce7caa4c444fe3418aca454dc578bc557d4
test -z "$(git -C "$TREELEARN_REPO" status --porcelain)"
test "$(md5sum "$TREELEARN_TEST_CHECKPOINT" | cut -d ' ' -f 1)" = 106a80de2991c5f23484a3f9d03e3b16
test "$(sha256sum "$TREELEARN_TEST_CHECKPOINT" | cut -d ' ' -f 1)" = "$TREELEARN_TEST_CHECKPOINT_SHA256"
TASK_INDEX="${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}"

module purge
module load miniforge3/25.3.0-python3.12.10
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$TREELEARN_ENV"
python methods/treelearn/scripts/validate_treelearn_environment.py \
  --treelearn-repo "$TREELEARN_REPO" --require-cuda

echo "run_id=$TREELEARN_TEST_RUN_ID"
echo "task_index=$TASK_INDEX"
echo "variant=published_pretrained"
echo "split=test"
echo "held_out_test_accessed=true"
set +e
python methods/treelearn/scripts/run_for_instance_finetuned_test_task.py \
  --config methods/treelearn/configs/for_instance_development.yml \
  --manifest "$TREELEARN_TEST_FREEZE" \
  --task-index "$TASK_INDEX" \
  --run-id "$TREELEARN_TEST_RUN_ID" \
  --dataset-root "$TREELEARN_DATASET_ROOT" \
  --treelearn-repo "$TREELEARN_REPO" \
  --checkpoint "$TREELEARN_TEST_CHECKPOINT" \
  --runtime-root "$TREELEARN_TEST_RUNTIME_BASE" \
  --predictions-root "$TREELEARN_TEST_PREDICTION_BASE" \
  --metadata-root "$TREELEARN_TEST_METADATA_BASE" \
  --tables-root "$TREELEARN_TEST_TABLE_BASE"
INFERENCE_RC=$?
set -e

"$HOME/fastscratch/venvs/treebench/bin/python" \
  methods/treelearn/scripts/evaluate_for_instance_finetuned_test_task.py \
  --manifest "$TREELEARN_TEST_FREEZE" \
  --task-index "$TASK_INDEX" \
  --run-id "$TREELEARN_TEST_RUN_ID" \
  --predictions-root "$TREELEARN_TEST_PREDICTION_BASE" \
  --metadata-root "$TREELEARN_TEST_METADATA_BASE" \
  --tables-root "$TREELEARN_TEST_TABLE_BASE"
echo "inference_return_code=$INFERENCE_RC"
echo "status=treelearn-pretrained-held-out-test-task-accounted"
