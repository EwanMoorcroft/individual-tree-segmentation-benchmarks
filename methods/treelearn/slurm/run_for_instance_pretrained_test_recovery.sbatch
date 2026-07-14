#!/bin/bash -l
#SBATCH --job-name=treelearn_pre_recover
#SBATCH --partition=gpu-l40s-low
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=12
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --output=logs/treelearn_for_instance/treelearn_pre_recover_%j.out
#SBATCH --error=logs/treelearn_for_instance/treelearn_pre_recover_%j.err
set -euo pipefail
cd "$HOME/scratch/tree-seg-benchmark"
test "${TREELEARN_PRETRAINED_TEST_RECOVERY_CONFIRMED:-0}" = 1
test "$(git rev-parse HEAD)" = "$TREELEARN_TEST_BENCHMARK_COMMIT"
test -z "$(git status --porcelain)"
test "$(git -C "$TREELEARN_REPO" rev-parse HEAD)" = fd240ce7caa4c444fe3418aca454dc578bc557d4
test -z "$(git -C "$TREELEARN_REPO" status --porcelain)"
test "$(md5sum "$TREELEARN_TEST_CHECKPOINT" | cut -d ' ' -f 1)" = 106a80de2991c5f23484a3f9d03e3b16
test "$(sha256sum "$TREELEARN_TEST_CHECKPOINT" | cut -d ' ' -f 1)" = "$TREELEARN_TEST_CHECKPOINT_SHA256"

module purge
module load miniforge3/25.3.0-python3.12.10
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$TREELEARN_ENV"
python methods/treelearn/scripts/validate_treelearn_environment.py \
  --treelearn-repo "$TREELEARN_REPO" --require-cuda

echo "run_id=$TREELEARN_TEST_RUN_ID"
echo "task_index=8"
echo "plot=SCION/plot_31_annotated.las"
echo "recovery_policy=map_all_unassigned_to_background_when_initial_grouping_is_empty"
echo "model_or_parameter_selection_performed=false"
set +e
python methods/treelearn/scripts/run_for_instance_finetuned_test_task.py \
  --config methods/treelearn/configs/for_instance_development.yml \
  --manifest "$TREELEARN_TEST_FREEZE" \
  --task-index 8 \
  --run-id "$TREELEARN_TEST_RUN_ID" \
  --dataset-root "$TREELEARN_DATASET_ROOT" \
  --treelearn-repo "$TREELEARN_REPO" \
  --checkpoint "$TREELEARN_TEST_CHECKPOINT" \
  --runtime-root "$TREELEARN_TEST_RUNTIME_BASE" \
  --predictions-root "$TREELEARN_TEST_PREDICTION_BASE" \
  --metadata-root "$TREELEARN_TEST_METADATA_BASE" \
  --tables-root "$TREELEARN_TEST_TABLE_BASE" \
  --allow-empty-group-recovery
INFERENCE_RC=$?
set -e

"$HOME/fastscratch/venvs/treebench/bin/python" \
  methods/treelearn/scripts/evaluate_for_instance_finetuned_test_task.py \
  --manifest "$TREELEARN_TEST_FREEZE" \
  --task-index 8 \
  --run-id "$TREELEARN_TEST_RUN_ID" \
  --predictions-root "$TREELEARN_TEST_PREDICTION_BASE" \
  --metadata-root "$TREELEARN_TEST_METADATA_BASE" \
  --tables-root "$TREELEARN_TEST_TABLE_BASE"
echo "inference_return_code=$INFERENCE_RC"
echo "status=treelearn-pretrained-held-out-test-recovery-task-accounted"
