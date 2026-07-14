#!/bin/bash -l
#SBATCH --job-name=treelearn_long_val
#SBATCH --partition=gpu-l40s
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=12
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --array=0-40%8
#SBATCH --output=logs/treelearn_for_instance/treelearn_long_val_%A_%a.out
#SBATCH --error=logs/treelearn_for_instance/treelearn_long_val_%A_%a.err
set -euo pipefail
TREELEARN_PROJECT_ROOT="${TREELEARN_PROJECT_ROOT:-$HOME/scratch/tree-seg-benchmark}"
cd "$TREELEARN_PROJECT_ROOT"
test "${TREELEARN_FINETUNE_LONG_CONFIRMED:-0}" = 1
test "$(git rev-parse HEAD)" = "$TREELEARN_EXPECTED_BENCHMARK_COMMIT"
test -z "$(git status --porcelain)"
test -f "$TREELEARN_LONG_FREEZE"
test -f "$TREELEARN_LONG_DEV_MANIFEST"

module purge
module load miniforge3/25.3.0-python3.12.10
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$TREELEARN_ENV"
python methods/treelearn/scripts/verify_for_instance_finetune_long_contract.py \
  --freeze "$TREELEARN_LONG_FREEZE" \
  --treelearn-repo "$TREELEARN_REPO"

mapfile -t TASK < <(
  python methods/treelearn/scripts/resolve_for_instance_finetune_long_validation_task.py \
    --freeze "$TREELEARN_LONG_FREEZE" \
    --array-task-id "$SLURM_ARRAY_TASK_ID" \
    --completion-root "$TREELEARN_LONG_ROOT/trial_completions"
)
test "${#TASK[@]}" -eq 14
TRIAL_INDEX=${TASK[0]}
CONFIG_ID=${TASK[1]}
SEED=${TASK[2]}
EPOCH=${TASK[3]}
MANIFEST_TASK_INDICES=${TASK[4]}
CHECKPOINT=${TASK[5]}
CHECKPOINT_SIZE_BYTES=${TASK[6]}
CHECKPOINT_SHA256=${TASK[7]}
COMPLETION_RECORD=${TASK[8]}
COMPLETION_RECORD_SHA256=${TASK[9]}
VALIDATION_RUN_ID=${TASK[10]}
EVALUATION_CONFIG=${TASK[11]}
EVALUATION_CONFIG_SHA256=${TASK[12]}
TRAINING_MODE=${TASK[13]}
test -f "$CHECKPOINT"
test "$(stat -c %s "$CHECKPOINT")" = "$CHECKPOINT_SIZE_BYTES"
test "$(sha256sum "$CHECKPOINT" | cut -d ' ' -f 1)" = "$CHECKPOINT_SHA256"
if test "$TRIAL_INDEX" -ge 0; then
  test -f "$COMPLETION_RECORD"
  test "$(sha256sum "$COMPLETION_RECORD" | cut -d ' ' -f 1)" = \
    "$COMPLETION_RECORD_SHA256"
fi
test -f "$EVALUATION_CONFIG"
test "$(sha256sum "$EVALUATION_CONFIG" | cut -d ' ' -f 1)" = \
  "$EVALUATION_CONFIG_SHA256"

IFS=',' read -r -a PLOT_TASKS <<< "$MANIFEST_TASK_INDICES"
test "${#PLOT_TASKS[@]}" -eq 5
for MANIFEST_TASK_INDEX in "${PLOT_TASKS[@]}"; do
  python methods/treelearn/scripts/run_for_instance_development_task.py \
    --config "$EVALUATION_CONFIG" \
    --manifest "$TREELEARN_LONG_DEV_MANIFEST" \
    --task-index "$MANIFEST_TASK_INDEX" \
    --run-id "$VALIDATION_RUN_ID" \
    --dataset-root "$TREELEARN_DATASET_ROOT" \
    --treelearn-repo "$TREELEARN_REPO" \
    --checkpoint "$CHECKPOINT" \
    --training-mode "$TRAINING_MODE"

  "$HOME/fastscratch/venvs/treebench/bin/python" \
    methods/treelearn/scripts/evaluate_for_instance_development_task.py \
    --config "$EVALUATION_CONFIG" \
    --manifest "$TREELEARN_LONG_DEV_MANIFEST" \
    --task-index "$MANIFEST_TASK_INDEX" \
    --run-id "$VALIDATION_RUN_ID"
done

echo "status=long-validation-checkpoint-completed trial=$TRIAL_INDEX config=$CONFIG_ID seed=$SEED epoch=$EPOCH plots=$MANIFEST_TASK_INDICES"
echo held_out_test_accessed=false
