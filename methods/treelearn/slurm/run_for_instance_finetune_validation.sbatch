#!/bin/bash -l
#SBATCH --job-name=treelearn_ft_val
#SBATCH --partition=gpu-l40s-low
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=12
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --output=logs/treelearn_for_instance/treelearn_ft_val_%A_%a.out
#SBATCH --error=logs/treelearn_for_instance/treelearn_ft_val_%A_%a.err
set -euo pipefail
cd "$HOME/scratch/tree-seg-benchmark"
test "${TREELEARN_FINETUNE_VALIDATION_CONFIRMED:-0}" = 1
test "$(git rev-parse HEAD)" = "$TREELEARN_EXPECTED_BENCHMARK_COMMIT"
test -z "$(git status --porcelain)"
test "$(sha256sum "$TREELEARN_FINETUNE_CHECKPOINT" | cut -d ' ' -f 1)" = "$TREELEARN_FINETUNE_CHECKPOINT_SHA256"
module purge
module load miniforge3/25.3.0-python3.12.10
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$TREELEARN_ENV"
python methods/treelearn/scripts/run_for_instance_development_task.py \
  --config methods/treelearn/configs/for_instance_development.yml \
  --manifest "$TREELEARN_DEV_MANIFEST_JSON" \
  --task-index "$SLURM_ARRAY_TASK_ID" \
  --run-id "$TREELEARN_FINETUNE_RUN_ID" \
  --dataset-root "$TREELEARN_DATASET_ROOT" \
  --treelearn-repo "$TREELEARN_REPO" \
  --checkpoint "$TREELEARN_FINETUNE_CHECKPOINT" \
  --training-mode fine_tuned_on_dev
"$HOME/fastscratch/venvs/treebench/bin/python" \
  methods/treelearn/scripts/evaluate_for_instance_development_task.py \
  --config methods/treelearn/configs/for_instance_development.yml \
  --manifest "$TREELEARN_DEV_MANIFEST_JSON" \
  --task-index "$SLURM_ARRAY_TASK_ID" \
  --run-id "$TREELEARN_FINETUNE_RUN_ID"
echo status=fine-tuned-development-validation-task-completed
echo held_out_test_accessed=false
