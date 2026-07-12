#!/bin/bash -l
#SBATCH --job-name=treelearn_long_train
#SBATCH --partition=gpu-l40s
#SBATCH --time=36:00:00
#SBATCH --cpus-per-task=12
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --output=logs/treelearn_for_instance/treelearn_long_train_%A_%a.out
#SBATCH --error=logs/treelearn_for_instance/treelearn_long_train_%A_%a.err
set -euo pipefail
TREELEARN_PROJECT_ROOT="${TREELEARN_PROJECT_ROOT:-$HOME/scratch/tree-seg-benchmark}"
cd "$TREELEARN_PROJECT_ROOT"
test "${TREELEARN_FINETUNE_LONG_CONFIRMED:-0}" = 1
test "$(git rev-parse HEAD)" = "$TREELEARN_EXPECTED_BENCHMARK_COMMIT"
test -z "$(git status --porcelain)"
"$HOME/fastscratch/venvs/treebench/bin/python" \
  methods/treelearn/scripts/verify_for_instance_finetune_long_contract.py \
  --freeze "$TREELEARN_LONG_FREEZE" \
  --treelearn-repo "$TREELEARN_REPO"
mapfile -t TRIAL < <(
  "$HOME/fastscratch/venvs/treebench/bin/python" \
    methods/treelearn/scripts/resolve_for_instance_finetune_long_trial.py \
    --freeze "$TREELEARN_LONG_FREEZE" \
    --trial-index "$SLURM_ARRAY_TASK_ID"
)
test "${#TRIAL[@]}" -eq 3
CONFIG=${TRIAL[0]}
SEED=${TRIAL[1]}
WORK_DIR=${TRIAL[2]}
export PYTHONHASHSEED="$SEED"
module purge
module load miniforge3/25.3.0-python3.12.10
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$TREELEARN_ENV"
cd "$TREELEARN_LONG_ROOT"
python "$TREELEARN_PROJECT_ROOT/methods/treelearn/scripts/run_treelearn_training_seeded.py" \
  --treelearn-repo "$TREELEARN_REPO" \
  --config "$CONFIG" \
  --seed "$SEED" \
  --work-dir "$WORK_DIR" \
  --environment-record "$TREELEARN_LONG_ROOT/trial_completions/trial_${SLURM_ARRAY_TASK_ID}_environment.json"
cd "$TREELEARN_PROJECT_ROOT"
python methods/treelearn/scripts/resolve_for_instance_finetune_long_trial.py \
  --freeze "$TREELEARN_LONG_FREEZE" \
  --trial-index "$SLURM_ARRAY_TASK_ID" \
  --verify-output "$TREELEARN_LONG_ROOT/trial_completions/trial_${SLURM_ARRAY_TASK_ID}.json"
echo status=treelearn-long-trial-completed
echo held_out_test_accessed=false
