#!/bin/bash -l
#SBATCH --job-name=for_sat_final_test
#SBATCH --partition=gpu-l40s-low
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --array=0-10%2
#SBATCH --output=logs/segmentanytree_for_instance/%x_%A_%a.out
#SBATCH --error=logs/segmentanytree_for_instance/%x_%A_%a.err

set -euo pipefail

PROJECT_ROOT="$HOME/scratch/tree-seg-benchmark"
RUN_ID="${SEGMENTANYTREE_TRAINING_RUN_ID:?Set SEGMENTANYTREE_TRAINING_RUN_ID to the accepted full training run.}"
CHECKPOINT_DIR="${SEGMENTANYTREE_CHECKPOINT_DIR:-$HOME/fastscratch/segmentanytree_for_instance_checkpoints/$RUN_ID/run}"

if [[ "${SEGMENTANYTREE_FINAL_TEST_CONFIRMED:-0}" != "1" ]]; then
  echo "Refusing held-out test inference. Freeze the development-selected configuration, then set SEGMENTANYTREE_FINAL_TEST_CONFIRMED=1." >&2
  exit 2
fi

cd "$PROJECT_ROOT"
module purge
module load miniforge3/25.3.0-python3.12.10
module load apptainer/1.3.6
source "$HOME/fastscratch/venvs/treebench/bin/activate"

export SEGMENTANYTREE_REQUIRED_SPLIT=test
export SEGMENTANYTREE_CHECKPOINT_DIR="$CHECKPOINT_DIR"
export SEGMENTANYTREE_RUN_TYPE=trained_held_out_test
export SEGMENTANYTREE_POINTWISE_OUTPUT_ROOT="$PROJECT_ROOT/data/predictions/segmentanytree/for_instance_trained_test/$RUN_ID"
export SEGMENTANYTREE_POINTWISE_RUN_METADATA_ROOT="$PROJECT_ROOT/results/metadata/segmentanytree_for_instance/trained_test_runs/$RUN_ID"

bash scripts/slurm/run_segmentanytree_for_instance_paper_test_task.sh \
  --split test \
  --array-index "$SLURM_ARRAY_TASK_ID"
