#!/bin/bash -l
#SBATCH --job-name=for_sat_dev_eval
#SBATCH --partition=nodes
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --array=0-0%4
#SBATCH --output=logs/segmentanytree_for_instance/%x_%A_%a.out
#SBATCH --error=logs/segmentanytree_for_instance/%x_%A_%a.err

set -euo pipefail

PROJECT_ROOT="$HOME/scratch/tree-seg-benchmark"
PROFILE="${SEGMENTANYTREE_TRAIN_PROFILE:-pilot}"
RUN_ID="${SEGMENTANYTREE_TRAINING_RUN_ID:?Set SEGMENTANYTREE_TRAINING_RUN_ID to the completed training run.}"
EVALUATION_RUN_ID="${SEGMENTANYTREE_EVALUATION_RUN_ID:-$RUN_ID}"
MANIFEST="$PROJECT_ROOT/results/metadata/segmentanytree_for_instance/training_splits/${PROFILE}_split_manifest.json"

cd "$PROJECT_ROOT"
module purge
module load miniforge3/25.3.0-python3.12.10
source "$HOME/fastscratch/venvs/treebench/bin/activate"

mapfile -t RECORD < <(
  python methods/segmentanytree/scripts/data/select_segmentanytree_training_plot.py \
    --manifest "$MANIFEST" \
    --role val \
    --array-index "$SLURM_ARRAY_TASK_ID" \
    --format lines
)
SOURCE_PATH="${RECORD[0]}"

export SEGMENTANYTREE_REQUIRED_SPLIT=dev
export SEGMENTANYTREE_POINTWISE_OUTPUT_ROOT="$PROJECT_ROOT/data/predictions/segmentanytree/for_instance_trained_validation/$RUN_ID"
export SEGMENTANYTREE_POINTWISE_EVALUATION_ROOT="$PROJECT_ROOT/results/metadata/segmentanytree_for_instance/trained_validation/$EVALUATION_RUN_ID"
export SEGMENTANYTREE_POINTWISE_TABLE_ROOT="$PROJECT_ROOT/results/tables/segmentanytree_for_instance/trained_validation/$EVALUATION_RUN_ID"

bash methods/segmentanytree/slurm/evaluation/evaluate_segmentanytree_pointwise_task.sh \
  --plot-path "$SOURCE_PATH"
