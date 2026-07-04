#!/bin/bash -l
#SBATCH --job-name=for_sat_prepare
#SBATCH --partition=nodes
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=logs/segmentanytree_for_instance/%x_%j.out
#SBATCH --error=logs/segmentanytree_for_instance/%x_%j.err

set -euo pipefail

PROJECT_ROOT="$HOME/scratch/tree-seg-benchmark"
DATASET_ROOT="$HOME/data/datasets/for_instance/FORinstance_dataset"
PROFILE="${SEGMENTANYTREE_TRAIN_PROFILE:-pilot}"
TRAINING_ROOT="$HOME/fastscratch/segmentanytree_for_instance_training/$PROFILE"
MANIFEST="$PROJECT_ROOT/results/metadata/segmentanytree_for_instance/training_splits/${PROFILE}_split_manifest.json"

if [[ "$PROFILE" != "pilot" && "$PROFILE" != "full" ]]; then
  echo "SEGMENTANYTREE_TRAIN_PROFILE must be pilot or full." >&2
  exit 2
fi

cd "$PROJECT_ROOT"
mkdir -p logs/segmentanytree_for_instance "$(dirname "$MANIFEST")"

module purge
module load miniforge3/25.3.0-python3.12.10
source "$HOME/fastscratch/venvs/treebench/bin/activate"

ARGS=(
  --dataset-root "$DATASET_ROOT"
  --split-metadata "$DATASET_ROOT/data_split_metadata.csv"
  --output-root "$TRAINING_ROOT"
  --manifest "$MANIFEST"
  --profile "$PROFILE"
  --seed 42
  --validation-fraction 0.25
  --pilot-train-count 2
  --pilot-val-count 1
)
if [[ "${SEGMENTANYTREE_OVERWRITE:-0}" == "1" ]]; then
  ARGS+=(--overwrite)
fi

python methods/segmentanytree/scripts/data/prepare_segmentanytree_for_instance_training.py "${ARGS[@]}"

echo "Training data root: $TRAINING_ROOT"
echo "Split manifest: $MANIFEST"
