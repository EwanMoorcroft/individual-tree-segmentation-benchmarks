#!/bin/bash -l
#SBATCH --job-name=for_sat_normalise
#SBATCH --partition=nodes
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --array=0-31%4
#SBATCH --output=logs/segmentanytree_for_instance/%x_%A_%a.out
#SBATCH --error=logs/segmentanytree_for_instance/%x_%A_%a.err

set -euo pipefail

if [[ -z "${SEGMENTANYTREE_OUTPUT_FORMAT:-}" ]]; then
  echo "Set SEGMENTANYTREE_OUTPUT_FORMAT after inspecting the pilot output." >&2
  exit 2
fi

module purge
module load miniforge3/25.3.0-python3.12.10
source "$HOME/fastscratch/venvs/treebench/bin/activate"

cd "$HOME/scratch/tree-seg-benchmark"
mkdir -p logs/segmentanytree_for_instance

mapfile -t PLOT_INFO < <(
  python scripts/data/select_for_instance_plot.py \
    --config configs/for_instance_segmentanytree_benchmark.yml \
    --array-index "$SLURM_ARRAY_TASK_ID" \
    --format lines
)
COLLECTION="${PLOT_INFO[2]}"
PLOT_NAME="${PLOT_INFO[3]}"
PREDICTION_INPUT="data/predictions/segmentanytree/for_instance/${COLLECTION}/${PLOT_NAME}/final_results"
NORMALISED_OUTPUT="data/interim/segmentanytree/for_instance/${COLLECTION}/${PLOT_NAME}/normalised_predictions"
METADATA_OUTPUT="results/metadata/segmentanytree_for_instance/normalisation/${COLLECTION}/${PLOT_NAME}.json"

NORMALISE_ARGS=(
  --input "$PREDICTION_INPUT"
  --output-dir "$NORMALISED_OUTPUT"
  --format "$SEGMENTANYTREE_OUTPUT_FORMAT"
  --metadata-output "$METADATA_OUTPUT"
)
if [[ -n "${SEGMENTANYTREE_INSTANCE_FIELD:-}" ]]; then
  NORMALISE_ARGS+=(--instance-field "$SEGMENTANYTREE_INSTANCE_FIELD")
fi

python scripts/methods/normalise_segmentanytree_predictions.py "${NORMALISE_ARGS[@]}"
