#!/bin/bash -l
#SBATCH --job-name=for_sat_evaluate
#SBATCH --partition=nodes
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --array=0-31%2
#SBATCH --output=logs/segmentanytree_for_instance/%x_%A_%a.out
#SBATCH --error=logs/segmentanytree_for_instance/%x_%A_%a.err

set -euo pipefail

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
REFERENCE_PATH="${PLOT_INFO[0]}"
RELATIVE_PATH="${PLOT_INFO[1]}"
COLLECTION="${PLOT_INFO[2]}"
PLOT_NAME="${PLOT_INFO[3]}"
SPLIT="${PLOT_INFO[4]}"
PREDICTION_DIR="data/interim/segmentanytree/for_instance/${COLLECTION}/${PLOT_NAME}/normalised_predictions"
EVALUATION_DIR="results/metadata/segmentanytree_for_instance/evaluations/${COLLECTION}"
RUN_METADATA="results/metadata/segmentanytree_for_instance/runs/${COLLECTION}/${PLOT_NAME}_run.json"

mkdir -p "$EVALUATION_DIR"
python scripts/evaluation/instance_iou_f1.py \
  --plot-name "$PLOT_NAME" \
  --collection "$COLLECTION" \
  --split "$SPLIT" \
  --relative-path "$RELATIVE_PATH" \
  --predicted-instance-dir "$PREDICTION_DIR" \
  --reference-labelled-point-cloud "$REFERENCE_PATH" \
  --reference-label-field treeID \
  --reference-semantic-field classification \
  --reference-classes 4 5 6 \
  --ignored-reference-classes 0 1 2 3 \
  --ignore-reference-labels 0 \
  --iou-threshold 0.5 \
  --coordinate-tolerance 0.02 \
  --run-metadata-json "$RUN_METADATA" \
  --output-json "${EVALUATION_DIR}/${PLOT_NAME}.json" \
  --output-metrics-csv "${EVALUATION_DIR}/${PLOT_NAME}_metrics.csv" \
  --output-matches-csv "${EVALUATION_DIR}/${PLOT_NAME}_matches.csv" \
  --output-unmatched-predictions-csv "${EVALUATION_DIR}/${PLOT_NAME}_unmatched_predictions.csv" \
  --output-unmatched-references-csv "${EVALUATION_DIR}/${PLOT_NAME}_unmatched_references.csv"
