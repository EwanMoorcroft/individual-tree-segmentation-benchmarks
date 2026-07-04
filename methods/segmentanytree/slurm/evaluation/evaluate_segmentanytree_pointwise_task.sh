#!/bin/bash

set -euo pipefail

PROJECT_ROOT="$HOME/scratch/tree-seg-benchmark"
CONFIG="$PROJECT_ROOT/methods/segmentanytree/configs/for_instance_benchmark.yml"
REQUIRED_SPLIT="${SEGMENTANYTREE_REQUIRED_SPLIT:-test}"
OUTPUT_BASE="${SEGMENTANYTREE_POINTWISE_OUTPUT_ROOT:-$PROJECT_ROOT/data/predictions/segmentanytree/for_instance_pointwise}"
RESULT_BASE="${SEGMENTANYTREE_POINTWISE_EVALUATION_ROOT:-$PROJECT_ROOT/results/metadata/segmentanytree_for_instance/pointwise_paper}"
TABLE_BASE="${SEGMENTANYTREE_POINTWISE_TABLE_ROOT:-$PROJECT_ROOT/results/tables/segmentanytree_for_instance/pointwise_paper}"

cd "$PROJECT_ROOT"
mkdir -p logs/segmentanytree_for_instance

mapfile -t PLOT_INFO < <(
  python methods/segmentanytree/scripts/data/select_for_instance_plot.py \
    --config "$CONFIG" \
    "$@" \
    --format lines
)

RELATIVE_PATH="${PLOT_INFO[1]}"
COLLECTION="${PLOT_INFO[2]}"
PLOT_NAME="${PLOT_INFO[3]}"
SPLIT="${PLOT_INFO[4]}"
if [[ "$SPLIT" != "$REQUIRED_SPLIT" ]]; then
  echo "Evaluation requires split $REQUIRED_SPLIT, not $SPLIT: $RELATIVE_PATH" >&2
  exit 2
fi

OUTPUT_ROOT="$OUTPUT_BASE/${COLLECTION}/${PLOT_NAME}"
INSTANCE_EVALUATION="$OUTPUT_ROOT/Instance_results_forEval_0.ply"
RESULT_ROOT="$RESULT_BASE/${COLLECTION}"

mapfile -t SEMANTIC_FILES < <(
  find "$OUTPUT_ROOT" -maxdepth 1 -type f \
    -name 'semantic_segmentation_*.ply' -print | sort
)
if [[ "${#SEMANTIC_FILES[@]}" -ne 1 ]]; then
  echo "Expected one semantic evaluation PLY; found ${#SEMANTIC_FILES[@]}" >&2
  exit 2
fi
SEMANTIC_EVALUATION="${SEMANTIC_FILES[0]}"

mkdir -p "$RESULT_ROOT" "$TABLE_BASE"

python methods/segmentanytree/scripts/evaluation/pointwise_instance_metrics.py \
  --instance-evaluation-ply "$INSTANCE_EVALUATION" \
  --semantic-evaluation-ply "$SEMANTIC_EVALUATION" \
  --predicted-instance-field preds \
  --reference-instance-field gt \
  --predicted-semantic-field preds \
  --reference-semantic-field gt \
  --semantic-offset 1 \
  --reference-tree-classes 2 \
  --prediction-tree-classes 2 \
  --ignored-reference-labels=-1 \
  --ignored-prediction-labels=-1 \
  --iou-threshold 0.5 \
  --plot-name "$PLOT_NAME" \
  --collection "$COLLECTION" \
  --split "$SPLIT" \
  --relative-path "$RELATIVE_PATH" \
  --output-json "$RESULT_ROOT/${PLOT_NAME}.json" \
  --paper-matches-csv "$TABLE_BASE/${COLLECTION}_${PLOT_NAME}_paper_matches.csv" \
  --harmonized-matches-csv "$TABLE_BASE/${COLLECTION}_${PLOT_NAME}_one_to_one_matches.csv"
