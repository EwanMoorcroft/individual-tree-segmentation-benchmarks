#!/bin/bash -l
#SBATCH --job-name=for_sat_pointwise
#SBATCH --partition=nodes
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --array=0-31%4
#SBATCH --output=logs/segmentanytree_for_instance/%x_%A_%a.out
#SBATCH --error=logs/segmentanytree_for_instance/%x_%A_%a.err

set -euo pipefail

module purge
module load miniforge3/25.3.0-python3.12.10
source "$HOME/fastscratch/venvs/treebench/bin/activate"

cd "$HOME/scratch/tree-seg-benchmark"
mkdir -p logs/segmentanytree_for_instance

SELECTION_ARGS=(--array-index "$SLURM_ARRAY_TASK_ID")
if [[ -n "${FOR_INSTANCE_SPLIT:-}" ]]; then
  SELECTION_ARGS+=(--split "$FOR_INSTANCE_SPLIT")
fi

mapfile -t PLOT_INFO < <(
  python methods/segmentanytree/scripts/data/select_for_instance_plot.py \
    --config methods/segmentanytree/configs/for_instance_benchmark.yml \
    "${SELECTION_ARGS[@]}" \
    --format lines
)

RELATIVE_PATH="${PLOT_INFO[1]}"
COLLECTION="${PLOT_INFO[2]}"
PLOT_NAME="${PLOT_INFO[3]}"
SPLIT="${PLOT_INFO[4]}"
PREDICTION_PATH="data/predictions/segmentanytree/for_instance/${COLLECTION}/${PLOT_NAME}/final_results/${PLOT_NAME}_out.laz"
AUDIT_PATH="results/metadata/segmentanytree_for_instance/export_validation/${COLLECTION}/${PLOT_NAME}.json"
OUTPUT_ROOT="results/metadata/segmentanytree_for_instance/pointwise/${COLLECTION}"
TABLE_ROOT="results/tables/segmentanytree_for_instance/pointwise"

python - "$AUDIT_PATH" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
if not path.is_file():
    raise SystemExit(f"Export audit is missing: {path}")
payload = json.loads(path.read_text(encoding="utf-8"))
if not payload.get("safe_for_final_accuracy_evaluation"):
    raise SystemExit(
        "Final LAZ failed row-preservation checks. "
        "Use aligned internal evaluation PLY files instead."
    )
PY

mkdir -p "$OUTPUT_ROOT" "$TABLE_ROOT"

python methods/segmentanytree/scripts/evaluation/pointwise_instance_metrics.py \
  --combined-labelled-point-cloud "$PREDICTION_PATH" \
  --predicted-instance-field PredInstance \
  --reference-instance-field treeID \
  --predicted-semantic-field PredSemantic \
  --reference-semantic-field classification \
  --semantic-offset 0 \
  --reference-tree-classes 4,5,6 \
  --prediction-tree-classes 1 \
  --ignored-reference-labels=-1,0 \
  --ignored-prediction-labels=-1,0 \
  --iou-threshold 0.5 \
  --plot-name "$PLOT_NAME" \
  --collection "$COLLECTION" \
  --split "$SPLIT" \
  --relative-path "$RELATIVE_PATH" \
  --output-json "$OUTPUT_ROOT/${PLOT_NAME}.json" \
  --paper-matches-csv "$TABLE_ROOT/${COLLECTION}_${PLOT_NAME}_paper_matches.csv" \
  --harmonized-matches-csv "$TABLE_ROOT/${COLLECTION}_${PLOT_NAME}_one_to_one_matches.csv"
