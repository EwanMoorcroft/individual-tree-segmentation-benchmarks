#!/bin/bash -l
#SBATCH --job-name=for_sat_audit
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
  python scripts/data/select_for_instance_plot.py \
    --config configs/for_instance_segmentanytree_benchmark.yml \
    "${SELECTION_ARGS[@]}" \
    --format lines
)

REFERENCE_PATH="${PLOT_INFO[0]}"
COLLECTION="${PLOT_INFO[2]}"
PLOT_NAME="${PLOT_INFO[3]}"
PREDICTION_PATH="data/predictions/segmentanytree/for_instance/${COLLECTION}/${PLOT_NAME}/final_results/${PLOT_NAME}_out.laz"
AUDIT_PATH="results/metadata/segmentanytree_for_instance/export_validation/${COLLECTION}/${PLOT_NAME}.json"

python scripts/evaluation/validate_segmentanytree_export.py \
  --source "$REFERENCE_PATH" \
  --output "$PREDICTION_PATH" \
  --coordinate-tolerance 0.001 \
  --reference-instance-field treeID \
  --prediction-instance-field PredInstance \
  --output-json "$AUDIT_PATH"
