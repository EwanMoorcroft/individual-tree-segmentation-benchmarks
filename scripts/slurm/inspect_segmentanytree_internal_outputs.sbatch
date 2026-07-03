#!/bin/bash -l
#SBATCH --job-name=for_sat_inspect
#SBATCH --partition=nodes
#SBATCH --time=00:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
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

COLLECTION="${PLOT_INFO[2]}"
PLOT_NAME="${PLOT_INFO[3]}"
OUTPUT_ROOT="data/predictions/segmentanytree/for_instance/${COLLECTION}/${PLOT_NAME}"
OUTPUT_JSON="results/metadata/segmentanytree_for_instance/internal_output_inventory/${COLLECTION}/${PLOT_NAME}.json"

python scripts/evaluation/inspect_segmentanytree_outputs.py \
  --output-root "$OUTPUT_ROOT" \
  --output-json "$OUTPUT_JSON"
