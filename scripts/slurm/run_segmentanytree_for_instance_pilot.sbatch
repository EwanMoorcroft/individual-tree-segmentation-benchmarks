#!/bin/bash -l
#SBATCH --job-name=for_sat_pilot
#SBATCH --partition=nodes
#SBATCH --time=00:15:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --output=logs/segmentanytree_for_instance/%x_%j.out
#SBATCH --error=logs/segmentanytree_for_instance/%x_%j.err

set -euo pipefail

module purge
module load miniforge3/25.3.0-python3.12.10
source "$HOME/fastscratch/venvs/treebench/bin/activate"

cd "$HOME/scratch/tree-seg-benchmark"
mkdir -p logs/segmentanytree_for_instance

RUN_ARGS=(--dry-run)
if [[ "${SEGMENTANYTREE_EXECUTE:-0}" == "1" ]]; then
  RUN_ARGS=()
fi

python scripts/methods/run_segmentanytree_for_instance.py \
  --config configs/for_instance_segmentanytree_benchmark.yml \
  --plot-path CULS/plot_1_annotated.las \
  "${RUN_ARGS[@]}"
