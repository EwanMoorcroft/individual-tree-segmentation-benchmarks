#!/bin/bash -l
#SBATCH --job-name=for_sat_predict
#SBATCH --partition=nodes
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --array=0-31%2
#SBATCH --output=logs/segmentanytree_for_instance/%x_%A_%a.out
#SBATCH --error=logs/segmentanytree_for_instance/%x_%A_%a.err

set -euo pipefail

if [[ "${SEGMENTANYTREE_EXECUTE:-0}" != "1" ]]; then
  echo "Refusing full execution. Set SEGMENTANYTREE_EXECUTE=1 only after the pilot passes." >&2
  exit 2
fi

module purge
module load miniforge3/25.3.0-python3.12.10
source "$HOME/fastscratch/venvs/treebench/bin/activate"

cd "$HOME/scratch/tree-seg-benchmark"
mkdir -p logs/segmentanytree_for_instance

python scripts/methods/run_segmentanytree_for_instance.py \
  --config configs/for_instance_segmentanytree_benchmark.yml \
  --array-index "$SLURM_ARRAY_TASK_ID"
