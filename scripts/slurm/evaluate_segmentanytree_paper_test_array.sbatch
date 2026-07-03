#!/bin/bash -l
#SBATCH --job-name=for_sat_paper_eval
#SBATCH --partition=nodes
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --array=0-10%4
#SBATCH --output=logs/segmentanytree_for_instance/%x_%A_%a.out
#SBATCH --error=logs/segmentanytree_for_instance/%x_%A_%a.err

set -euo pipefail

module purge
module load miniforge3/25.3.0-python3.12.10
source "$HOME/fastscratch/venvs/treebench/bin/activate"

cd "$HOME/scratch/tree-seg-benchmark"
mkdir -p logs/segmentanytree_for_instance

bash scripts/slurm/evaluate_segmentanytree_pointwise_task.sh \
  --split test \
  --array-index "$SLURM_ARRAY_TASK_ID"
