#!/bin/bash -l
#SBATCH --job-name=for_sat_summary
#SBATCH --partition=nodes
#SBATCH --time=00:15:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --output=logs/segmentanytree_for_instance/%x_%j.out
#SBATCH --error=logs/segmentanytree_for_instance/%x_%j.err

set -euo pipefail

module purge
module load miniforge3/25.3.0-python3.12.10
source "$HOME/fastscratch/venvs/treebench/bin/activate"

cd "$HOME/scratch/tree-seg-benchmark"
mkdir -p logs/segmentanytree_for_instance

python scripts/evaluation/summarise_for_instance_segmentanytree_benchmark.py \
  --metrics-root results/tables/segmentanytree_for_instance/per_plot \
  --inventory-csv results/metadata/segmentanytree_for_instance/inventory.csv
