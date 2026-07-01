#!/bin/bash -l
#SBATCH --job-name=for_sat_inventory
#SBATCH --partition=nodes
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --output=logs/segmentanytree_for_instance/%x_%j.out
#SBATCH --error=logs/segmentanytree_for_instance/%x_%j.err

set -euo pipefail

module purge
module load miniforge3/25.3.0-python3.12.10
source "$HOME/fastscratch/venvs/treebench/bin/activate"

cd "$HOME/scratch/tree-seg-benchmark"
mkdir -p results/metadata/segmentanytree_for_instance

python scripts/data/inspect_for_instance_inventory.py \
  --dataset-root "$HOME/data/datasets/for_instance/FORinstance_dataset" \
  --split-metadata "$HOME/data/datasets/for_instance/FORinstance_dataset/data_split_metadata.csv" \
  --csv-output results/metadata/segmentanytree_for_instance/inventory.csv \
  --json-output results/metadata/segmentanytree_for_instance/inventory.json
