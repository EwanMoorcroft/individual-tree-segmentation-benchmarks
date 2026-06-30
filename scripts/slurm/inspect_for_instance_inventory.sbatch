#!/bin/bash -l
#SBATCH --job-name=for_inventory
#SBATCH --partition=nodes
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --output=logs/for_instance_tls2trees/%x_%j.out
#SBATCH --error=logs/for_instance_tls2trees/%x_%j.err

set -euo pipefail

module purge
module load miniforge3/25.3.0-python3.12.10
source ~/fastscratch/venvs/treebench/bin/activate

cd "$HOME/scratch/tree-seg-benchmark"
mkdir -p results/metadata/for_instance_tls2trees

python scripts/data/inspect_for_instance_inventory.py \
  --dataset-root "$HOME/data/datasets/for_instance/FORinstance_dataset" \
  --csv-output results/metadata/for_instance_tls2trees/inventory.csv \
  --json-output results/metadata/for_instance_tls2trees/inventory.json
