#!/bin/bash -l
#SBATCH --job-name=frdr_inventory
#SBATCH --partition=short
#SBATCH --time=00:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --output=logs/tls2trees_frdr_full/%x_%j.out
#SBATCH --error=logs/tls2trees_frdr_full/%x_%j.err

set -euo pipefail

module purge
module load miniforge3/25.3.0-python3.12.10
source ~/fastscratch/venvs/treebench/bin/activate

cd "$HOME/scratch/tree-seg-benchmark"
mkdir -p logs/tls2trees_frdr_full results/metadata

echo "job_id=${SLURM_JOB_ID:-unset}"
echo "hostname=$(hostname)"
echo "started_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
python --version

python scripts/data/inspect_frdr_dataset_inventory.py \
  --dataset-root "$HOME/data/datasets/frdr_treeiso" \
  --csv-out results/metadata/frdr_dataset_inventory.csv \
  --json-out results/metadata/frdr_dataset_inventory.json

echo "finished_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
