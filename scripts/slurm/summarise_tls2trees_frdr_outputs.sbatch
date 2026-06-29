#!/bin/bash -l
#SBATCH --job-name=summarise_tls2trees_frdr
#SBATCH --partition=short
#SBATCH --time=01:00:00
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
mkdir -p \
  logs/tls2trees_frdr_full \
  results/metadata/tls2trees_outputs \
  results/tables

echo "job_id=${SLURM_JOB_ID:-unset}"
echo "hostname=$(hostname)"
echo "started_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"

python scripts/methods/summarise_tls2trees_outputs.py \
  --config configs/frdr_tls2trees_benchmark.yml \
  --all-plots

echo "finished_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
