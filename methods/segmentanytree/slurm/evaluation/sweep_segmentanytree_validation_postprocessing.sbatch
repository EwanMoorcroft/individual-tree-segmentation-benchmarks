#!/bin/bash -l
#SBATCH --job-name=sat_postprocess_sweep
#SBATCH --partition=nodes
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=logs/segmentanytree_for_instance/%x_%j.out
#SBATCH --error=logs/segmentanytree_for_instance/%x_%j.err

set -euo pipefail

PROJECT_ROOT="$HOME/scratch/tree-seg-benchmark"
RUN_ID="${SEGMENTANYTREE_TRAINING_RUN_ID:-sat_for_quicktune_to49_20260706_140730}"

cd "$PROJECT_ROOT"
module purge
module load miniforge3/25.3.0-python3.12.10
source "$HOME/fastscratch/venvs/treebench/bin/activate"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

python methods/segmentanytree/scripts/diagnostics/sweep_sat_validation_postprocessing.py \
  --run-id "$RUN_ID" \
  --split trained_validation \
  --min-predicted-instance-points "${SEGMENTANYTREE_SWEEP_MIN_POINTS:-0,500,1000,2500,5000,10000}" \
  --min-predicted-tree-fraction "${SEGMENTANYTREE_SWEEP_TREE_FRACTIONS:-0,0.6,0.7,0.8,0.9,0.95}"
