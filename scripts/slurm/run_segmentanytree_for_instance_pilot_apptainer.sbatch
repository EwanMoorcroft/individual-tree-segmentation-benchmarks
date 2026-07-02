#!/bin/bash -l
#SBATCH --job-name=for_sat_pilot
#SBATCH --partition=gpu-l40s
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --output=logs/segmentanytree_for_instance/%x_%j.out
#SBATCH --error=logs/segmentanytree_for_instance/%x_%j.err

set -euo pipefail

if [[ "${SEGMENTANYTREE_EXECUTE:-0}" != "1" ]]; then
  echo "Refusing execution. Set SEGMENTANYTREE_EXECUTE=1 after reviewing the pilot command." >&2
  exit 2
fi

cd "$HOME/scratch/tree-seg-benchmark"
module purge
module load miniforge3/25.3.0-python3.12.10
module load apptainer/1.3.6
source "$HOME/fastscratch/venvs/treebench/bin/activate"

bash scripts/slurm/run_segmentanytree_for_instance_task.sh \
  --plot-path CULS/plot_1_annotated.las
