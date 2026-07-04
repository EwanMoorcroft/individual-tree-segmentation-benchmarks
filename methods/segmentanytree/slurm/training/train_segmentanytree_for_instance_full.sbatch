#!/bin/bash -l
#SBATCH --job-name=for_sat_train_full
#SBATCH --partition=gpu-l40s-low
#SBATCH --gres=gpu:1
#SBATCH --time=23:59:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --output=logs/segmentanytree_for_instance/%x_%j.out
#SBATCH --error=logs/segmentanytree_for_instance/%x_%j.err

set -euo pipefail

cd "$HOME/scratch/tree-seg-benchmark"
module purge
module load miniforge3/25.3.0-python3.12.10
module load apptainer/1.3.6
source "$HOME/fastscratch/venvs/treebench/bin/activate"

if [[ "${SEGMENTANYTREE_FULL_TRAIN_CONFIRMED:-0}" != "1" ]]; then
  echo "Refusing full training. Validate the pilot, then set SEGMENTANYTREE_FULL_TRAIN_CONFIRMED=1." >&2
  exit 2
fi

export SEGMENTANYTREE_TRAIN_PROFILE=full
bash methods/segmentanytree/slurm/training/train_segmentanytree_for_instance_task.sh
