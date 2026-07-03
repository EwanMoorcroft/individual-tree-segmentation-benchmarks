#!/bin/bash -l
#SBATCH --job-name=for_sat_train_pilot
#SBATCH --partition=gpu-l40s-low
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
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

export SEGMENTANYTREE_TRAIN_PROFILE=pilot
bash scripts/slurm/train_segmentanytree_for_instance_task.sh
