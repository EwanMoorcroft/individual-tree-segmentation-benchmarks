#!/bin/bash -l
#SBATCH --job-name=for_sat_paper
#SBATCH --partition=gpu-l40s-low
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --array=0-10%2
#SBATCH --output=logs/segmentanytree_for_instance/%x_%A_%a.out
#SBATCH --error=logs/segmentanytree_for_instance/%x_%A_%a.err

set -euo pipefail

if [[ "${SEGMENTANYTREE_EXECUTE:-0}" != "1" ]]; then
  echo "Refusing execution. Set SEGMENTANYTREE_EXECUTE=1 after reviewing the test selection." >&2
  exit 2
fi

cd "$HOME/scratch/tree-seg-benchmark"
module purge
module load miniforge3/25.3.0-python3.12.10
module load apptainer/1.3.6
source "$HOME/fastscratch/venvs/treebench/bin/activate"

bash scripts/slurm/run_segmentanytree_for_instance_paper_test_task.sh \
  --split test \
  --array-index "$SLURM_ARRAY_TASK_ID"
