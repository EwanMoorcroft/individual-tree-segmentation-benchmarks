#!/bin/bash -l
#SBATCH --job-name=for_sat_pre_dev_eval
#SBATCH --partition=nodes
#SBATCH --time=00:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --output=logs/segmentanytree_for_instance/%x_%j.out
#SBATCH --error=logs/segmentanytree_for_instance/%x_%j.err

set -euo pipefail

PROJECT_ROOT="$HOME/scratch/tree-seg-benchmark"
OUTPUT_ROOT="${SEGMENTANYTREE_POINTWISE_OUTPUT_ROOT:?Set SEGMENTANYTREE_POINTWISE_OUTPUT_ROOT.}"
METRICS_ROOT="${SEGMENTANYTREE_POINTWISE_EVALUATION_ROOT:?Set SEGMENTANYTREE_POINTWISE_EVALUATION_ROOT.}"
TABLE_ROOT="${SEGMENTANYTREE_POINTWISE_TABLE_ROOT:?Set SEGMENTANYTREE_POINTWISE_TABLE_ROOT.}"

cd "$PROJECT_ROOT"
module purge
module load miniforge3/25.3.0-python3.12.10
source "$HOME/fastscratch/venvs/treebench/bin/activate"

SEGMENTANYTREE_REQUIRED_SPLIT=dev \
SEGMENTANYTREE_POINTWISE_OUTPUT_ROOT="$OUTPUT_ROOT" \
SEGMENTANYTREE_POINTWISE_EVALUATION_ROOT="$METRICS_ROOT" \
SEGMENTANYTREE_POINTWISE_TABLE_ROOT="$TABLE_ROOT" \
  bash methods/segmentanytree/slurm/evaluation/evaluate_segmentanytree_pointwise_task.sh \
    --split dev \
    --plot-path CULS/plot_1_annotated.las
