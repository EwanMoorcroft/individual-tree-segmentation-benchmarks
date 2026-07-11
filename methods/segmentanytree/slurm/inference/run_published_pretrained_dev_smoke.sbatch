#!/bin/bash -l
#SBATCH --job-name=for_sat_pre_dev
#SBATCH --partition=gpu-l40s-low
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --output=logs/segmentanytree_for_instance/%x_%j.out
#SBATCH --error=logs/segmentanytree_for_instance/%x_%j.err

set -euo pipefail

if [[ "${SEGMENTANYTREE_PRETRAINED_DEV_SMOKE_CONFIRMED:-0}" != "1" ]]; then
  echo "Set SEGMENTANYTREE_PRETRAINED_DEV_SMOKE_CONFIRMED=1 after reviewing the development-only smoke route." >&2
  exit 2
fi

PROJECT_ROOT="$HOME/scratch/tree-seg-benchmark"
OUTPUT_ROOT="${SEGMENTANYTREE_POINTWISE_OUTPUT_ROOT:?Set SEGMENTANYTREE_POINTWISE_OUTPUT_ROOT.}"
METADATA_ROOT="${SEGMENTANYTREE_POINTWISE_RUN_METADATA_ROOT:?Set SEGMENTANYTREE_POINTWISE_RUN_METADATA_ROOT.}"
CHECKPOINT_BUNDLE="${SEGMENTANYTREE_CHECKPOINT_DIR:?Set SEGMENTANYTREE_CHECKPOINT_DIR.}"

cd "$PROJECT_ROOT"
module purge
module load miniforge3/25.3.0-python3.12.10
module load apptainer/1.3.6
source "$HOME/fastscratch/venvs/treebench/bin/activate"

test -f "$CHECKPOINT_BUNDLE/PointGroup-PAPER.pt"
test -s "$CHECKPOINT_BUNDLE/.hydra/overrides.yaml"

SEGMENTANYTREE_REQUIRED_SPLIT=dev \
SEGMENTANYTREE_RUN_TYPE=published_pretrained \
SEGMENTANYTREE_POINTWISE_OUTPUT_ROOT="$OUTPUT_ROOT" \
SEGMENTANYTREE_POINTWISE_RUN_METADATA_ROOT="$METADATA_ROOT" \
SEGMENTANYTREE_CHECKPOINT_DIR="$CHECKPOINT_BUNDLE" \
  bash methods/segmentanytree/slurm/inference/run_segmentanytree_for_instance_paper_test_task.sh \
    --split dev \
    --plot-path CULS/plot_1_annotated.las
