#!/bin/bash -l
#SBATCH --job-name=for_sat_pre_dev_gate
#SBATCH --partition=nodes
#SBATCH --time=00:10:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --output=logs/segmentanytree_for_instance/%x_%j.out
#SBATCH --error=logs/segmentanytree_for_instance/%x_%j.err

set -euo pipefail

PROJECT_ROOT="$HOME/scratch/tree-seg-benchmark"
RUN_METADATA="${SEGMENTANYTREE_DEV_SMOKE_RUN_METADATA:?Set SEGMENTANYTREE_DEV_SMOKE_RUN_METADATA.}"
METRICS="${SEGMENTANYTREE_DEV_SMOKE_METRICS:?Set SEGMENTANYTREE_DEV_SMOKE_METRICS.}"
CHECKPOINT_BUNDLE="${SEGMENTANYTREE_CHECKPOINT_DIR:?Set SEGMENTANYTREE_CHECKPOINT_DIR.}"
EVIDENCE="${SEGMENTANYTREE_DEV_SMOKE_EVIDENCE:?Set SEGMENTANYTREE_DEV_SMOKE_EVIDENCE.}"

cd "$PROJECT_ROOT"
module purge
module load miniforge3/25.3.0-python3.12.10
source "$HOME/fastscratch/venvs/treebench/bin/activate"

python methods/segmentanytree/scripts/evaluation/validate_published_pretrained_dev_smoke.py \
  --run-metadata "$RUN_METADATA" \
  --metrics "$METRICS" \
  --checkpoint-bundle "$CHECKPOINT_BUNDLE" \
  --expected-relative-path CULS/plot_1_annotated.las \
  --output "$EVIDENCE"
