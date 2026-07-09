#!/bin/bash -l
#SBATCH --job-name=for_sat_release_ckpt
#SBATCH --partition=nodes
#SBATCH --time=00:15:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --output=logs/segmentanytree_for_instance/%x_%j.out
#SBATCH --error=logs/segmentanytree_for_instance/%x_%j.err

set -euo pipefail

PROJECT_ROOT="$HOME/scratch/tree-seg-benchmark"
IMAGE="$HOME/scratch/containers/segment-any-tree_latest.sif"
OUTPUT_DIR="${SEGMENTANYTREE_RELEASED_CHECKPOINT_DIR:-$HOME/fastscratch/segmentanytree_pretrained/released}"
OUTPUT="$OUTPUT_DIR/PointGroup-PAPER.pt"
SOURCE="/home/nibio/mutable-outside-world/model_file/PointGroup-PAPER.pt"
EXPECTED_SHA256="${SEGMENTANYTREE_RELEASED_CHECKPOINT_SHA256:-0b4d74b4644e37a16f59008ad0f5c62894fc4d2d906f3abd803bbfc5b5dd803a}"

cd "$PROJECT_ROOT"
module purge
module load apptainer/1.3.6
mkdir -p "$OUTPUT_DIR"
test -f "$IMAGE"

if [[ -f "$OUTPUT" ]]; then
  actual=$(sha256sum "$OUTPUT" | awk '{print $1}')
  if [[ "$actual" == "$EXPECTED_SHA256" ]]; then
    echo "released_checkpoint=ready"
    exit 0
  fi
  echo "Existing released checkpoint has unexpected SHA-256: $actual" >&2
  exit 2
fi

partial=".PointGroup-PAPER.pt.partial.${SLURM_JOB_ID:-manual}"
apptainer exec --bind "$OUTPUT_DIR:/sat_export" "$IMAGE" \
  cp "$SOURCE" "/sat_export/$partial"
actual=$(sha256sum "$OUTPUT_DIR/$partial" | awk '{print $1}')
if [[ "$actual" != "$EXPECTED_SHA256" ]]; then
  rm -f "$OUTPUT_DIR/$partial"
  echo "Released checkpoint SHA-256 mismatch: $actual" >&2
  exit 2
fi
mv "$OUTPUT_DIR/$partial" "$OUTPUT"
printf '%s  %s\n' "$EXPECTED_SHA256" "PointGroup-PAPER.pt" > "$OUTPUT_DIR/SHA256SUMS"
echo "released_checkpoint=$OUTPUT"
