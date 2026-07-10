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
OUTPUT_DIR="${SEGMENTANYTREE_RELEASED_CHECKPOINT_DIR:-$HOME/fastscratch/segmentanytree_pretrained/released_model_bundle}"
OUTPUT="$OUTPUT_DIR/PointGroup-PAPER.pt"
OVERRIDES="$OUTPUT_DIR/.hydra/overrides.yaml"
SOURCE_DIR="/home/nibio/mutable-outside-world/model_file"
EXPECTED_SHA256="${SEGMENTANYTREE_RELEASED_CHECKPOINT_SHA256:-0b4d74b4644e37a16f59008ad0f5c62894fc4d2d906f3abd803bbfc5b5dd803a}"

cd "$PROJECT_ROOT"
module purge
module load apptainer/1.3.6
test -f "$IMAGE"
mkdir -p "$(dirname "$OUTPUT_DIR")"

if [[ -f "$OUTPUT" && -s "$OVERRIDES" ]]; then
  actual=$(sha256sum "$OUTPUT" | awk '{print $1}')
  if [[ "$actual" == "$EXPECTED_SHA256" ]]; then
    echo "released_model_bundle=ready"
    echo "released_checkpoint=$OUTPUT"
    echo "released_overrides=$OVERRIDES"
    exit 0
  fi
  echo "Existing released checkpoint has unexpected SHA-256: $actual" >&2
  exit 2
fi
if [[ -e "$OUTPUT_DIR" ]]; then
  echo "Existing released model bundle is incomplete: $OUTPUT_DIR" >&2
  echo "Archive it before retrying; PointGroup-PAPER.pt and .hydra/overrides.yaml are required." >&2
  exit 2
fi

partial="$OUTPUT_DIR.partial.${SLURM_JOB_ID:-manual}"
cleanup_partial() {
  rm -rf "$partial"
}
trap cleanup_partial EXIT
mkdir -p "$partial"
apptainer exec --bind "$partial:/sat_export" "$IMAGE" \
  cp -a "$SOURCE_DIR/." /sat_export/
test -s "$partial/.hydra/overrides.yaml"
actual=$(sha256sum "$partial/PointGroup-PAPER.pt" | awk '{print $1}')
if [[ "$actual" != "$EXPECTED_SHA256" ]]; then
  echo "Released checkpoint SHA-256 mismatch: $actual" >&2
  exit 2
fi
printf '%s  %s\n' "$EXPECTED_SHA256" "PointGroup-PAPER.pt" > "$partial/SHA256SUMS"
mv "$partial" "$OUTPUT_DIR"
trap - EXIT
printf '%s  %s\n' "$EXPECTED_SHA256" "PointGroup-PAPER.pt" > "$OUTPUT_DIR/SHA256SUMS"
echo "released_model_bundle=$OUTPUT_DIR"
echo "released_checkpoint=$OUTPUT"
echo "released_overrides=$OVERRIDES"
