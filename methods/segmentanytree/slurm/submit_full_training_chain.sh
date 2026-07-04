#!/bin/bash

set -euo pipefail

PROJECT_ROOT="${SEGMENTANYTREE_PROJECT_ROOT:-$HOME/scratch/tree-seg-benchmark}"
MANIFEST="$PROJECT_ROOT/results/metadata/segmentanytree_for_instance/training_splits/full_split_manifest.json"
LOG_ROOT="$PROJECT_ROOT/logs/segmentanytree_for_instance"

if [[ "${SEGMENTANYTREE_SUBMIT_CONFIRMED:-0}" != "1" ]]; then
  echo "Refusing submission. Review the full split manifest, then set SEGMENTANYTREE_SUBMIT_CONFIRMED=1." >&2
  exit 2
fi

cd "$PROJECT_ROOT"
mkdir -p "$LOG_ROOT"
test -f "$MANIFEST"

read -r TRAIN_COUNT VALIDATION_COUNT TEST_COUNT < <(
  python - "$MANIFEST" <<'PY'
import json
import sys

manifest = json.loads(open(sys.argv[1], encoding="utf-8").read())
counts = manifest["selected_role_counts"]
print(
    counts.get("train", 0),
    counts.get("val", 0),
    counts.get("held_out_test", 0),
)
PY
)

if [[ "$TRAIN_COUNT" != "16" || "$VALIDATION_COUNT" != "5" || "$TEST_COUNT" != "0" ]]; then
  echo "Unexpected full-profile selection: train=$TRAIN_COUNT val=$VALIDATION_COUNT held_out_test=$TEST_COUNT" >&2
  exit 2
fi

RUN_ID="${SEGMENTANYTREE_TRAINING_RUN_ID:-sat_for_train_full_$(date +%Y%m%d_%H%M%S)}"
VALIDATION_LAST=$((VALIDATION_COUNT - 1))

TRAIN_JOB=$(sbatch --parsable \
  --export="ALL,SEGMENTANYTREE_EXECUTE=1,SEGMENTANYTREE_FULL_TRAIN_CONFIRMED=1,SEGMENTANYTREE_TRAIN_PROFILE=full,SEGMENTANYTREE_TRAINING_RUN_ID=$RUN_ID" \
  methods/segmentanytree/slurm/training/train_segmentanytree_for_instance_full.sbatch)

VALIDATE_JOB=$(sbatch --parsable \
  --array="0-${VALIDATION_LAST}%1" \
  --dependency="afterok:${TRAIN_JOB}" \
  --export="ALL,SEGMENTANYTREE_EXECUTE=1,SEGMENTANYTREE_TRAIN_PROFILE=full,SEGMENTANYTREE_TRAINING_RUN_ID=$RUN_ID" \
  methods/segmentanytree/slurm/inference/run_segmentanytree_for_instance_trained_validation.sbatch)

EVALUATE_JOB=$(sbatch --parsable \
  --array="0-${VALIDATION_LAST}%4" \
  --dependency="afterok:${VALIDATE_JOB}" \
  --export="ALL,SEGMENTANYTREE_TRAIN_PROFILE=full,SEGMENTANYTREE_TRAINING_RUN_ID=$RUN_ID" \
  methods/segmentanytree/slurm/evaluation/evaluate_segmentanytree_for_instance_trained_validation.sbatch)

STATE_FILE="$HOME/fastscratch/segmentanytree_full_jobs_${RUN_ID}.env"
{
  printf 'FULL_RUN_ID=%q\n' "$RUN_ID"
  printf 'FULL_TRAIN_JOB=%q\n' "$TRAIN_JOB"
  printf 'FULL_VALIDATE_JOB=%q\n' "$VALIDATE_JOB"
  printf 'FULL_EVALUATE_JOB=%q\n' "$EVALUATE_JOB"
} > "$STATE_FILE"

echo "RUN_ID=$RUN_ID"
echo "TRAIN=$TRAIN_JOB"
echo "VALIDATE=$VALIDATE_JOB"
echo "EVALUATE=$EVALUATE_JOB"
echo "STATE_FILE=$STATE_FILE"
