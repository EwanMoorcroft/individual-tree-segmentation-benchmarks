#!/bin/bash

set -euo pipefail

if [[ "${SEGMENTANYTREE_FINETUNE_DEV_CONFIRMED:-0}" != "1" ]]; then
  echo "Set SEGMENTANYTREE_FINETUNE_DEV_CONFIRMED=1 after reviewing the development-only route." >&2
  exit 2
fi

PROJECT_ROOT="${SEGMENTANYTREE_PROJECT_ROOT:-$HOME/scratch/tree-seg-benchmark}"
STAMP="${SEGMENTANYTREE_FINETUNE_STAMP:-$(date +%Y%m%d_%H%M%S)}"
RUN_ID="segmentanytree_for-instance_fine_tuned_on_dev_${STAMP}"
SMOKE_ID="${RUN_ID}_smoke"
REJECTED_RUN_ID="segmentanytree_for-instance_fine_tuned_on_dev_20260708_215054_full"
RELEASED_SHA256="0b4d74b4644e37a16f59008ad0f5c62894fc4d2d906f3abd803bbfc5b5dd803a"
CHECKPOINT_BUNDLE="$HOME/fastscratch/segmentanytree_pretrained/released_model_bundle"
RELEASED_CHECKPOINT="$CHECKPOINT_BUNDLE/PointGroup-PAPER.pt"
FULL_MANIFEST="$PROJECT_ROOT/results/metadata/segmentanytree_for_instance/training_splits/full_split_manifest.json"
FULL_DATA="$HOME/fastscratch/segmentanytree_for_instance_training/full/treeinsfused/raw"
STAGE1_TEST_FREEZE="${SEGMENTANYTREE_STAGE1_TEST_FREEZE:?Set SEGMENTANYTREE_STAGE1_TEST_FREEZE to the reviewed Stage 1 freeze JSON.}"
STAGE1_FINAL_SUMMARY="${SEGMENTANYTREE_STAGE1_FINAL_SUMMARY:?Set SEGMENTANYTREE_STAGE1_FINAL_SUMMARY to the completed Stage 1 summary CSV.}"
TRAIN_PARTITION="${SEGMENTANYTREE_FINETUNE_PARTITION:-gpu-l40s}"
FREEZE_MANIFEST="$PROJECT_ROOT/results/metadata/segmentanytree_for_instance/finetune_freezes/$RUN_ID.json"
TRAINING_ROOT="$HOME/fastscratch/segmentanytree_for_instance_checkpoints/$RUN_ID"
SMOKE_ROOT="$HOME/fastscratch/segmentanytree_for_instance_checkpoints/$SMOKE_ID"
PREDICTIONS="$PROJECT_ROOT/data/predictions/segmentanytree/for_instance_trained_validation/$RUN_ID"
METRICS="$PROJECT_ROOT/results/metadata/segmentanytree_for_instance/trained_validation/$RUN_ID"
TABLES="$PROJECT_ROOT/results/tables/segmentanytree_for_instance/trained_validation/$RUN_ID"
SUMMARY="$TABLES/validation_summary.csv"
STATE_FILE="$HOME/fastscratch/segmentanytree_finetuned_dev_${RUN_ID}.env"
MIN_FREE_BYTES="${SEGMENTANYTREE_FINETUNE_MIN_FREE_BYTES:-85899345920}"
submitted_jobs=()

cancel_partial_submission() {
  local status=$?
  if ((${#submitted_jobs[@]})); then
    scancel "${submitted_jobs[@]}" 2>/dev/null || true
  fi
  echo "Submission failed; cancelled jobs created by this attempt." >&2
  exit "$status"
}
trap cancel_partial_submission ERR

if [[ "$RUN_ID" == "$REJECTED_RUN_ID" ]]; then
  echo "Refusing to reuse the rejected fine-tuning run ID." >&2
  exit 2
fi
if [[ ! "$STAMP" =~ ^[0-9]{8}_[0-9]{6}$ ]]; then
  echo "SEGMENTANYTREE_FINETUNE_STAMP must use YYYYMMDD_HHMMSS." >&2
  exit 2
fi
if [[ ! "$TRAIN_PARTITION" =~ ^[A-Za-z0-9_-]+$ ]]; then
  echo "Invalid training partition: $TRAIN_PARTITION" >&2
  exit 2
fi

cd "$PROJECT_ROOT"
mkdir -p logs/segmentanytree_for_instance
STAGE1_TEST_FREEZE=$(realpath "$STAGE1_TEST_FREEZE")
STAGE1_FINAL_SUMMARY=$(realpath "$STAGE1_FINAL_SUMMARY")
test -f "$HOME/scratch/containers/segment-any-tree_latest.sif"
test -d "$HOME/fastscratch/venvs/treebench"
test -f "$FULL_MANIFEST"
test -d "$FULL_DATA"
test -f "$RELEASED_CHECKPOINT"
test -s "$CHECKPOINT_BUNDLE/.hydra/overrides.yaml"
test -f "$STAGE1_TEST_FREEZE"
test -f "$STAGE1_FINAL_SUMMARY"
test "$(git -C external/SegmentAnyTree rev-parse HEAD)" = "a3561ed8447bbb7938f059ba65a3e9c97d6e2ee9"
if [[ ! "$MIN_FREE_BYTES" =~ ^[1-9][0-9]*$ ]]; then
  echo "SEGMENTANYTREE_FINETUNE_MIN_FREE_BYTES must be a positive integer." >&2
  exit 2
fi
scratch_free=$(df -PB1 "$PROJECT_ROOT" | awk 'NR == 2 {print $4}')
fastscratch_free=$(df -PB1 "$HOME/fastscratch" | awk 'NR == 2 {print $4}')
if ((scratch_free < MIN_FREE_BYTES || fastscratch_free < MIN_FREE_BYTES)); then
  echo "Need at least $MIN_FREE_BYTES free bytes on scratch and fastscratch." >&2
  echo "scratch=$scratch_free fastscratch=$fastscratch_free" >&2
  exit 2
fi
if [[ -n "$(git status --porcelain)" ]]; then
  echo "Barkla repository is not clean; refusing fine-tuning submission." >&2
  exit 2
fi
for target in "$FREEZE_MANIFEST" "$TRAINING_ROOT" "$SMOKE_ROOT" "$PREDICTIONS" "$METRICS" "$TABLES"; do
  if [[ -e "$target" ]]; then
    echo "Fine-tuning target already exists: $target" >&2
    exit 2
  fi
done

module purge
module load miniforge3/25.3.0-python3.12.10
source "$HOME/fastscratch/venvs/treebench/bin/activate"
python methods/segmentanytree/scripts/evaluation/prepare_finetuned_dev_training_freeze.py \
  --split-manifest "$FULL_MANIFEST" \
  --checkpoint-bundle "$CHECKPOINT_BUNDLE" \
  --stage1-test-freeze "$STAGE1_TEST_FREEZE" \
  --stage1-final-summary "$STAGE1_FINAL_SUMMARY" \
  --run-id "$RUN_ID" \
  --output "$FREEZE_MANIFEST"

finetune_smoke_job=$(sbatch --parsable \
  --partition="$TRAIN_PARTITION" --time=02:00:00 --cpus-per-task=16 --mem=64G \
  --export="ALL,SEGMENTANYTREE_EXECUTE=1,SEGMENTANYTREE_FULL_TRAIN_CONFIRMED=1,SEGMENTANYTREE_TRAIN_PROFILE=full,SEGMENTANYTREE_TRAINING_RUN_ID=$SMOKE_ID,SEGMENTANYTREE_TRAIN_EPOCHS=1,SEGMENTANYTREE_TRAIN_BATCH_SIZE=8,SEGMENTANYTREE_TRAIN_BASE_LR=0.0001,SEGMENTANYTREE_MEANSHIFT_JOBS=1,SEGMENTANYTREE_OMP_NUM_THREADS=1,SEGMENTANYTREE_STALL_TIMEOUT_SECONDS=2400,SEGMENTANYTREE_PRETRAINED_CHECKPOINT=$RELEASED_CHECKPOINT,SEGMENTANYTREE_PRETRAINED_CHECKPOINT_SHA256=$RELEASED_SHA256,SEGMENTANYTREE_PRETRAINED_WEIGHT_NAME=latest" \
  methods/segmentanytree/slurm/training/train_segmentanytree_for_instance_full.sbatch)
submitted_jobs+=("$finetune_smoke_job")

finetune_train_job=$(sbatch --parsable \
  --partition="$TRAIN_PARTITION" --time=12:00:00 --cpus-per-task=16 --mem=64G \
  --dependency="afterok:$finetune_smoke_job" \
  --export="ALL,SEGMENTANYTREE_EXECUTE=1,SEGMENTANYTREE_FULL_TRAIN_CONFIRMED=1,SEGMENTANYTREE_TRAIN_PROFILE=full,SEGMENTANYTREE_TRAINING_RUN_ID=$RUN_ID,SEGMENTANYTREE_TRAIN_EPOCHS=35,SEGMENTANYTREE_TRAIN_BATCH_SIZE=8,SEGMENTANYTREE_TRAIN_BASE_LR=0.0001,SEGMENTANYTREE_MEANSHIFT_JOBS=1,SEGMENTANYTREE_OMP_NUM_THREADS=1,SEGMENTANYTREE_STALL_TIMEOUT_SECONDS=2400,SEGMENTANYTREE_PRETRAINED_CHECKPOINT=$RELEASED_CHECKPOINT,SEGMENTANYTREE_PRETRAINED_CHECKPOINT_SHA256=$RELEASED_SHA256,SEGMENTANYTREE_PRETRAINED_WEIGHT_NAME=latest" \
  methods/segmentanytree/slurm/training/train_segmentanytree_for_instance_full.sbatch)
submitted_jobs+=("$finetune_train_job")

validation_inference_job=$(sbatch --parsable \
  --array=0-4%2 \
  --dependency="afterok:$finetune_train_job" \
  --export="ALL,SEGMENTANYTREE_EXECUTE=1,SEGMENTANYTREE_TRAIN_PROFILE=full,SEGMENTANYTREE_TRAINING_RUN_ID=$RUN_ID" \
  methods/segmentanytree/slurm/inference/run_segmentanytree_for_instance_trained_validation.sbatch)
submitted_jobs+=("$validation_inference_job")

validation_evaluation_job=$(sbatch --parsable \
  --array=0-4%4 \
  --dependency="afterok:$validation_inference_job" \
  --export="ALL,SEGMENTANYTREE_TRAIN_PROFILE=full,SEGMENTANYTREE_TRAINING_RUN_ID=$RUN_ID" \
  methods/segmentanytree/slurm/evaluation/evaluate_segmentanytree_for_instance_trained_validation.sbatch)
submitted_jobs+=("$validation_evaluation_job")

validation_gate_job=$(sbatch --parsable \
  --dependency="afterok:$validation_evaluation_job" \
  --export="ALL,SEGMENTANYTREE_VARIANT_LABEL=fine_tuned_on_dev_validation,SEGMENTANYTREE_VARIANT_METRICS_ROOT=$METRICS,SEGMENTANYTREE_VARIANT_EXPECTED_PLOTS=5,SEGMENTANYTREE_VARIANT_EXPECTED_SPLIT=dev,SEGMENTANYTREE_VARIANT_REQUIRE_PREDICTIONS=1,SEGMENTANYTREE_VARIANT_SUMMARY=$SUMMARY" \
  methods/segmentanytree/slurm/evaluation/validate_segmentanytree_variant.sbatch)
submitted_jobs+=("$validation_gate_job")

submitted_epoch=$(date +%s)
{
  printf 'SAT_FINETUNE_DEV_RUN_ID=%q\n' "$RUN_ID"
  printf 'SAT_FINETUNE_DEV_SUBMITTED_EPOCH=%q\n' "$submitted_epoch"
  printf 'SAT_FINETUNE_DEV_SMOKE_JOB=%q\n' "$finetune_smoke_job"
  printf 'SAT_FINETUNE_DEV_TRAIN_JOB=%q\n' "$finetune_train_job"
  printf 'SAT_FINETUNE_DEV_INFERENCE_JOB=%q\n' "$validation_inference_job"
  printf 'SAT_FINETUNE_DEV_EVALUATION_JOB=%q\n' "$validation_evaluation_job"
  printf 'SAT_FINETUNE_DEV_GATE_JOB=%q\n' "$validation_gate_job"
  printf 'SAT_FINETUNE_DEV_CHECKPOINT_ROOT=%q\n' "$TRAINING_ROOT"
  printf 'SAT_FINETUNE_DEV_METRICS=%q\n' "$METRICS"
  printf 'SAT_FINETUNE_DEV_SUMMARY=%q\n' "$SUMMARY"
  printf 'SAT_FINETUNE_DEV_FREEZE_MANIFEST=%q\n' "$FREEZE_MANIFEST"
} > "$STATE_FILE"
trap - ERR

echo "run_id=$RUN_ID"
echo "smoke_job=$finetune_smoke_job training_job=$finetune_train_job"
echo "inference_job=$validation_inference_job evaluation_job=$validation_evaluation_job gate_job=$validation_gate_job"
echo "state_file=$STATE_FILE"
echo "freeze_manifest=$FREEZE_MANIFEST"
echo "summary=$SUMMARY"
echo "No held-out test job was submitted."
