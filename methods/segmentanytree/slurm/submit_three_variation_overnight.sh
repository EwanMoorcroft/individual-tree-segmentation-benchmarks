#!/bin/bash

set -euo pipefail

PROJECT_ROOT="${SEGMENTANYTREE_PROJECT_ROOT:-$HOME/scratch/tree-seg-benchmark}"
STAMP="${SEGMENTANYTREE_OVERNIGHT_STAMP:-$(date +%Y%m%d_%H%M%S)}"
RELEASED_SHA256="0b4d74b4644e37a16f59008ad0f5c62894fc4d2d906f3abd803bbfc5b5dd803a"
RELEASED_DIR="$HOME/fastscratch/segmentanytree_pretrained/released"
RELEASED_CHECKPOINT="$RELEASED_DIR/PointGroup-PAPER.pt"
RETRAINED_RUN_ID="sat_for_quicktune_to49_20260706_140730"
RETRAINED_EVALUATION_ID="sat_for_quicktune_to49_20260706_140730_final_test_aligned_20260709_212341"
RETRAINED_CHECKPOINT="$HOME/fastscratch/segmentanytree_for_instance_checkpoints/$RETRAINED_RUN_ID/run/PointGroup-PAPER.pt"
RETRAINED_SHA256="9b871b15ac61589ea27c507e054ee66d3f543caa01fed9a5b790e4ce97bcecea"
RETRAINED_METRICS="$PROJECT_ROOT/results/metadata/segmentanytree_for_instance/trained_test/$RETRAINED_EVALUATION_ID"
PRETRAINED_ID="sat_published_pretrained_aligned_$STAMP"
FINETUNE_ID="sat_fine_tuned_on_dev_e35_$STAMP"
FINETUNE_SMOKE_ID="${FINETUNE_ID}_smoke"
FINETUNE_EPOCHS="${SEGMENTANYTREE_FINETUNE_EPOCHS:-35}"
TRAIN_PARTITION="${SEGMENTANYTREE_OVERNIGHT_TRAIN_PARTITION:-gpu-l40s}"
PRETRAINED_PREDICTIONS="$PROJECT_ROOT/data/predictions/segmentanytree/for_instance_variants/$PRETRAINED_ID"
PRETRAINED_RUN_METADATA="$PROJECT_ROOT/results/metadata/segmentanytree_for_instance/variant_runs/$PRETRAINED_ID"
PRETRAINED_METRICS="$PROJECT_ROOT/results/metadata/segmentanytree_for_instance/variants/$PRETRAINED_ID"
PRETRAINED_TABLES="$PROJECT_ROOT/results/tables/segmentanytree_for_instance/variants/$PRETRAINED_ID"
FINETUNE_VALIDATION_METRICS="$PROJECT_ROOT/results/metadata/segmentanytree_for_instance/trained_validation/$FINETUNE_ID"
FINETUNE_TEST_METRICS="$PROJECT_ROOT/results/metadata/segmentanytree_for_instance/trained_test/$FINETUNE_ID"
SUMMARY="$PROJECT_ROOT/results/tables/segmentanytree_for_instance/three_variations/three_variations_$STAMP.csv"
STATE_FILE="$HOME/fastscratch/segmentanytree_three_variation_$STAMP.env"
LATEST_STATE="$HOME/fastscratch/segmentanytree_three_variation_latest.env"
FULL_MANIFEST="$PROJECT_ROOT/results/metadata/segmentanytree_for_instance/training_splits/full_split_manifest.json"
FULL_DATA="$HOME/fastscratch/segmentanytree_for_instance_training/full/treeinsfused/raw"
MIN_FREE_BYTES="${SEGMENTANYTREE_OVERNIGHT_MIN_FREE_BYTES:-85899345920}"
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

if [[ "${SEGMENTANYTREE_THREE_VARIATION_CONFIRMED:-0}" != "1" ]]; then
  echo "Refusing submission. Set SEGMENTANYTREE_THREE_VARIATION_CONFIRMED=1 after reviewing this workflow." >&2
  exit 2
fi
if [[ ! "$FINETUNE_EPOCHS" =~ ^[3-9][0-9]$|^[1-9][0-9]{2,}$ ]]; then
  echo "SEGMENTANYTREE_FINETUNE_EPOCHS must be at least 30." >&2
  exit 2
fi
if [[ ! "$TRAIN_PARTITION" =~ ^[A-Za-z0-9_-]+$ ]]; then
  echo "Invalid training partition: $TRAIN_PARTITION" >&2
  exit 2
fi

cd "$PROJECT_ROOT"
mkdir -p logs/segmentanytree_for_instance "$(dirname "$SUMMARY")"
test -f "$HOME/scratch/containers/segment-any-tree_latest.sif"
test -d "$HOME/fastscratch/venvs/treebench"
test -f "$FULL_MANIFEST"
test -d "$FULL_DATA"
test -f "$RETRAINED_CHECKPOINT"
test -d "$RETRAINED_METRICS"
test "$(find "$RETRAINED_METRICS" -type f -name '*.json' | wc -l)" -eq 11
test "$(sha256sum "$RETRAINED_CHECKPOINT" | awk '{print $1}')" = "$RETRAINED_SHA256"
test "$(git -C external/SegmentAnyTree rev-parse HEAD)" = "a3561ed8447bbb7938f059ba65a3e9c97d6e2ee9"
if [[ ! "$MIN_FREE_BYTES" =~ ^[1-9][0-9]*$ ]]; then
  echo "SEGMENTANYTREE_OVERNIGHT_MIN_FREE_BYTES must be a positive integer." >&2
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
  echo "Barkla repository is not clean; refusing an overnight benchmark run." >&2
  exit 2
fi
python - "$FULL_MANIFEST" <<'PY'
import json
import sys

manifest = json.load(open(sys.argv[1], encoding="utf-8"))
counts = manifest["selected_role_counts"]
actual = (counts.get("train", 0), counts.get("val", 0), counts.get("held_out_test", 0))
if actual != (16, 5, 0):
    raise SystemExit(f"Unexpected full split counts: {actual}")
if manifest.get("test_data_converted") is not False:
    raise SystemExit("Training manifest does not prove test isolation")
PY
for target in "$PRETRAINED_PREDICTIONS" "$PRETRAINED_METRICS" \
  "$FINETUNE_VALIDATION_METRICS" "$FINETUNE_TEST_METRICS"; do
  if [[ -e "$target" ]]; then
    echo "Target already exists: $target" >&2
    exit 2
  fi
done

checkpoint_job=$(sbatch --parsable \
  --export="ALL,SEGMENTANYTREE_RELEASED_CHECKPOINT_DIR=$RELEASED_DIR,SEGMENTANYTREE_RELEASED_CHECKPOINT_SHA256=$RELEASED_SHA256" \
  methods/segmentanytree/slurm/training/prepare_segmentanytree_released_checkpoint.sbatch)
submitted_jobs+=("$checkpoint_job")

pretrained_pilot_infer=$(sbatch --parsable \
  --array=0 \
  --dependency="afterok:$checkpoint_job" \
  --export="ALL,SEGMENTANYTREE_EXECUTE=1,SEGMENTANYTREE_CHECKPOINT_DIR=$RELEASED_DIR,SEGMENTANYTREE_RUN_TYPE=published_pretrained_inference,SEGMENTANYTREE_POINTWISE_OUTPUT_ROOT=$PRETRAINED_PREDICTIONS,SEGMENTANYTREE_POINTWISE_RUN_METADATA_ROOT=$PRETRAINED_RUN_METADATA" \
  methods/segmentanytree/slurm/inference/run_segmentanytree_for_instance_paper_test_array.sbatch)
submitted_jobs+=("$pretrained_pilot_infer")
pretrained_pilot_eval=$(sbatch --parsable \
  --array=0 \
  --dependency="afterok:$pretrained_pilot_infer" \
  --export="ALL,SEGMENTANYTREE_POINTWISE_OUTPUT_ROOT=$PRETRAINED_PREDICTIONS,SEGMENTANYTREE_POINTWISE_EVALUATION_ROOT=$PRETRAINED_METRICS,SEGMENTANYTREE_POINTWISE_TABLE_ROOT=$PRETRAINED_TABLES" \
  methods/segmentanytree/slurm/evaluation/evaluate_segmentanytree_paper_test_array.sbatch)
submitted_jobs+=("$pretrained_pilot_eval")
pretrained_pilot_gate=$(sbatch --parsable \
  --dependency="afterok:$pretrained_pilot_eval" \
  --export="ALL,SEGMENTANYTREE_VARIANT_LABEL=published_pretrained_pilot,SEGMENTANYTREE_VARIANT_METRICS_ROOT=$PRETRAINED_METRICS,SEGMENTANYTREE_VARIANT_EXPECTED_PLOTS=1,SEGMENTANYTREE_VARIANT_EXPECTED_SPLIT=test,SEGMENTANYTREE_VARIANT_REQUIRE_PREDICTIONS=1,SEGMENTANYTREE_VARIANT_SUMMARY=${PRETRAINED_TABLES}/pilot_summary.csv" \
  methods/segmentanytree/slurm/evaluation/validate_segmentanytree_variant.sbatch)
submitted_jobs+=("$pretrained_pilot_gate")
pretrained_rest_infer=$(sbatch --parsable \
  --array=1-10%2 \
  --dependency="afterok:$pretrained_pilot_gate" \
  --export="ALL,SEGMENTANYTREE_EXECUTE=1,SEGMENTANYTREE_CHECKPOINT_DIR=$RELEASED_DIR,SEGMENTANYTREE_RUN_TYPE=published_pretrained_inference,SEGMENTANYTREE_POINTWISE_OUTPUT_ROOT=$PRETRAINED_PREDICTIONS,SEGMENTANYTREE_POINTWISE_RUN_METADATA_ROOT=$PRETRAINED_RUN_METADATA" \
  methods/segmentanytree/slurm/inference/run_segmentanytree_for_instance_paper_test_array.sbatch)
submitted_jobs+=("$pretrained_rest_infer")
pretrained_rest_eval=$(sbatch --parsable \
  --array=1-10%4 \
  --dependency="afterok:$pretrained_rest_infer" \
  --export="ALL,SEGMENTANYTREE_POINTWISE_OUTPUT_ROOT=$PRETRAINED_PREDICTIONS,SEGMENTANYTREE_POINTWISE_EVALUATION_ROOT=$PRETRAINED_METRICS,SEGMENTANYTREE_POINTWISE_TABLE_ROOT=$PRETRAINED_TABLES" \
  methods/segmentanytree/slurm/evaluation/evaluate_segmentanytree_paper_test_array.sbatch)
submitted_jobs+=("$pretrained_rest_eval")
pretrained_final_gate=$(sbatch --parsable \
  --dependency="afterok:$pretrained_rest_eval" \
  --export="ALL,SEGMENTANYTREE_VARIANT_LABEL=published_pretrained,SEGMENTANYTREE_VARIANT_METRICS_ROOT=$PRETRAINED_METRICS,SEGMENTANYTREE_VARIANT_EXPECTED_PLOTS=11,SEGMENTANYTREE_VARIANT_EXPECTED_SPLIT=test,SEGMENTANYTREE_VARIANT_REQUIRE_PREDICTIONS=1,SEGMENTANYTREE_VARIANT_SUMMARY=${PRETRAINED_TABLES}/final_summary.csv" \
  methods/segmentanytree/slurm/evaluation/validate_segmentanytree_variant.sbatch)
submitted_jobs+=("$pretrained_final_gate")

finetune_smoke_job=$(sbatch --parsable \
  --partition="$TRAIN_PARTITION" --time=02:00:00 --cpus-per-task=16 --mem=64G \
  --dependency="afterok:$checkpoint_job" \
  --export="ALL,SEGMENTANYTREE_EXECUTE=1,SEGMENTANYTREE_FULL_TRAIN_CONFIRMED=1,SEGMENTANYTREE_TRAIN_PROFILE=full,SEGMENTANYTREE_TRAINING_RUN_ID=$FINETUNE_SMOKE_ID,SEGMENTANYTREE_TRAIN_EPOCHS=1,SEGMENTANYTREE_TRAIN_BATCH_SIZE=8,SEGMENTANYTREE_TRAIN_BASE_LR=0.0001,SEGMENTANYTREE_MEANSHIFT_JOBS=1,SEGMENTANYTREE_OMP_NUM_THREADS=1,SEGMENTANYTREE_STALL_TIMEOUT_SECONDS=2400,SEGMENTANYTREE_PRETRAINED_CHECKPOINT=$RELEASED_CHECKPOINT,SEGMENTANYTREE_PRETRAINED_CHECKPOINT_SHA256=$RELEASED_SHA256,SEGMENTANYTREE_PRETRAINED_WEIGHT_NAME=latest" \
  methods/segmentanytree/slurm/training/train_segmentanytree_for_instance_full.sbatch)
submitted_jobs+=("$finetune_smoke_job")
finetune_train_job=$(sbatch --parsable \
  --partition="$TRAIN_PARTITION" --time=12:00:00 --cpus-per-task=16 --mem=64G \
  --dependency="afterok:$finetune_smoke_job" \
  --export="ALL,SEGMENTANYTREE_EXECUTE=1,SEGMENTANYTREE_FULL_TRAIN_CONFIRMED=1,SEGMENTANYTREE_TRAIN_PROFILE=full,SEGMENTANYTREE_TRAINING_RUN_ID=$FINETUNE_ID,SEGMENTANYTREE_TRAIN_EPOCHS=$FINETUNE_EPOCHS,SEGMENTANYTREE_TRAIN_BATCH_SIZE=8,SEGMENTANYTREE_TRAIN_BASE_LR=0.0001,SEGMENTANYTREE_MEANSHIFT_JOBS=1,SEGMENTANYTREE_OMP_NUM_THREADS=1,SEGMENTANYTREE_STALL_TIMEOUT_SECONDS=2400,SEGMENTANYTREE_PRETRAINED_CHECKPOINT=$RELEASED_CHECKPOINT,SEGMENTANYTREE_PRETRAINED_CHECKPOINT_SHA256=$RELEASED_SHA256,SEGMENTANYTREE_PRETRAINED_WEIGHT_NAME=latest" \
  methods/segmentanytree/slurm/training/train_segmentanytree_for_instance_full.sbatch)
submitted_jobs+=("$finetune_train_job")
finetune_validation_infer=$(sbatch --parsable \
  --array=0-4%2 \
  --dependency="afterok:$finetune_train_job" \
  --export="ALL,SEGMENTANYTREE_EXECUTE=1,SEGMENTANYTREE_TRAIN_PROFILE=full,SEGMENTANYTREE_TRAINING_RUN_ID=$FINETUNE_ID" \
  methods/segmentanytree/slurm/inference/run_segmentanytree_for_instance_trained_validation.sbatch)
submitted_jobs+=("$finetune_validation_infer")
finetune_validation_eval=$(sbatch --parsable \
  --array=0-4%4 \
  --dependency="afterok:$finetune_validation_infer" \
  --export="ALL,SEGMENTANYTREE_TRAIN_PROFILE=full,SEGMENTANYTREE_TRAINING_RUN_ID=$FINETUNE_ID" \
  methods/segmentanytree/slurm/evaluation/evaluate_segmentanytree_for_instance_trained_validation.sbatch)
submitted_jobs+=("$finetune_validation_eval")
finetune_validation_gate=$(sbatch --parsable \
  --dependency="afterok:$finetune_validation_eval" \
  --export="ALL,SEGMENTANYTREE_VARIANT_LABEL=fine_tuned_on_dev_validation,SEGMENTANYTREE_VARIANT_METRICS_ROOT=$FINETUNE_VALIDATION_METRICS,SEGMENTANYTREE_VARIANT_EXPECTED_PLOTS=5,SEGMENTANYTREE_VARIANT_EXPECTED_SPLIT=dev,SEGMENTANYTREE_VARIANT_REQUIRE_PREDICTIONS=1,SEGMENTANYTREE_VARIANT_SUMMARY=$PROJECT_ROOT/results/tables/segmentanytree_for_instance/trained_validation/$FINETUNE_ID/validation_summary.csv" \
  methods/segmentanytree/slurm/evaluation/validate_segmentanytree_variant.sbatch)
submitted_jobs+=("$finetune_validation_gate")
finetune_test_infer=$(sbatch --parsable \
  --array=0-10%2 \
  --dependency="afterok:$finetune_validation_gate" \
  --export="ALL,SEGMENTANYTREE_FINAL_TEST_CONFIRMED=1,SEGMENTANYTREE_TRAINING_RUN_ID=$FINETUNE_ID" \
  methods/segmentanytree/slurm/inference/run_segmentanytree_for_instance_test_from_checkpoint.sbatch)
submitted_jobs+=("$finetune_test_infer")
finetune_test_eval=$(sbatch --parsable \
  --array=0-10%4 \
  --dependency="afterok:$finetune_test_infer" \
  --export="ALL,SEGMENTANYTREE_TRAINING_RUN_ID=$FINETUNE_ID" \
  methods/segmentanytree/slurm/evaluation/evaluate_segmentanytree_for_instance_test_from_checkpoint.sbatch)
submitted_jobs+=("$finetune_test_eval")
finetune_final_gate=$(sbatch --parsable \
  --dependency="afterok:$finetune_test_eval" \
  --export="ALL,SEGMENTANYTREE_VARIANT_LABEL=fine_tuned_on_dev,SEGMENTANYTREE_VARIANT_METRICS_ROOT=$FINETUNE_TEST_METRICS,SEGMENTANYTREE_VARIANT_EXPECTED_PLOTS=11,SEGMENTANYTREE_VARIANT_EXPECTED_SPLIT=test,SEGMENTANYTREE_VARIANT_REQUIRE_PREDICTIONS=1,SEGMENTANYTREE_VARIANT_SUMMARY=$PROJECT_ROOT/results/tables/segmentanytree_for_instance/trained_test/$FINETUNE_ID/final_summary.csv" \
  methods/segmentanytree/slurm/evaluation/validate_segmentanytree_variant.sbatch)
submitted_jobs+=("$finetune_final_gate")

summary_job=$(sbatch --parsable \
  --dependency="afterok:$pretrained_final_gate:$finetune_final_gate" \
  --export="ALL,SEGMENTANYTREE_PRETRAINED_METRICS_ROOT=$PRETRAINED_METRICS,SEGMENTANYTREE_RETRAINED_METRICS_ROOT=$RETRAINED_METRICS,SEGMENTANYTREE_FINETUNED_METRICS_ROOT=$FINETUNE_TEST_METRICS,SEGMENTANYTREE_THREE_VARIATION_SUMMARY=$SUMMARY" \
  methods/segmentanytree/slurm/evaluation/summarise_segmentanytree_three_variations.sbatch)
submitted_jobs+=("$summary_job")

submitted_epoch=$(date +%s)
expected_finish_epoch=$((submitted_epoch + 8 * 3600))
{
  printf 'SAT_THREE_STATE_FILE=%q\n' "$STATE_FILE"
  printf 'SAT_THREE_SUBMITTED_EPOCH=%q\n' "$submitted_epoch"
  printf 'SAT_THREE_EXPECTED_FINISH_EPOCH=%q\n' "$expected_finish_epoch"
  printf 'SAT_THREE_FINETUNE_EPOCHS=%q\n' "$FINETUNE_EPOCHS"
  printf 'SAT_THREE_PRETRAINED_ID=%q\n' "$PRETRAINED_ID"
  printf 'SAT_THREE_FINETUNE_ID=%q\n' "$FINETUNE_ID"
  printf 'SAT_THREE_PRETRAINED_METRICS=%q\n' "$PRETRAINED_METRICS"
  printf 'SAT_THREE_FINETUNE_VALIDATION_METRICS=%q\n' "$FINETUNE_VALIDATION_METRICS"
  printf 'SAT_THREE_FINETUNE_TEST_METRICS=%q\n' "$FINETUNE_TEST_METRICS"
  printf 'SAT_THREE_SUMMARY=%q\n' "$SUMMARY"
  printf 'SAT_THREE_CHECKPOINT_JOB=%q\n' "$checkpoint_job"
  printf 'SAT_THREE_PRETRAINED_PILOT_INFER=%q\n' "$pretrained_pilot_infer"
  printf 'SAT_THREE_PRETRAINED_PILOT_EVAL=%q\n' "$pretrained_pilot_eval"
  printf 'SAT_THREE_PRETRAINED_PILOT_GATE=%q\n' "$pretrained_pilot_gate"
  printf 'SAT_THREE_PRETRAINED_REST_INFER=%q\n' "$pretrained_rest_infer"
  printf 'SAT_THREE_PRETRAINED_REST_EVAL=%q\n' "$pretrained_rest_eval"
  printf 'SAT_THREE_PRETRAINED_FINAL_GATE=%q\n' "$pretrained_final_gate"
  printf 'SAT_THREE_FINETUNE_SMOKE_JOB=%q\n' "$finetune_smoke_job"
  printf 'SAT_THREE_FINETUNE_TRAIN_JOB=%q\n' "$finetune_train_job"
  printf 'SAT_THREE_FINETUNE_VALIDATION_INFER=%q\n' "$finetune_validation_infer"
  printf 'SAT_THREE_FINETUNE_VALIDATION_EVAL=%q\n' "$finetune_validation_eval"
  printf 'SAT_THREE_FINETUNE_VALIDATION_GATE=%q\n' "$finetune_validation_gate"
  printf 'SAT_THREE_FINETUNE_TEST_INFER=%q\n' "$finetune_test_infer"
  printf 'SAT_THREE_FINETUNE_TEST_EVAL=%q\n' "$finetune_test_eval"
  printf 'SAT_THREE_FINETUNE_FINAL_GATE=%q\n' "$finetune_final_gate"
  printf 'SAT_THREE_SUMMARY_JOB=%q\n' "$summary_job"
} > "$STATE_FILE"
ln -sfn "$STATE_FILE" "$LATEST_STATE"
trap - ERR

echo "submitted: pretrained=$PRETRAINED_ID fine_tuned=$FINETUNE_ID"
echo "summary_job=$summary_job expected_finish=$(date -d "@$expected_finish_epoch" '+%F %T') plus queue delay"
echo "monitor: bash methods/segmentanytree/slurm/monitor_three_variation_overnight.sh --follow"
