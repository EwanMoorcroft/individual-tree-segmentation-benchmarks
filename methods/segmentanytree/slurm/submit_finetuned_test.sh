#!/bin/bash

set -euo pipefail

if [[ "${SEGMENTANYTREE_FINETUNED_TEST_CONFIRMED:-0}" != "1" ]]; then
  echo "Set SEGMENTANYTREE_FINETUNED_TEST_CONFIRMED=1 after accepting the development-selected run." >&2
  exit 2
fi

PROJECT_ROOT="${SEGMENTANYTREE_PROJECT_ROOT:-$HOME/scratch/tree-seg-benchmark}"
RUN_ID="${SEGMENTANYTREE_TRAINING_RUN_ID:?Set SEGMENTANYTREE_TRAINING_RUN_ID to the accepted development run.}"
CHECKPOINT_DIR="$HOME/fastscratch/segmentanytree_for_instance_checkpoints/$RUN_ID/run"
CHECKPOINT="$CHECKPOINT_DIR/PointGroup-PAPER.pt"
DEVELOPMENT_FREEZE="$PROJECT_ROOT/results/metadata/segmentanytree_for_instance/finetune_freezes/$RUN_ID.json"
DEVELOPMENT_SUMMARY="$PROJECT_ROOT/results/tables/segmentanytree_for_instance/trained_validation/$RUN_ID/validation_summary.csv"
TRAINING_METADATA="$PROJECT_ROOT/results/metadata/segmentanytree_for_instance/training_runs/$RUN_ID.json"
PREDICTIONS="$PROJECT_ROOT/data/predictions/segmentanytree/for_instance_trained_test/$RUN_ID"
RUN_METADATA="$PROJECT_ROOT/results/metadata/segmentanytree_for_instance/trained_test_runs/$RUN_ID"
METRICS="$PROJECT_ROOT/results/metadata/segmentanytree_for_instance/trained_test/$RUN_ID"
TABLES="$PROJECT_ROOT/results/tables/segmentanytree_for_instance/trained_test/$RUN_ID"
SUMMARY="$TABLES/final_summary.csv"
FREEZE_MANIFEST="$PROJECT_ROOT/results/metadata/segmentanytree_for_instance/finetuned_test_freezes/$RUN_ID.json"
STATE_FILE="$HOME/fastscratch/segmentanytree_finetuned_test_${RUN_ID}.env"
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

if [[ ! "$RUN_ID" =~ ^segmentanytree_for-instance_fine_tuned_on_dev_[0-9]{8}_[0-9]{6}$ ]]; then
  echo "Unexpected fine-tuned run ID: $RUN_ID" >&2
  exit 2
fi

cd "$PROJECT_ROOT"
mkdir -p logs/segmentanytree_for_instance
test -f "$CHECKPOINT"
test -f "$DEVELOPMENT_FREEZE"
test -f "$DEVELOPMENT_SUMMARY"
test -f "$TRAINING_METADATA"
test "$(git -C external/SegmentAnyTree rev-parse HEAD)" = "a3561ed8447bbb7938f059ba65a3e9c97d6e2ee9"
if [[ -n "$(git status --porcelain)" ]]; then
  echo "Barkla repository is not clean; refusing held-out evaluation." >&2
  exit 2
fi
for target in "$PREDICTIONS" "$RUN_METADATA" "$METRICS" "$TABLES" "$FREEZE_MANIFEST"; do
  if [[ -e "$target" ]]; then
    echo "Held-out target already exists; refusing repeat evaluation: $target" >&2
    exit 2
  fi
done

module purge
module load miniforge3/25.3.0-python3.12.10
source "$HOME/fastscratch/venvs/treebench/bin/activate"
python methods/segmentanytree/scripts/evaluation/prepare_finetuned_test_freeze.py \
  --run-id "$RUN_ID" \
  --development-freeze "$DEVELOPMENT_FREEZE" \
  --development-summary "$DEVELOPMENT_SUMMARY" \
  --training-metadata "$TRAINING_METADATA" \
  --checkpoint "$CHECKPOINT" \
  --output "$FREEZE_MANIFEST"

inference_job=$(sbatch --parsable \
  --array=0-10%2 \
  --export="ALL,SEGMENTANYTREE_FINAL_TEST_CONFIRMED=1,SEGMENTANYTREE_TRAINING_RUN_ID=$RUN_ID,SEGMENTANYTREE_CHECKPOINT_DIR=$CHECKPOINT_DIR" \
  methods/segmentanytree/slurm/inference/run_segmentanytree_for_instance_test_from_checkpoint.sbatch)
submitted_jobs+=("$inference_job")

evaluation_job=$(sbatch --parsable \
  --array=0-10%4 \
  --dependency="afterok:$inference_job" \
  --export="ALL,SEGMENTANYTREE_TRAINING_RUN_ID=$RUN_ID" \
  methods/segmentanytree/slurm/evaluation/evaluate_segmentanytree_for_instance_test_from_checkpoint.sbatch)
submitted_jobs+=("$evaluation_job")

gate_job=$(sbatch --parsable \
  --dependency="afterok:$evaluation_job" \
  --export="ALL,SEGMENTANYTREE_VARIANT_LABEL=fine_tuned_on_dev,SEGMENTANYTREE_VARIANT_METRICS_ROOT=$METRICS,SEGMENTANYTREE_VARIANT_EXPECTED_PLOTS=11,SEGMENTANYTREE_VARIANT_EXPECTED_SPLIT=test,SEGMENTANYTREE_VARIANT_REQUIRE_PREDICTIONS=1,SEGMENTANYTREE_VARIANT_SUMMARY=$SUMMARY" \
  methods/segmentanytree/slurm/evaluation/validate_segmentanytree_variant.sbatch)
submitted_jobs+=("$gate_job")

{
  printf 'SAT_FINETUNED_TEST_RUN_ID=%q\n' "$RUN_ID"
  printf 'SAT_FINETUNED_TEST_INFERENCE_JOB=%q\n' "$inference_job"
  printf 'SAT_FINETUNED_TEST_EVALUATION_JOB=%q\n' "$evaluation_job"
  printf 'SAT_FINETUNED_TEST_GATE_JOB=%q\n' "$gate_job"
  printf 'SAT_FINETUNED_TEST_METRICS=%q\n' "$METRICS"
  printf 'SAT_FINETUNED_TEST_SUMMARY=%q\n' "$SUMMARY"
  printf 'SAT_FINETUNED_TEST_FREEZE=%q\n' "$FREEZE_MANIFEST"
} > "$STATE_FILE"
trap - ERR

echo "run_id=$RUN_ID"
echo "inference_job=$inference_job evaluation_job=$evaluation_job gate_job=$gate_job"
echo "state_file=$STATE_FILE"
echo "freeze_manifest=$FREEZE_MANIFEST"
echo "summary=$SUMMARY"
echo "No training job was submitted. Repeated test submission is refused."
