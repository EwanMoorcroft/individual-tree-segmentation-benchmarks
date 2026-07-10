#!/bin/bash

set -euo pipefail

if [[ "${SEGMENTANYTREE_PUBLISHED_PRETRAINED_TEST_CONFIRMED:-0}" != "1" ]]; then
  echo "Set SEGMENTANYTREE_PUBLISHED_PRETRAINED_TEST_CONFIRMED=1 after manual smoke review." >&2
  exit 2
fi
if [[ "${SEGMENTANYTREE_ACCEPT_UNRESOLVED_TRAINING_MANIFEST:-0}" != "1" ]]; then
  echo "Set SEGMENTANYTREE_ACCEPT_UNRESOLVED_TRAINING_MANIFEST=1 to record the accepted provenance limitation." >&2
  exit 2
fi

PROJECT_ROOT="${SEGMENTANYTREE_PROJECT_ROOT:-$HOME/scratch/tree-seg-benchmark}"
SMOKE_EVIDENCE="${SEGMENTANYTREE_DEV_SMOKE_EVIDENCE:?Set SEGMENTANYTREE_DEV_SMOKE_EVIDENCE to the reviewed smoke JSON.}"
SMOKE_EVIDENCE=$(realpath "$SMOKE_EVIDENCE")
RUN_ID=$(basename "$SMOKE_EVIDENCE" .json)
CHECKPOINT_BUNDLE="$HOME/fastscratch/segmentanytree_pretrained/released_model_bundle"
PREDICTIONS="$PROJECT_ROOT/data/predictions/segmentanytree/for_instance_variants/$RUN_ID/held_out_test"
RUN_METADATA="$PROJECT_ROOT/results/metadata/segmentanytree_for_instance/variant_runs/$RUN_ID/held_out_test"
METRICS="$PROJECT_ROOT/results/metadata/segmentanytree_for_instance/variants/$RUN_ID/held_out_test"
TABLES="$PROJECT_ROOT/results/tables/segmentanytree_for_instance/variants/$RUN_ID/held_out_test"
FREEZE_MANIFEST="$PROJECT_ROOT/results/metadata/segmentanytree_for_instance/test_freezes/$RUN_ID.json"
FINAL_SUMMARY="$TABLES/final_summary.csv"
STATE_FILE="$HOME/fastscratch/segmentanytree_published_pretrained_test_${RUN_ID}.env"
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

if [[ ! "$RUN_ID" =~ ^segmentanytree_for-instance_published_pretrained_[0-9]{8}_[0-9]{6}$ ]]; then
  echo "Unexpected smoke run ID: $RUN_ID" >&2
  exit 2
fi

cd "$PROJECT_ROOT"
mkdir -p logs/segmentanytree_for_instance
test -f "$SMOKE_EVIDENCE"
test -f "$CHECKPOINT_BUNDLE/PointGroup-PAPER.pt"
test -s "$CHECKPOINT_BUNDLE/.hydra/overrides.yaml"
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
python methods/segmentanytree/scripts/evaluation/prepare_published_pretrained_test_freeze.py \
  --smoke-evidence "$SMOKE_EVIDENCE" \
  --checkpoint-bundle "$CHECKPOINT_BUNDLE" \
  --run-id "$RUN_ID" \
  --accept-unresolved-training-manifest \
  --output "$FREEZE_MANIFEST"

inference_job=$(sbatch --parsable \
  --array=0-10%2 \
  --export="ALL,SEGMENTANYTREE_EXECUTE=1,SEGMENTANYTREE_REQUIRED_SPLIT=test,SEGMENTANYTREE_RUN_TYPE=published_pretrained,SEGMENTANYTREE_POINTWISE_OUTPUT_ROOT=$PREDICTIONS,SEGMENTANYTREE_POINTWISE_RUN_METADATA_ROOT=$RUN_METADATA,SEGMENTANYTREE_CHECKPOINT_DIR=$CHECKPOINT_BUNDLE" \
  methods/segmentanytree/slurm/inference/run_segmentanytree_for_instance_paper_test_array.sbatch)
submitted_jobs+=("$inference_job")

evaluation_job=$(sbatch --parsable \
  --array=0-10%4 \
  --dependency="afterok:$inference_job" \
  --export="ALL,SEGMENTANYTREE_REQUIRED_SPLIT=test,SEGMENTANYTREE_POINTWISE_OUTPUT_ROOT=$PREDICTIONS,SEGMENTANYTREE_POINTWISE_EVALUATION_ROOT=$METRICS,SEGMENTANYTREE_POINTWISE_TABLE_ROOT=$TABLES" \
  methods/segmentanytree/slurm/evaluation/evaluate_segmentanytree_paper_test_array.sbatch)
submitted_jobs+=("$evaluation_job")

gate_job=$(sbatch --parsable \
  --dependency="afterok:$evaluation_job" \
  --export="ALL,SEGMENTANYTREE_VARIANT_LABEL=published_pretrained,SEGMENTANYTREE_VARIANT_METRICS_ROOT=$METRICS,SEGMENTANYTREE_VARIANT_EXPECTED_PLOTS=11,SEGMENTANYTREE_VARIANT_EXPECTED_SPLIT=test,SEGMENTANYTREE_VARIANT_REQUIRE_PREDICTIONS=1,SEGMENTANYTREE_VARIANT_SUMMARY=$FINAL_SUMMARY" \
  methods/segmentanytree/slurm/evaluation/validate_segmentanytree_variant.sbatch)
submitted_jobs+=("$gate_job")

submitted_epoch=$(date +%s)
{
  printf 'SAT_PRETRAINED_TEST_RUN_ID=%q\n' "$RUN_ID"
  printf 'SAT_PRETRAINED_TEST_SUBMITTED_EPOCH=%q\n' "$submitted_epoch"
  printf 'SAT_PRETRAINED_TEST_INFERENCE_JOB=%q\n' "$inference_job"
  printf 'SAT_PRETRAINED_TEST_EVALUATION_JOB=%q\n' "$evaluation_job"
  printf 'SAT_PRETRAINED_TEST_GATE_JOB=%q\n' "$gate_job"
  printf 'SAT_PRETRAINED_TEST_METRICS=%q\n' "$METRICS"
  printf 'SAT_PRETRAINED_TEST_SUMMARY=%q\n' "$FINAL_SUMMARY"
  printf 'SAT_PRETRAINED_TEST_FREEZE_MANIFEST=%q\n' "$FREEZE_MANIFEST"
} > "$STATE_FILE"
trap - ERR

echo "run_id=$RUN_ID"
echo "inference_job=$inference_job evaluation_job=$evaluation_job gate_job=$gate_job"
echo "state_file=$STATE_FILE"
echo "freeze_manifest=$FREEZE_MANIFEST"
echo "summary=$FINAL_SUMMARY"
echo "No training or fine-tuning job was submitted."
