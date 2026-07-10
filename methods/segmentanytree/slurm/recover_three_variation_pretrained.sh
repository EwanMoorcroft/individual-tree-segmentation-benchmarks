#!/bin/bash

set -euo pipefail

PROJECT_ROOT="${SEGMENTANYTREE_PROJECT_ROOT:-$HOME/scratch/tree-seg-benchmark}"
STATE_FILE="${1:-$HOME/fastscratch/segmentanytree_three_variation_latest.env}"
STATE_FILE=$(readlink -f "$STATE_FILE")
test -f "$STATE_FILE"
source "$STATE_FILE"

if [[ "${SEGMENTANYTREE_RECOVER_PRETRAINED_CONFIRMED:-0}" != "1" ]]; then
  echo "Set SEGMENTANYTREE_RECOVER_PRETRAINED_CONFIRMED=1 to recover the failed pretrained branch." >&2
  exit 2
fi
if [[ ! "$SAT_THREE_PRETRAINED_ID" =~ ^sat_published_pretrained_aligned_[0-9]{8}_[0-9]{6}$ ]]; then
  echo "Unexpected pretrained run ID: $SAT_THREE_PRETRAINED_ID" >&2
  exit 2
fi
pretrained_metric_count=0
if [[ -d "$SAT_THREE_PRETRAINED_METRICS" ]]; then
  pretrained_metric_count=$(find "$SAT_THREE_PRETRAINED_METRICS" -type f -name '*.json' | wc -l)
fi
if [[ "$pretrained_metric_count" -eq 11 ]]; then
  echo "Complete pretrained metrics already exist; recovery is unnecessary." >&2
  exit 2
fi
if sacct -X -n -j "$SAT_THREE_FINETUNE_SMOKE_JOB,$SAT_THREE_FINETUNE_TRAIN_JOB,$SAT_THREE_FINETUNE_VALIDATION_INFER,$SAT_THREE_FINETUNE_VALIDATION_EVAL,$SAT_THREE_FINETUNE_VALIDATION_GATE,$SAT_THREE_FINETUNE_TEST_INFER,$SAT_THREE_FINETUNE_TEST_EVAL,$SAT_THREE_FINETUNE_FINAL_GATE" -o State 2>/dev/null | grep -Eq 'FAILED|TIMEOUT|OUT_OF_MEMORY|NODE_FAIL'; then
  echo "The fine-tune branch has an independent failure; refusing pretrained-only recovery." >&2
  exit 2
fi

cd "$PROJECT_ROOT"
PRETRAINED_PREDICTIONS="$PROJECT_ROOT/data/predictions/segmentanytree/for_instance_variants/$SAT_THREE_PRETRAINED_ID"
PRETRAINED_RUN_METADATA="$PROJECT_ROOT/results/metadata/segmentanytree_for_instance/variant_runs/$SAT_THREE_PRETRAINED_ID"
PRETRAINED_TABLES="$PROJECT_ROOT/results/tables/segmentanytree_for_instance/variants/$SAT_THREE_PRETRAINED_ID"
RECOVERY_STAMP=$(date +%Y%m%d_%H%M%S)
ARCHIVE_ROOT="$HOME/fastscratch/segmentanytree_recovery_archive/$SAT_THREE_PRETRAINED_ID/$RECOVERY_STAMP"

scancel \
  "$SAT_THREE_PRETRAINED_PILOT_EVAL" \
  "$SAT_THREE_PRETRAINED_PILOT_GATE" \
  "$SAT_THREE_PRETRAINED_REST_INFER" \
  "$SAT_THREE_PRETRAINED_REST_EVAL" \
  "$SAT_THREE_PRETRAINED_FINAL_GATE" \
  "$SAT_THREE_SUMMARY_JOB" 2>/dev/null || true

mkdir -p "$ARCHIVE_ROOT"
for target in \
  "$PRETRAINED_PREDICTIONS" \
  "$PRETRAINED_RUN_METADATA" \
  "$SAT_THREE_PRETRAINED_METRICS" \
  "$PRETRAINED_TABLES"; do
  if [[ -e "$target" ]]; then
    relative_target=${target#"$PROJECT_ROOT"/}
    archive_target="$ARCHIVE_ROOT/$relative_target"
    mkdir -p "$(dirname "$archive_target")"
    mv "$target" "$archive_target"
  fi
done
printf 'source_state=%s\nrun_id=%s\narchived_at=%s\n' \
  "$STATE_FILE" "$SAT_THREE_PRETRAINED_ID" "$RECOVERY_STAMP" \
  > "$ARCHIVE_ROOT/recovery_manifest.txt"

unset SEGMENTANYTREE_CHECKPOINT_DIR
new_jobs=()
cancel_new_jobs() {
  local status=$?
  if ((${#new_jobs[@]})); then
    scancel "${new_jobs[@]}" 2>/dev/null || true
  fi
  echo "Recovery submission failed; cancelled only the replacement jobs." >&2
  exit "$status"
}
trap cancel_new_jobs ERR

pilot_infer=$(sbatch --parsable \
  --array=0 \
  --export="ALL,SEGMENTANYTREE_EXECUTE=1,SEGMENTANYTREE_RUN_TYPE=published_pretrained_inference,SEGMENTANYTREE_POINTWISE_OUTPUT_ROOT=$PRETRAINED_PREDICTIONS,SEGMENTANYTREE_POINTWISE_RUN_METADATA_ROOT=$PRETRAINED_RUN_METADATA" \
  methods/segmentanytree/slurm/inference/run_segmentanytree_for_instance_paper_test_array.sbatch)
new_jobs+=("$pilot_infer")
pilot_eval=$(sbatch --parsable \
  --array=0 \
  --dependency="afterok:$pilot_infer" \
  --export="ALL,SEGMENTANYTREE_POINTWISE_OUTPUT_ROOT=$PRETRAINED_PREDICTIONS,SEGMENTANYTREE_POINTWISE_EVALUATION_ROOT=$SAT_THREE_PRETRAINED_METRICS,SEGMENTANYTREE_POINTWISE_TABLE_ROOT=$PRETRAINED_TABLES" \
  methods/segmentanytree/slurm/evaluation/evaluate_segmentanytree_paper_test_array.sbatch)
new_jobs+=("$pilot_eval")
pilot_gate=$(sbatch --parsable \
  --dependency="afterok:$pilot_eval" \
  --export="ALL,SEGMENTANYTREE_VARIANT_LABEL=published_pretrained_pilot,SEGMENTANYTREE_VARIANT_METRICS_ROOT=$SAT_THREE_PRETRAINED_METRICS,SEGMENTANYTREE_VARIANT_EXPECTED_PLOTS=1,SEGMENTANYTREE_VARIANT_EXPECTED_SPLIT=test,SEGMENTANYTREE_VARIANT_REQUIRE_PREDICTIONS=1,SEGMENTANYTREE_VARIANT_SUMMARY=${PRETRAINED_TABLES}/pilot_summary.csv" \
  methods/segmentanytree/slurm/evaluation/validate_segmentanytree_variant.sbatch)
new_jobs+=("$pilot_gate")
rest_infer=$(sbatch --parsable \
  --array=1-10%2 \
  --dependency="afterok:$pilot_gate" \
  --export="ALL,SEGMENTANYTREE_EXECUTE=1,SEGMENTANYTREE_RUN_TYPE=published_pretrained_inference,SEGMENTANYTREE_POINTWISE_OUTPUT_ROOT=$PRETRAINED_PREDICTIONS,SEGMENTANYTREE_POINTWISE_RUN_METADATA_ROOT=$PRETRAINED_RUN_METADATA" \
  methods/segmentanytree/slurm/inference/run_segmentanytree_for_instance_paper_test_array.sbatch)
new_jobs+=("$rest_infer")
rest_eval=$(sbatch --parsable \
  --array=1-10%4 \
  --dependency="afterok:$rest_infer" \
  --export="ALL,SEGMENTANYTREE_POINTWISE_OUTPUT_ROOT=$PRETRAINED_PREDICTIONS,SEGMENTANYTREE_POINTWISE_EVALUATION_ROOT=$SAT_THREE_PRETRAINED_METRICS,SEGMENTANYTREE_POINTWISE_TABLE_ROOT=$PRETRAINED_TABLES" \
  methods/segmentanytree/slurm/evaluation/evaluate_segmentanytree_paper_test_array.sbatch)
new_jobs+=("$rest_eval")
final_gate=$(sbatch --parsable \
  --dependency="afterok:$rest_eval" \
  --export="ALL,SEGMENTANYTREE_VARIANT_LABEL=published_pretrained,SEGMENTANYTREE_VARIANT_METRICS_ROOT=$SAT_THREE_PRETRAINED_METRICS,SEGMENTANYTREE_VARIANT_EXPECTED_PLOTS=11,SEGMENTANYTREE_VARIANT_EXPECTED_SPLIT=test,SEGMENTANYTREE_VARIANT_REQUIRE_PREDICTIONS=1,SEGMENTANYTREE_VARIANT_SUMMARY=${PRETRAINED_TABLES}/final_summary.csv" \
  methods/segmentanytree/slurm/evaluation/validate_segmentanytree_variant.sbatch)
new_jobs+=("$final_gate")
summary_job=$(sbatch --parsable \
  --dependency="afterok:$final_gate:$SAT_THREE_FINETUNE_FINAL_GATE" \
  --export="ALL,SEGMENTANYTREE_PRETRAINED_METRICS_ROOT=$SAT_THREE_PRETRAINED_METRICS,SEGMENTANYTREE_FINETUNED_METRICS_ROOT=$SAT_THREE_FINETUNE_TEST_METRICS,SEGMENTANYTREE_PRETRAINED_FINETUNE_SUMMARY=$SAT_THREE_SUMMARY" \
  methods/segmentanytree/slurm/evaluation/summarise_segmentanytree_pretrained_finetune.sbatch)
new_jobs+=("$summary_job")

{
  printf 'SAT_THREE_PRETRAINED_PILOT_INFER=%q\n' "$pilot_infer"
  printf 'SAT_THREE_PRETRAINED_PILOT_EVAL=%q\n' "$pilot_eval"
  printf 'SAT_THREE_PRETRAINED_PILOT_GATE=%q\n' "$pilot_gate"
  printf 'SAT_THREE_PRETRAINED_REST_INFER=%q\n' "$rest_infer"
  printf 'SAT_THREE_PRETRAINED_REST_EVAL=%q\n' "$rest_eval"
  printf 'SAT_THREE_PRETRAINED_FINAL_GATE=%q\n' "$final_gate"
  printf 'SAT_THREE_SUMMARY_JOB=%q\n' "$summary_job"
} >> "$SAT_THREE_STATE_FILE"
trap - ERR

echo "pretrained branch resubmitted; prior partial outputs archived at $ARCHIVE_ROOT"
echo "monitor: bash methods/segmentanytree/slurm/monitor_pretrained_finetune_comparison.sh --follow"
