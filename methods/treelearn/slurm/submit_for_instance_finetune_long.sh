#!/usr/bin/env bash
set -euo pipefail
test "${TREELEARN_FINETUNE_LONG_CONFIRMED:-0}" = 1 || {
  echo "Set TREELEARN_FINETUNE_LONG_CONFIRMED=1." >&2
  exit 2
}
PROJECT_ROOT="${TREELEARN_PROJECT_ROOT:-$HOME/scratch/tree-seg-benchmark}"
TREELEARN_ENV="${TREELEARN_ENV:-$HOME/fastscratch/venvs/treelearn}"
TREELEARN_REPO="${TREELEARN_REPO:-$HOME/fastscratch/external/TreeLearn}"
TREELEARN_DATASET_ROOT="${TREELEARN_DATASET_ROOT:-$HOME/data/datasets/for_instance/FORinstance_dataset}"
TREELEARN_LONG_INITIAL_CHECKPOINT="${TREELEARN_LONG_INITIAL_CHECKPOINT:-$HOME/fastscratch/treelearn_checkpoints/model_weights_finetuned.pth}"
DEV_RUN=treelearn_for-instance_published_pretrained_development_20260712_150030
TREELEARN_LONG_DEV_MANIFEST="${TREELEARN_LONG_DEV_MANIFEST:-$PROJECT_ROOT/results/metadata/treelearn_for_instance/development_runs/$DEV_RUN/development_manifest.json}"

cd "$PROJECT_ROOT"
mkdir -p logs/treelearn_for_instance
test -x "$TREELEARN_ENV/bin/python"
test -d "$TREELEARN_REPO"
test -f "$TREELEARN_LONG_DEV_MANIFEST"
python3 methods/treelearn/scripts/fetch_treelearn_clean_checkpoint.py \
  --output "$TREELEARN_LONG_INITIAL_CHECKPOINT"
test -f "$TREELEARN_LONG_INITIAL_CHECKPOINT"
test "$(md5sum "$TREELEARN_LONG_INITIAL_CHECKPOINT" | cut -d ' ' -f 1)" = \
  106a80de2991c5f23484a3f9d03e3b16
test "$(git -C "$TREELEARN_REPO" rev-parse HEAD)" = \
  fd240ce7caa4c444fe3418aca454dc578bc557d4
test -z "$(git status --porcelain)"
test -z "$(git -C "$TREELEARN_REPO" status --porcelain)"

COMMIT=$(git rev-parse HEAD)
STAMP="${TREELEARN_LONG_STAMP:-$(date -u +%Y%m%d_%H%M%S)}"
TREELEARN_LONG_RUN_ID="treelearn_for-instance_fine_tuned_on_dev_long_$STAMP"
TREELEARN_LONG_ROOT="$HOME/fastscratch/treelearn_finetune_long/$TREELEARN_LONG_RUN_ID"
TREELEARN_LONG_FREEZE="$TREELEARN_LONG_ROOT/long_finetune_freeze.json"
TREELEARN_LONG_SELECTION_FREEZE="$TREELEARN_LONG_ROOT/selection/selection_freeze.json"
TREELEARN_LONG_DURABLE_ROOT="$PROJECT_ROOT/results/metadata/treelearn_for_instance/long_finetune_runs/$TREELEARN_LONG_RUN_ID"
TREELEARN_LONG_DURABLE_CHECKPOINT="$PROJECT_ROOT/data/checkpoints/treelearn/$TREELEARN_LONG_RUN_ID/epoch_35.pth"
TREELEARN_LONG_RETENTION_MANIFEST="$TREELEARN_LONG_DURABLE_ROOT/selection_retention_manifest.json"
TREELEARN_LONG_SELECTED_FREEZE="$TREELEARN_LONG_DURABLE_ROOT/selected_checkpoint_freeze.json"
TREELEARN_LONG_METADATA_BASE="$PROJECT_ROOT/results/metadata/treelearn_for_instance/development_runs"
TREELEARN_LONG_TABLES_BASE="$PROJECT_ROOT/results/tables/treelearn_for_instance/development_runs"
STATE_FILE="$HOME/fastscratch/treelearn_finetune_long_${TREELEARN_LONG_RUN_ID}.env"
test ! -e "$TREELEARN_LONG_ROOT"
test ! -e "$TREELEARN_LONG_DURABLE_ROOT"
test ! -e "$TREELEARN_LONG_DURABLE_CHECKPOINT"
test ! -e "$STATE_FILE"

EXPORTS="ALL,TREELEARN_FINETUNE_LONG_CONFIRMED=1,TREELEARN_EXPECTED_BENCHMARK_COMMIT=$COMMIT,TREELEARN_ENV=$TREELEARN_ENV,TREELEARN_REPO=$TREELEARN_REPO,TREELEARN_DATASET_ROOT=$TREELEARN_DATASET_ROOT,TREELEARN_LONG_INITIAL_CHECKPOINT=$TREELEARN_LONG_INITIAL_CHECKPOINT,TREELEARN_DEV_MANIFEST_JSON=$TREELEARN_LONG_DEV_MANIFEST,TREELEARN_LONG_DEV_MANIFEST=$TREELEARN_LONG_DEV_MANIFEST,TREELEARN_LONG_CROPS_PER_PLOT=1500,TREELEARN_LONG_RUN_ID=$TREELEARN_LONG_RUN_ID,TREELEARN_LONG_ROOT=$TREELEARN_LONG_ROOT,TREELEARN_LONG_FREEZE=$TREELEARN_LONG_FREEZE,TREELEARN_LONG_SELECTION_FREEZE=$TREELEARN_LONG_SELECTION_FREEZE,TREELEARN_LONG_DURABLE_ROOT=$TREELEARN_LONG_DURABLE_ROOT,TREELEARN_LONG_DURABLE_CHECKPOINT=$TREELEARN_LONG_DURABLE_CHECKPOINT,TREELEARN_LONG_RETENTION_MANIFEST=$TREELEARN_LONG_RETENTION_MANIFEST,TREELEARN_LONG_SELECTED_FREEZE=$TREELEARN_LONG_SELECTED_FREEZE,TREELEARN_LONG_METADATA_BASE=$TREELEARN_LONG_METADATA_BASE,TREELEARN_LONG_TABLES_BASE=$TREELEARN_LONG_TABLES_BASE"

PREP=""
CROPS=""
CONSOLIDATE=""
TRAIN=""
VALIDATION=""
SELECTION=""
GATE=""
SUBMITTED_JOBS=()
SUBMISSION_COMPLETE=0

write_state() {
  {
    printf 'TREELEARN_LONG_SUBMISSION_STATUS=%q\n' "${1:-submitting}"
    printf 'TREELEARN_LONG_RUN_ID=%q\n' "$TREELEARN_LONG_RUN_ID"
    printf 'TREELEARN_LONG_ROOT=%q\n' "$TREELEARN_LONG_ROOT"
    printf 'TREELEARN_LONG_FREEZE=%q\n' "$TREELEARN_LONG_FREEZE"
    printf 'TREELEARN_LONG_PREP_JOB=%q\n' "$PREP"
    printf 'TREELEARN_LONG_CROPS_JOB=%q\n' "$CROPS"
    printf 'TREELEARN_LONG_CONSOLIDATE_JOB=%q\n' "$CONSOLIDATE"
    printf 'TREELEARN_LONG_TRAIN_JOB=%q\n' "$TRAIN"
    printf 'TREELEARN_LONG_VALIDATION_JOB=%q\n' "$VALIDATION"
    printf 'TREELEARN_LONG_SELECTION_JOB=%q\n' "$SELECTION"
    printf 'TREELEARN_LONG_GATE_JOB=%q\n' "$GATE"
    printf 'TREELEARN_LONG_SELECTION_FREEZE=%q\n' "$TREELEARN_LONG_SELECTION_FREEZE"
    printf 'TREELEARN_LONG_DURABLE_ROOT=%q\n' "$TREELEARN_LONG_DURABLE_ROOT"
    printf 'TREELEARN_LONG_DURABLE_CHECKPOINT=%q\n' "$TREELEARN_LONG_DURABLE_CHECKPOINT"
    printf 'TREELEARN_LONG_RETENTION_MANIFEST=%q\n' "$TREELEARN_LONG_RETENTION_MANIFEST"
    printf 'TREELEARN_LONG_SELECTED_FREEZE=%q\n' "$TREELEARN_LONG_SELECTED_FREEZE"
  } > "$STATE_FILE"
}

rollback_partial_submission() {
  status=$?
  trap - EXIT
  if [[ "$status" -ne 0 && "$SUBMISSION_COMPLETE" -ne 1 ]]; then
    if [[ "${#SUBMITTED_JOBS[@]}" -gt 0 ]]; then
      scancel "${SUBMITTED_JOBS[@]}" 2>/dev/null || true
    fi
    write_state rolled_back_after_partial_submission
  fi
  exit "$status"
}
trap rollback_partial_submission EXIT
write_state submitting

PREP=$(sbatch --parsable --export="$EXPORTS" \
  methods/treelearn/slurm/prepare_for_instance_finetune_long.sbatch)
SUBMITTED_JOBS+=("$PREP")
write_state submitting
CROPS=$(sbatch --parsable --array=1,2,4-6,9-19%16 \
  --dependency="afterok:$PREP" --kill-on-invalid-dep=yes --export="$EXPORTS" \
  methods/treelearn/slurm/generate_for_instance_finetune_long_crops.sbatch)
SUBMITTED_JOBS+=("$CROPS")
write_state submitting
CONSOLIDATE=$(sbatch --parsable --dependency="afterok:$CROPS" \
  --kill-on-invalid-dep=yes --export="$EXPORTS" \
  methods/treelearn/slurm/consolidate_for_instance_finetune_long_crops.sbatch)
SUBMITTED_JOBS+=("$CONSOLIDATE")
write_state submitting
TRAIN=$(sbatch --parsable --array=0-7%8 \
  --dependency="afterok:$CONSOLIDATE" --kill-on-invalid-dep=yes --export="$EXPORTS" \
  methods/treelearn/slurm/run_for_instance_finetune_long_trial.sbatch)
SUBMITTED_JOBS+=("$TRAIN")
write_state submitting
VALIDATION=$(sbatch --parsable --array=0-40%8 \
  --dependency="afterok:$TRAIN" --kill-on-invalid-dep=yes --export="$EXPORTS" \
  methods/treelearn/slurm/run_for_instance_finetune_long_validation.sbatch)
SUBMITTED_JOBS+=("$VALIDATION")
write_state submitting
SELECTION=$(sbatch --parsable --dependency="afterok:$VALIDATION" \
  --kill-on-invalid-dep=yes --export="$EXPORTS" \
  methods/treelearn/slurm/select_for_instance_finetune_long.sbatch)
SUBMITTED_JOBS+=("$SELECTION")
write_state submitting
GATE=$(sbatch --parsable --dependency="afterok:$SELECTION" \
  --kill-on-invalid-dep=yes --export="$EXPORTS" \
  methods/treelearn/slurm/gate_for_instance_finetune_long.sbatch)
SUBMITTED_JOBS+=("$GATE")
SUBMISSION_COMPLETE=1
write_state chain_submitted
trap - EXIT

echo "run_id=$TREELEARN_LONG_RUN_ID"
echo "prep_job=$PREP crops_job=$CROPS consolidate_job=$CONSOLIDATE"
echo "training_job=$TRAIN validation_job=$VALIDATION selection_job=$SELECTION"
echo "gate_job=$GATE"
echo "state_file=$STATE_FILE"
echo "No held-out test job was submitted."
