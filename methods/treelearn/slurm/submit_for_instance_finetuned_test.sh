#!/usr/bin/env bash
set -euo pipefail

if [[ "${TREELEARN_FINETUNED_TEST_CONFIRMED:-0}" != "1" ]]; then
  echo "Set TREELEARN_FINETUNED_TEST_CONFIRMED=1 after manual held-out-test authorization." >&2
  exit 2
fi

PROJECT_ROOT="${TREELEARN_PROJECT_ROOT:-$HOME/scratch/tree-seg-benchmark}"
TREELEARN_ENV="${TREELEARN_ENV:-$HOME/fastscratch/venvs/treelearn}"
TREELEARN_REPO="${TREELEARN_REPO:-$HOME/fastscratch/external/TreeLearn}"
TREELEARN_DATASET_ROOT="${TREELEARN_DATASET_ROOT:-$HOME/data/datasets/for_instance/FORinstance_dataset}"
RUN_ID="${TREELEARN_LONG_RUN_ID:?Set TREELEARN_LONG_RUN_ID to the frozen long-run ID.}"
if [[ ! "$RUN_ID" =~ ^treelearn_for-instance_fine_tuned_on_dev_long_[0-9]{8}_[0-9]{6}$ ]]; then
  echo "Unexpected TreeLearn long-run ID: $RUN_ID" >&2
  exit 2
fi

cd "$PROJECT_ROOT"
mkdir -p logs/treelearn_for_instance
COMMIT=$(git rev-parse HEAD)
test -z "$(git status --porcelain)"
test -x "$TREELEARN_ENV/bin/python"
test -x "$HOME/fastscratch/venvs/treebench/bin/python"
test -d "$TREELEARN_REPO/.git"
test "$(git -C "$TREELEARN_REPO" rev-parse HEAD)" = fd240ce7caa4c444fe3418aca454dc578bc557d4
test -z "$(git -C "$TREELEARN_REPO" status --porcelain)"
test -d "$TREELEARN_DATASET_ROOT"
test -f "$TREELEARN_DATASET_ROOT/data_split_metadata.csv"

SELECTED_FREEZE="$PROJECT_ROOT/results/metadata/treelearn_for_instance/long_finetune_runs/$RUN_ID/selected_checkpoint_freeze.json"
test -f "$SELECTED_FREEZE"
mapfile -t SELECTED < <(
  "$HOME/fastscratch/venvs/treebench/bin/python" -c \
    'import json,sys; p=json.load(open(sys.argv[1])); assert p["status"] == "frozen_selected_checkpoint_pending_manual_held_out_test_authorisation"; assert p["source_long_run_id"] == sys.argv[2]; assert p["held_out_test_accessed"] is False; assert p["test_jobs_submitted"] == 0; print(p["checkpoint"]); print(p["checkpoint_sha256"])' \
    "$SELECTED_FREEZE" "$RUN_ID"
)
CHECKPOINT="${SELECTED[0]:?Selected checkpoint path is missing}"
CHECKPOINT_SHA256="${SELECTED[1]:?Selected checkpoint SHA-256 is missing}"
test -f "$CHECKPOINT"
test "$(sha256sum "$CHECKPOINT" | cut -d ' ' -f 1)" = "$CHECKPOINT_SHA256"

RUNTIME_BASE="$PROJECT_ROOT/data/interim/treelearn/for_instance/finetuned_test"
PREDICTION_BASE="$PROJECT_ROOT/data/predictions/treelearn/for_instance_finetuned_test"
METADATA_BASE="$PROJECT_ROOT/results/metadata/treelearn_for_instance/finetuned_test_runs"
TABLE_BASE="$PROJECT_ROOT/results/tables/treelearn_for_instance/finetuned_test"
RUNTIME_ROOT="$RUNTIME_BASE/$RUN_ID"
PREDICTION_ROOT="$PREDICTION_BASE/$RUN_ID"
METADATA_ROOT="$METADATA_BASE/$RUN_ID"
TABLE_ROOT="$TABLE_BASE/$RUN_ID"
FREEZE_ROOT="$PROJECT_ROOT/results/metadata/treelearn_for_instance/finetuned_test_freezes/$RUN_ID"
TEST_FREEZE="$FREEZE_ROOT/test_freeze.json"
TEST_MANIFEST="$FREEZE_ROOT/test_manifest.csv"
RUN_SUMMARY="$TABLE_ROOT/run_summary.json"
FINAL_SUMMARY="$TABLE_ROOT/final_summary.csv"
RETENTION_MANIFEST="$TABLE_ROOT/retention_manifest.json"
COMPLETION_GATE="$METADATA_ROOT/completion_gate.json"
STATE_FILE="$HOME/fastscratch/treelearn_finetuned_test_${RUN_ID}.env"

for target in "$RUNTIME_ROOT" "$PREDICTION_ROOT" "$METADATA_ROOT" "$TABLE_ROOT" "$FREEZE_ROOT" "$STATE_FILE"; do
  if [[ -e "$target" ]]; then
    echo "Refusing repeated or colliding TreeLearn test target: $target" >&2
    exit 2
  fi
done

EXPORTS="ALL,TREELEARN_FINETUNED_TEST_CONFIRMED=1,TREELEARN_TEST_RUN_ID=$RUN_ID,TREELEARN_TEST_BENCHMARK_COMMIT=$COMMIT,TREELEARN_ENV=$TREELEARN_ENV,TREELEARN_REPO=$TREELEARN_REPO,TREELEARN_DATASET_ROOT=$TREELEARN_DATASET_ROOT,TREELEARN_TEST_SELECTED_FREEZE=$SELECTED_FREEZE,TREELEARN_TEST_CHECKPOINT=$CHECKPOINT,TREELEARN_TEST_CHECKPOINT_SHA256=$CHECKPOINT_SHA256,TREELEARN_TEST_RUNTIME_BASE=$RUNTIME_BASE,TREELEARN_TEST_PREDICTION_BASE=$PREDICTION_BASE,TREELEARN_TEST_METADATA_BASE=$METADATA_BASE,TREELEARN_TEST_TABLE_BASE=$TABLE_BASE,TREELEARN_TEST_RUNTIME_ROOT=$RUNTIME_ROOT,TREELEARN_TEST_PREDICTION_ROOT=$PREDICTION_ROOT,TREELEARN_TEST_METADATA_ROOT=$METADATA_ROOT,TREELEARN_TEST_TABLE_ROOT=$TABLE_ROOT,TREELEARN_TEST_FREEZE=$TEST_FREEZE,TREELEARN_TEST_MANIFEST=$TEST_MANIFEST,TREELEARN_TEST_RUN_SUMMARY=$RUN_SUMMARY,TREELEARN_TEST_FINAL_SUMMARY=$FINAL_SUMMARY,TREELEARN_TEST_RETENTION_MANIFEST=$RETENTION_MANIFEST,TREELEARN_TEST_COMPLETION_GATE=$COMPLETION_GATE"

PREP=""
TEST_JOB=""
SUMMARY=""
GATE=""
SUBMITTED=()
COMPLETE=0
write_state() {
  {
    printf 'TREELEARN_TEST_SUBMISSION_STATUS=%q\n' "${1:-submitting}"
    printf 'TREELEARN_TEST_RUN_ID=%q\n' "$RUN_ID"
    printf 'TREELEARN_TEST_PREP_JOB=%q\n' "$PREP"
    printf 'TREELEARN_TEST_ARRAY_JOB=%q\n' "$TEST_JOB"
    printf 'TREELEARN_TEST_SUMMARY_JOB=%q\n' "$SUMMARY"
    printf 'TREELEARN_TEST_GATE_JOB=%q\n' "$GATE"
    printf 'TREELEARN_TEST_FREEZE=%q\n' "$TEST_FREEZE"
    printf 'TREELEARN_TEST_MANIFEST=%q\n' "$TEST_MANIFEST"
    printf 'TREELEARN_TEST_PREDICTION_ROOT=%q\n' "$PREDICTION_ROOT"
    printf 'TREELEARN_TEST_METADATA_ROOT=%q\n' "$METADATA_ROOT"
    printf 'TREELEARN_TEST_TABLE_ROOT=%q\n' "$TABLE_ROOT"
    printf 'TREELEARN_TEST_RUN_SUMMARY=%q\n' "$RUN_SUMMARY"
    printf 'TREELEARN_TEST_FINAL_SUMMARY=%q\n' "$FINAL_SUMMARY"
    printf 'TREELEARN_TEST_RETENTION_MANIFEST=%q\n' "$RETENTION_MANIFEST"
    printf 'TREELEARN_TEST_COMPLETION_GATE=%q\n' "$COMPLETION_GATE"
  } > "$STATE_FILE"
}
rollback() {
  status=$?
  trap - EXIT
  if [[ "$status" -ne 0 && "$COMPLETE" -ne 1 ]]; then
    if ((${#SUBMITTED[@]})); then scancel "${SUBMITTED[@]}" 2>/dev/null || true; fi
    write_state rolled_back_after_partial_submission
  fi
  exit "$status"
}
trap rollback EXIT
write_state submitting

PREP=$(sbatch --parsable --export="$EXPORTS" \
  methods/treelearn/slurm/prepare_for_instance_finetuned_test.sbatch)
SUBMITTED+=("$PREP")
write_state submitting
TEST_JOB=$(sbatch --parsable --array=0-10%2 --dependency="afterok:$PREP" \
  --kill-on-invalid-dep=yes --export="$EXPORTS" \
  methods/treelearn/slurm/run_for_instance_finetuned_test.sbatch)
SUBMITTED+=("$TEST_JOB")
write_state submitting
SUMMARY=$(sbatch --parsable --dependency="afterany:$TEST_JOB" \
  --export="$EXPORTS" methods/treelearn/slurm/summarise_for_instance_finetuned_test.sbatch)
SUBMITTED+=("$SUMMARY")
write_state submitting
GATE=$(sbatch --parsable --dependency="afterok:$SUMMARY" \
  --kill-on-invalid-dep=yes --export="$EXPORTS" \
  methods/treelearn/slurm/gate_for_instance_finetuned_test.sbatch)
SUBMITTED+=("$GATE")
COMPLETE=1
write_state chain_submitted
trap - EXIT

echo "run_id=$RUN_ID"
echo "prep_job=$PREP test_job=$TEST_JOB summary_job=$SUMMARY gate_job=$GATE"
echo "state_file=$STATE_FILE"
echo "freeze=$TEST_FREEZE"
echo "final_summary=$FINAL_SUMMARY"
echo "No training job was submitted. Repeated test submission is refused."
