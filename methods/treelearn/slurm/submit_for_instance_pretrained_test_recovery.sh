#!/usr/bin/env bash
set -euo pipefail

if [[ "${TREELEARN_PRETRAINED_TEST_RECOVERY_CONFIRMED:-0}" != "1" ]]; then
  echo "Set TREELEARN_PRETRAINED_TEST_RECOVERY_CONFIRMED=1 for the approved execution-only recovery." >&2
  exit 2
fi
SOURCE_STATE="${TREELEARN_PRETRAINED_TEST_STATE_FILE:?Set TREELEARN_PRETRAINED_TEST_STATE_FILE to the original run state file.}"
SOURCE_STATE=$(realpath "$SOURCE_STATE")
source "$SOURCE_STATE"
ORIGINAL_COMMIT="$TREELEARN_TEST_BENCHMARK_COMMIT"
if [[ "$TREELEARN_TEST_RUN_ID" != "treelearn_for-instance_published_pretrained_20260714_134109" ]]; then
  echo "Recovery authorization does not match run $TREELEARN_TEST_RUN_ID." >&2
  exit 2
fi

PROJECT_ROOT="${TREELEARN_PROJECT_ROOT:-$HOME/scratch/tree-seg-benchmark}"
TREELEARN_ENV="${TREELEARN_ENV:-$HOME/fastscratch/venvs/treelearn}"
TREELEARN_REPO="${TREELEARN_REPO:-$HOME/fastscratch/external/TreeLearn}"
TREELEARN_DATASET_ROOT="${TREELEARN_DATASET_ROOT:-$HOME/data/datasets/for_instance/FORinstance_dataset}"
cd "$PROJECT_ROOT"
RECOVERY_COMMIT=$(git rev-parse HEAD)
test -z "$(git status --porcelain)"
test -x "$TREELEARN_ENV/bin/python"
test -x "$HOME/fastscratch/venvs/treebench/bin/python"
test "$(git -C "$TREELEARN_REPO" rev-parse HEAD)" = fd240ce7caa4c444fe3418aca454dc578bc557d4
test -z "$(git -C "$TREELEARN_REPO" status --porcelain)"
test "$(md5sum "$TREELEARN_TEST_CHECKPOINT" | cut -d ' ' -f 1)" = 106a80de2991c5f23484a3f9d03e3b16
test "$(sha256sum "$TREELEARN_TEST_CHECKPOINT" | cut -d ' ' -f 1)" = "$TREELEARN_TEST_CHECKPOINT_SHA256"
test ! -e "$TREELEARN_TEST_COMPLETION_GATE"

RUNTIME_BASE="$PROJECT_ROOT/data/interim/treelearn/for_instance/pretrained_test"
PREDICTION_BASE="$PROJECT_ROOT/data/predictions/treelearn/for_instance_pretrained_test"
METADATA_BASE="$PROJECT_ROOT/results/metadata/treelearn_for_instance/pretrained_test_runs"
TABLE_BASE="$PROJECT_ROOT/results/tables/treelearn_for_instance/pretrained_test"
RUNTIME_ROOT="$RUNTIME_BASE/$TREELEARN_TEST_RUN_ID"
RECOVERY_MANIFEST="$TREELEARN_TEST_METADATA_ROOT/task_8_execution_recovery.json"
RECOVERY_ARCHIVE="$HOME/fastscratch/treelearn_pretrained_test_recovery_archive/$TREELEARN_TEST_RUN_ID"
RECOVERY_GUARD="$HOME/fastscratch/treelearn_pretrained_test_recovery_once/$TREELEARN_TEST_RUN_ID"
RECOVERY_STATE="$HOME/fastscratch/treelearn_pretrained_test_recovery_${TREELEARN_TEST_RUN_ID}.env"
if ! mkdir -p "$(dirname "$RECOVERY_GUARD")" || ! mkdir "$RECOVERY_GUARD" 2>/dev/null; then
  echo "Refusing repeated pretrained execution recovery: $RECOVERY_GUARD" >&2
  exit 2
fi

EXPORTS="ALL,TREELEARN_PRETRAINED_TEST_CONFIRMED=1,TREELEARN_PRETRAINED_TEST_RECOVERY_CONFIRMED=1,TREELEARN_TEST_RUN_ID=$TREELEARN_TEST_RUN_ID,TREELEARN_TEST_ORIGINAL_BENCHMARK_COMMIT=$ORIGINAL_COMMIT,TREELEARN_TEST_BENCHMARK_COMMIT=$RECOVERY_COMMIT,TREELEARN_ENV=$TREELEARN_ENV,TREELEARN_REPO=$TREELEARN_REPO,TREELEARN_DATASET_ROOT=$TREELEARN_DATASET_ROOT,TREELEARN_TEST_CHECKPOINT=$TREELEARN_TEST_CHECKPOINT,TREELEARN_TEST_CHECKPOINT_SHA256=$TREELEARN_TEST_CHECKPOINT_SHA256,TREELEARN_TEST_RUNTIME_BASE=$RUNTIME_BASE,TREELEARN_TEST_PREDICTION_BASE=$PREDICTION_BASE,TREELEARN_TEST_METADATA_BASE=$METADATA_BASE,TREELEARN_TEST_TABLE_BASE=$TABLE_BASE,TREELEARN_TEST_RUNTIME_ROOT=$RUNTIME_ROOT,TREELEARN_TEST_PREDICTION_ROOT=$TREELEARN_TEST_PREDICTION_ROOT,TREELEARN_TEST_METADATA_ROOT=$TREELEARN_TEST_METADATA_ROOT,TREELEARN_TEST_TABLE_ROOT=$TREELEARN_TEST_TABLE_ROOT,TREELEARN_TEST_FREEZE=$TREELEARN_TEST_FREEZE,TREELEARN_TEST_MANIFEST=$TREELEARN_TEST_MANIFEST,TREELEARN_TEST_RUN_SUMMARY=$TREELEARN_TEST_RUN_SUMMARY,TREELEARN_TEST_FINAL_SUMMARY=$TREELEARN_TEST_FINAL_SUMMARY,TREELEARN_TEST_RETENTION_MANIFEST=$TREELEARN_TEST_RETENTION_MANIFEST,TREELEARN_TEST_COMPLETION_GATE=$TREELEARN_TEST_COMPLETION_GATE,TREELEARN_TEST_RECOVERY_MANIFEST=$RECOVERY_MANIFEST,TREELEARN_TEST_RECOVERY_ARCHIVE=$RECOVERY_ARCHIVE"

PREP=""; TASK=""; SUMMARY=""; GATE=""; SUBMITTED=(); COMPLETE=0
write_state() {
  {
    printf 'TREELEARN_TEST_RECOVERY_STATUS=%q\n' "${1:-submitting}"
    printf 'TREELEARN_TEST_RUN_ID=%q\n' "$TREELEARN_TEST_RUN_ID"
    printf 'TREELEARN_TEST_ORIGINAL_BENCHMARK_COMMIT=%q\n' "$ORIGINAL_COMMIT"
    printf 'TREELEARN_TEST_BENCHMARK_COMMIT=%q\n' "$RECOVERY_COMMIT"
    printf 'TREELEARN_TEST_RECOVERY_PREP_JOB=%q\n' "$PREP"
    printf 'TREELEARN_TEST_RECOVERY_TASK_JOB=%q\n' "$TASK"
    printf 'TREELEARN_TEST_RECOVERY_SUMMARY_JOB=%q\n' "$SUMMARY"
    printf 'TREELEARN_TEST_RECOVERY_GATE_JOB=%q\n' "$GATE"
    printf 'TREELEARN_TEST_RECOVERY_MANIFEST=%q\n' "$RECOVERY_MANIFEST"
    printf 'TREELEARN_TEST_RECOVERY_ARCHIVE=%q\n' "$RECOVERY_ARCHIVE"
    printf 'TREELEARN_TEST_FREEZE=%q\n' "$TREELEARN_TEST_FREEZE"
    printf 'TREELEARN_TEST_PREDICTION_ROOT=%q\n' "$TREELEARN_TEST_PREDICTION_ROOT"
    printf 'TREELEARN_TEST_METADATA_ROOT=%q\n' "$TREELEARN_TEST_METADATA_ROOT"
    printf 'TREELEARN_TEST_TABLE_ROOT=%q\n' "$TREELEARN_TEST_TABLE_ROOT"
    printf 'TREELEARN_TEST_RUN_SUMMARY=%q\n' "$TREELEARN_TEST_RUN_SUMMARY"
    printf 'TREELEARN_TEST_FINAL_SUMMARY=%q\n' "$TREELEARN_TEST_FINAL_SUMMARY"
    printf 'TREELEARN_TEST_RETENTION_MANIFEST=%q\n' "$TREELEARN_TEST_RETENTION_MANIFEST"
    printf 'TREELEARN_TEST_COMPLETION_GATE=%q\n' "$TREELEARN_TEST_COMPLETION_GATE"
  } > "$RECOVERY_STATE"
}
rollback() {
  status=$?; trap - EXIT
  if [[ "$status" -ne 0 && "$COMPLETE" -ne 1 ]]; then
    if ((${#SUBMITTED[@]})); then scancel "${SUBMITTED[@]}" 2>/dev/null || true; fi
    write_state rolled_back_after_partial_submission
    rmdir "$RECOVERY_GUARD" 2>/dev/null || true
  fi
  exit "$status"
}
trap rollback EXIT
write_state submitting
PREP=$(sbatch --parsable --export="$EXPORTS" methods/treelearn/slurm/prepare_for_instance_pretrained_test_recovery.sbatch)
SUBMITTED+=("$PREP"); write_state submitting
TASK=$(sbatch --parsable --dependency="afterok:$PREP" --kill-on-invalid-dep=yes --export="$EXPORTS" methods/treelearn/slurm/run_for_instance_pretrained_test_recovery.sbatch)
SUBMITTED+=("$TASK"); write_state submitting
SUMMARY=$(sbatch --parsable --dependency="afterany:$TASK" --export="$EXPORTS" methods/treelearn/slurm/summarise_for_instance_pretrained_test_recovery.sbatch)
SUBMITTED+=("$SUMMARY"); write_state submitting
GATE=$(sbatch --parsable --dependency="afterok:$SUMMARY" --kill-on-invalid-dep=yes --export="$EXPORTS" methods/treelearn/slurm/gate_for_instance_pretrained_test.sbatch)
SUBMITTED+=("$GATE"); write_state chain_submitted
cp "$RECOVERY_STATE" "$RECOVERY_GUARD/submission.env"
COMPLETE=1; trap - EXIT

echo "run_id=$TREELEARN_TEST_RUN_ID"
echo "prep_job=$PREP recovery_job=$TASK summary_job=$SUMMARY gate_job=$GATE"
echo "state_file=$RECOVERY_STATE"
echo "recovery_manifest=$RECOVERY_MANIFEST"
echo "Recovery is restricted to test task 8; no training or model selection was submitted."
