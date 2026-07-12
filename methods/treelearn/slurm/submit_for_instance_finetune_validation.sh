#!/usr/bin/env bash
set -euo pipefail
test "${TREELEARN_FINETUNE_VALIDATION_CONFIRMED:-0}" = 1 || {
  echo "Set TREELEARN_FINETUNE_VALIDATION_CONFIRMED=1." >&2; exit 2;
}
PROJECT_ROOT="${TREELEARN_PROJECT_ROOT:-$HOME/scratch/tree-seg-benchmark}"
TRAIN_STATE="${TREELEARN_FINETUNE_STATE_FILE:-$(ls -t "$HOME"/fastscratch/treelearn_finetune_treelearn_for-instance_fine_tuned_on_dev_*.env | head -1)}"
source "$TRAIN_STATE"
TREELEARN_ENV="${TREELEARN_ENV:-$HOME/fastscratch/venvs/treelearn}"
TREELEARN_REPO="${TREELEARN_REPO:-$HOME/fastscratch/external/TreeLearn}"
TREELEARN_DATASET_ROOT="${TREELEARN_DATASET_ROOT:-$HOME/data/datasets/for_instance/FORinstance_dataset}"
DEV_RUN=treelearn_for-instance_published_pretrained_development_20260712_150030
DEV_MANIFEST="$PROJECT_ROOT/results/metadata/treelearn_for_instance/development_runs/$DEV_RUN/development_manifest.json"
FREEZE="$TREELEARN_FINETUNE_ROOT/finetune_freeze.json"
cd "$PROJECT_ROOT"
mkdir -p logs/treelearn_for_instance
test -f "$TREELEARN_FINETUNE_CHECKPOINT"; test -f "$FREEZE"; test -f "$DEV_MANIFEST"
test -z "$(git status --porcelain)"
COMMIT=$(git rev-parse HEAD)
SHA=$(sha256sum "$TREELEARN_FINETUNE_CHECKPOINT" | cut -d ' ' -f 1)
METADATA_ROOT="$PROJECT_ROOT/results/metadata/treelearn_for_instance/development_runs/$TREELEARN_FINETUNE_RUN_ID"
TABLE_ROOT="$PROJECT_ROOT/results/tables/treelearn_for_instance/development_runs/$TREELEARN_FINETUNE_RUN_ID"
STATE_FILE="$HOME/fastscratch/treelearn_finetune_validation_${TREELEARN_FINETUNE_RUN_ID}.env"
test ! -e "$METADATA_ROOT"; test ! -e "$TABLE_ROOT"; test ! -e "$STATE_FILE"
EXPORTS="ALL,TREELEARN_FINETUNE_VALIDATION_CONFIRMED=1,TREELEARN_EXPECTED_BENCHMARK_COMMIT=$COMMIT,TREELEARN_ENV=$TREELEARN_ENV,TREELEARN_REPO=$TREELEARN_REPO,TREELEARN_DATASET_ROOT=$TREELEARN_DATASET_ROOT,TREELEARN_FINETUNE_RUN_ID=$TREELEARN_FINETUNE_RUN_ID,TREELEARN_FINETUNE_CHECKPOINT=$TREELEARN_FINETUNE_CHECKPOINT,TREELEARN_FINETUNE_CHECKPOINT_SHA256=$SHA,TREELEARN_FINETUNE_FREEZE=$FREEZE,TREELEARN_DEV_MANIFEST_JSON=$DEV_MANIFEST,TREELEARN_FINETUNE_METADATA_ROOT=$METADATA_ROOT,TREELEARN_FINETUNE_TABLE_ROOT=$TABLE_ROOT"
ARRAY=$(sbatch --parsable --array=0,3,7,8,20%2 --export="$EXPORTS" methods/treelearn/slurm/run_for_instance_finetune_validation.sbatch)
SUMMARY=$(sbatch --parsable --dependency="afterok:$ARRAY" --kill-on-invalid-dep=yes --export="$EXPORTS" methods/treelearn/slurm/summarise_for_instance_finetune_validation.sbatch)
{
  printf 'TREELEARN_FINETUNE_VALIDATION_RUN_ID=%q\n' "$TREELEARN_FINETUNE_RUN_ID"
  printf 'TREELEARN_FINETUNE_VALIDATION_ARRAY_JOB=%q\n' "$ARRAY"
  printf 'TREELEARN_FINETUNE_VALIDATION_SUMMARY_JOB=%q\n' "$SUMMARY"
  printf 'TREELEARN_FINETUNE_VALIDATION_SUMMARY=%q\n' "$TABLE_ROOT/validation_summary.json"
  printf 'TREELEARN_FINETUNE_VALIDATION_SITE_SUMMARY=%q\n' "$TABLE_ROOT/validation_site_summary.csv"
} > "$STATE_FILE"
echo "run_id=$TREELEARN_FINETUNE_RUN_ID"
echo "validation_job=$ARRAY summary_job=$SUMMARY"
echo "state_file=$STATE_FILE"
echo "No held-out test job was submitted."
