#!/usr/bin/env bash
set -euo pipefail
test "${TREELEARN_FINETUNE_DEV_CONFIRMED:-0}" = 1 || {
  echo "Set TREELEARN_FINETUNE_DEV_CONFIRMED=1." >&2; exit 2;
}
PROJECT_ROOT="${TREELEARN_PROJECT_ROOT:-$HOME/scratch/tree-seg-benchmark}"
TREELEARN_ENV="${TREELEARN_ENV:-$HOME/fastscratch/venvs/treelearn}"
TREELEARN_REPO="${TREELEARN_REPO:-$HOME/fastscratch/external/TreeLearn}"
TREELEARN_CHECKPOINT="${TREELEARN_CHECKPOINT:-$HOME/fastscratch/treelearn_checkpoints/model_weights_20241213.pth}"
DEV_RUN="treelearn_for-instance_published_pretrained_development_20260712_150030"
DEV_MANIFEST="${TREELEARN_DEV_MANIFEST_JSON:-$PROJECT_ROOT/results/metadata/treelearn_for_instance/development_runs/$DEV_RUN/development_manifest.json}"
CROPS="${TREELEARN_FINETUNE_CROPS:-512}"
cd "$PROJECT_ROOT"
mkdir -p logs/treelearn_for_instance
test -f "$DEV_MANIFEST"; test -f "$TREELEARN_CHECKPOINT"; test -x "$TREELEARN_ENV/bin/python"
test "$(git -C "$TREELEARN_REPO" rev-parse HEAD)" = fd240ce7caa4c444fe3418aca454dc578bc557d4
test -z "$(git status --porcelain)"; test -z "$(git -C "$TREELEARN_REPO" status --porcelain)"
COMMIT=$(git rev-parse HEAD)
STAMP="${TREELEARN_FINETUNE_STAMP:-$(date -u +%Y%m%d_%H%M%S)}"
RUN_ID="treelearn_for-instance_fine_tuned_on_dev_$STAMP"
RUN_ROOT="$HOME/fastscratch/treelearn_finetune/$RUN_ID"
STATE_FILE="$HOME/fastscratch/treelearn_finetune_${RUN_ID}.env"
test ! -e "$RUN_ROOT"; test ! -e "$STATE_FILE"
EXPORTS="ALL,TREELEARN_FINETUNE_DEV_CONFIRMED=1,TREELEARN_EXPECTED_BENCHMARK_COMMIT=$COMMIT,TREELEARN_ENV=$TREELEARN_ENV,TREELEARN_REPO=$TREELEARN_REPO,TREELEARN_CHECKPOINT=$TREELEARN_CHECKPOINT,TREELEARN_DEV_MANIFEST_JSON=$DEV_MANIFEST,TREELEARN_FINETUNE_ROOT=$RUN_ROOT,TREELEARN_FINETUNE_CROPS=$CROPS"
PREP=$(sbatch --parsable --export="$EXPORTS" methods/treelearn/slurm/prepare_for_instance_finetune.sbatch)
SMOKE=$(sbatch --parsable --dependency="afterok:$PREP" --kill-on-invalid-dep=yes --time=01:00:00 --export="$EXPORTS,TREELEARN_FINETUNE_MODE=smoke,TREELEARN_EXPECTED_EPOCH=1" methods/treelearn/slurm/run_for_instance_finetune.sbatch)
FULL=$(sbatch --parsable --dependency="afterok:$SMOKE" --kill-on-invalid-dep=yes --export="$EXPORTS,TREELEARN_FINETUNE_MODE=full,TREELEARN_EXPECTED_EPOCH=100" methods/treelearn/slurm/run_for_instance_finetune.sbatch)
{
  printf 'TREELEARN_FINETUNE_RUN_ID=%q\n' "$RUN_ID"
  printf 'TREELEARN_FINETUNE_PREP_JOB=%q\n' "$PREP"
  printf 'TREELEARN_FINETUNE_SMOKE_JOB=%q\n' "$SMOKE"
  printf 'TREELEARN_FINETUNE_FULL_JOB=%q\n' "$FULL"
  printf 'TREELEARN_FINETUNE_ROOT=%q\n' "$RUN_ROOT"
  printf 'TREELEARN_FINETUNE_CHECKPOINT=%q\n' "$RUN_ROOT/work_dirs/finetune_full/epoch_100.pth"
} > "$STATE_FILE"
echo "run_id=$RUN_ID"
echo "prep_job=$PREP smoke_job=$SMOKE full_job=$FULL"
echo "state_file=$STATE_FILE"
echo "No held-out test job was submitted."
