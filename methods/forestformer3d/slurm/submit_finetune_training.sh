#!/usr/bin/env bash

set -euo pipefail

if [[ "${FF3D_FINETUNE_TRAINING_CONFIRMED:-0}" != "1" ]]; then
  echo "Set FF3D_FINETUNE_TRAINING_CONFIRMED=1 after the initialization smoke passes." >&2
  exit 2
fi
: "${FF3D_RUN_ROOT:?Set FF3D_RUN_ROOT to the prepared fine-tuning run}"

BENCHMARK_ROOT="${FF3D_BENCHMARK_ROOT:-$(pwd)}"
RUNTIME_ROOT="${FF3D_RUNTIME_ROOT:-$HOME/fastscratch/forestformer3d}"
BASE_SIF="${FF3D_BASE_SIF:-$RUNTIME_ROOT/containers/pytorch_1.13.1_cuda11.6_cudnn8_devel.sif}"
ENV_ROOT="${FF3D_ENV_ROOT:-$RUNTIME_ROOT/environments/forestformer3d_6a75c3735e4a_741e13d08e51_20260723T191340}"
CHECKPOINT="${FF3D_CHECKPOINT:-$RUNTIME_ROOT/checkpoints/clean_forestformer/clean_forestformer/epoch_3000_fix.pth}"
METHOD_ROOT="$BENCHMARK_ROOT/methods/forestformer3d"
BASE_SIF_SHA256="4a35d5a57c1d57061f899b514329ad8ec2bf74a9ff31d103c0a53a289e07c84f"
CHECKPOINT_SHA256="01037a648596832238ac72ea2f5eef87ceaf5aeb399e56ff4b760ba1ed1c777e"

cd "$BENCHMARK_ROOT"
test "$(git branch --show-current)" = "method/forestformer3d"
test -z "$(git status --porcelain)"
BENCHMARK_COMMIT="$(git rev-parse HEAD)"
echo "$CHECKPOINT_SHA256  $CHECKPOINT" | sha256sum --check --status
test -f "$ENV_ROOT/environment_build.complete"
test -f "$FF3D_RUN_ROOT/fine_tune_initialization_smoke.complete"
test ! -e "$FF3D_RUN_ROOT/fine_tune_training.started"
RUN_ID="$(basename "$FF3D_RUN_ROOT")"
STATE_FILE="$RUNTIME_ROOT/state/${RUN_ID}__training.env"
test ! -e "$STATE_FILE"
mkdir -p "$FF3D_RUN_ROOT/logs" "$(dirname "$STATE_FILE")"

JOB="$(
  sbatch --parsable \
    --output="$FF3D_RUN_ROOT/logs/training_%j.out" \
    --error="$FF3D_RUN_ROOT/logs/training_%j.err" \
    --export=ALL,FF3D_FINETUNE_TRAINING_CONFIRMED=1,FF3D_BENCHMARK_ROOT="$BENCHMARK_ROOT",FF3D_BENCHMARK_COMMIT="$BENCHMARK_COMMIT",FF3D_BASE_SIF="$BASE_SIF",FF3D_BASE_SIF_SHA256="$BASE_SIF_SHA256",FF3D_ENV_ROOT="$ENV_ROOT",FF3D_CHECKPOINT="$CHECKPOINT",FF3D_RUN_ROOT="$FF3D_RUN_ROOT" \
    "$METHOD_ROOT/slurm/run_finetune_training.sbatch"
)"
{
  printf 'FF3D_WORKFLOW=%q\n' "fine_tune_training"
  printf 'FF3D_RUN_ID=%q\n' "$RUN_ID"
  printf 'FF3D_RUN_ROOT=%q\n' "$FF3D_RUN_ROOT"
  printf 'FF3D_JOB_IDS=%q\n' "$JOB"
  printf 'FF3D_EXPECTED_FILES=%q\n' "$FF3D_RUN_ROOT/training_full/epoch_7.pth|$FF3D_RUN_ROOT/training_full/epoch_14.pth|$FF3D_RUN_ROOT/training_full/epoch_21.pth|$FF3D_RUN_ROOT/training_full/epoch_28.pth|$FF3D_RUN_ROOT/training_full/epoch_35.pth|$FF3D_RUN_ROOT/checkpoint_inventory.json|$FF3D_RUN_ROOT/fine_tune_training.complete"
  printf 'FF3D_BENCHMARK_COMMIT=%q\n' "$BENCHMARK_COMMIT"
} > "$STATE_FILE"
echo "run_id=$RUN_ID"
echo "state_file=$STATE_FILE"
echo "training_job=$JOB"
echo "cancel_command=scancel $JOB"
echo "scope=16 development-training plots only; held-out access false"
echo "resources=gpu-a100-lowbig, 1 A100, 12 CPUs, 128 GiB, 24 hours"
echo "training_budget=35 epochs, 560 examples, 280 optimizer steps, checkpoints 7/14/21/28/35"
echo "runtime_estimate=to be recalculated from the initialization smoke; queue waiting separate"
if [[ "${FF3D_NO_WATCH:-0}" != "1" ]]; then
  exec bash "$METHOD_ROOT/slurm/monitor_workflow.sh" \
    "$STATE_FILE" --watch "${FF3D_MONITOR_SECONDS:-30}"
fi
