#!/usr/bin/env bash

set -euo pipefail

if [[ "${FF3D_FINETUNE_PREPARATION_CONFIRMED:-0}" != "1" ]]; then
  echo "Set FF3D_FINETUNE_PREPARATION_CONFIRMED=1 after the development run verifies." >&2
  exit 2
fi
: "${FF3D_SOURCE_RUN_ROOT:?Set FF3D_SOURCE_RUN_ROOT}"
: "${FF3D_VERIFICATION_JSON:?Set FF3D_VERIFICATION_JSON}"

BENCHMARK_ROOT="${FF3D_BENCHMARK_ROOT:-$(pwd)}"
RUNTIME_ROOT="${FF3D_RUNTIME_ROOT:-$HOME/fastscratch/forestformer3d}"
BASE_SIF="${FF3D_BASE_SIF:-$RUNTIME_ROOT/containers/pytorch_1.13.1_cuda11.6_cudnn8_devel.sif}"
ENV_ROOT="${FF3D_ENV_ROOT:-$RUNTIME_ROOT/environments/forestformer3d_6a75c3735e4a_741e13d08e51_20260723T191340}"
CHECKPOINT="${FF3D_CHECKPOINT:-$RUNTIME_ROOT/checkpoints/clean_forestformer/clean_forestformer/epoch_3000_fix.pth}"
TREEBENCH_PYTHON="${FF3D_TREEBENCH_PYTHON:-$HOME/fastscratch/venvs/treebench/bin/python}"
METHOD_ROOT="$BENCHMARK_ROOT/methods/forestformer3d"
BASE_SIF_SHA256="4a35d5a57c1d57061f899b514329ad8ec2bf74a9ff31d103c0a53a289e07c84f"
CHECKPOINT_SHA256="01037a648596832238ac72ea2f5eef87ceaf5aeb399e56ff4b760ba1ed1c777e"

cd "$BENCHMARK_ROOT"
test "$(git branch --show-current)" = "method/forestformer3d"
test -z "$(git status --porcelain)"
BENCHMARK_COMMIT="$(git rev-parse HEAD)"
echo "$BASE_SIF_SHA256  $BASE_SIF" | sha256sum --check --status
echo "$CHECKPOINT_SHA256  $CHECKPOINT" | sha256sum --check --status
test -f "$ENV_ROOT/environment_build.complete"
test -f "$FF3D_SOURCE_RUN_ROOT/development.complete"
"$TREEBENCH_PYTHON" -c \
  'import json,sys; p=json.load(open(sys.argv[1])); assert p["status"] == "verified"; assert p["split"] == "development"; assert p["held_out_access"] is False; assert p["task_count"] == 21' \
  "$FF3D_VERIFICATION_JSON"

TIMESTAMP="$(date -u +%Y%m%dT%H%M%S)"
RUN_ID="forestformer3d__for-instance__fine-tuned-on-dev__best-validation__development__${TIMESTAMP}"
RUN_ROOT="$RUNTIME_ROOT/runs/development-finetune/$RUN_ID"
STATE_FILE="$RUNTIME_ROOT/state/${RUN_ID}.env"
test ! -e "$RUN_ROOT"
test ! -e "$STATE_FILE"
mkdir -p "$RUNTIME_ROOT/logs" "$(dirname "$STATE_FILE")"

JOB="$(
  sbatch --parsable \
    --output="$RUNTIME_ROOT/logs/${RUN_ID}_prepare_%j.out" \
    --error="$RUNTIME_ROOT/logs/${RUN_ID}_prepare_%j.err" \
    --export=ALL,FF3D_FINETUNE_PREPARATION_CONFIRMED=1,FF3D_BENCHMARK_ROOT="$BENCHMARK_ROOT",FF3D_BENCHMARK_COMMIT="$BENCHMARK_COMMIT",FF3D_BASE_SIF="$BASE_SIF",FF3D_ENV_ROOT="$ENV_ROOT",FF3D_CHECKPOINT="$CHECKPOINT",FF3D_CHECKPOINT_SHA256="$CHECKPOINT_SHA256",FF3D_TREEBENCH_PYTHON="$TREEBENCH_PYTHON",FF3D_SOURCE_RUN_ROOT="$FF3D_SOURCE_RUN_ROOT",FF3D_VERIFICATION_JSON="$FF3D_VERIFICATION_JSON",FF3D_RUN_ROOT="$RUN_ROOT" \
    "$METHOD_ROOT/slurm/prepare_finetune.sbatch"
)"
{
  printf 'FF3D_WORKFLOW=%q\n' "fine_tune_preparation"
  printf 'FF3D_RUN_ID=%q\n' "$RUN_ID"
  printf 'FF3D_RUN_ROOT=%q\n' "$RUN_ROOT"
  printf 'FF3D_JOB_IDS=%q\n' "$JOB"
  printf 'FF3D_EXPECTED_FILES=%q\n' "$RUN_ROOT/fine_tune_freeze.json|$RUN_ROOT/fine_tune_split.csv|$RUN_ROOT/configs/effective_smoke.py|$RUN_ROOT/configs/effective_full.py|$RUN_ROOT/configs/config_manifest.json|$RUN_ROOT/fine_tune_preparation.complete"
  printf 'FF3D_BENCHMARK_COMMIT=%q\n' "$BENCHMARK_COMMIT"
} > "$STATE_FILE"
echo "run_id=$RUN_ID"
echo "state_file=$STATE_FILE"
echo "preparation_job=$JOB"
echo "cancel_command=scancel $JOB"
echo "scope=16 development-training plus 5 development-validation plots; held-out access false"
echo "resources=nodes partition, 4 CPUs, 32 GiB, 4 hours"
echo "runtime_estimate=10-45 minutes to stage and hash about 2.6 GB; queue waiting separate"
if [[ "${FF3D_NO_WATCH:-0}" != "1" ]]; then
  exec bash "$METHOD_ROOT/slurm/monitor_workflow.sh" \
    "$STATE_FILE" --watch "${FF3D_MONITOR_SECONDS:-30}"
fi
