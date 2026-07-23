#!/usr/bin/env bash

set -euo pipefail

if [[ "${FF3D_DEVELOPMENT_CONFIRMED:-0}" != "1" ]]; then
  echo "Set FF3D_DEVELOPMENT_CONFIRMED=1 after accepting the smoke alignment." >&2
  exit 2
fi
: "${FF3D_PREFLIGHT_MANIFEST:?Set FF3D_PREFLIGHT_MANIFEST to the accepted manifest}"

BENCHMARK_ROOT="${FF3D_BENCHMARK_ROOT:-$(pwd)}"
RUNTIME_ROOT="${FF3D_RUNTIME_ROOT:-$HOME/fastscratch/forestformer3d}"
BASE_SIF="${FF3D_BASE_SIF:-$RUNTIME_ROOT/containers/pytorch_1.13.1_cuda11.6_cudnn8_devel.sif}"
ENV_ROOT="${FF3D_ENV_ROOT:-$RUNTIME_ROOT/environments/forestformer3d_6a75c3735e4a_741e13d08e51_20260723T191340}"
CHECKPOINT="${FF3D_CHECKPOINT:-$RUNTIME_ROOT/checkpoints/clean_forestformer/clean_forestformer/epoch_3000_fix.pth}"
TREEBENCH_PYTHON="${FF3D_TREEBENCH_PYTHON:-$HOME/fastscratch/venvs/treebench/bin/python}"
SMOKE_CONFIRMATION="${FF3D_SMOKE_CONFIRMATION:-$RUNTIME_ROOT/runs/development-smoke/forestformer3d__for-instance__published-pretrained__not-applicable__development-smoke__20260723T213106/evaluation/manual_alignment_confirmation.json}"
METHOD_ROOT="$BENCHMARK_ROOT/methods/forestformer3d"

BASE_SIF_SHA256="4a35d5a57c1d57061f899b514329ad8ec2bf74a9ff31d103c0a53a289e07c84f"
CHECKPOINT_SHA256="01037a648596832238ac72ea2f5eef87ceaf5aeb399e56ff4b760ba1ed1c777e"
CONFIRMATION_SHA256="c4381d65add546910dc901332b56ae16d3df515c69e4c6f61dffbec66a9036f5"

cd "$BENCHMARK_ROOT"
test "$(git branch --show-current)" = "method/forestformer3d"
test -z "$(git status --porcelain)"
BENCHMARK_COMMIT="$(git rev-parse HEAD)"
echo "$BASE_SIF_SHA256  $BASE_SIF" | sha256sum --check --status
echo "$CHECKPOINT_SHA256  $CHECKPOINT" | sha256sum --check --status
echo "$CONFIRMATION_SHA256  $SMOKE_CONFIRMATION" | sha256sum --check --status
test -f "$ENV_ROOT/environment_build.complete"
test -f "$FF3D_PREFLIGHT_MANIFEST"
"$TREEBENCH_PYTHON" -c \
  'import json,sys; p=json.load(open(sys.argv[1])); assert p["schema"] == "forestformer3d_development_manifest_v1"; assert p["dataset_split"] == "development"; assert p["held_out_access"] is False; assert len(p["plots"]) == 21' \
  "$FF3D_PREFLIGHT_MANIFEST"

TIMESTAMP="$(date -u +%Y%m%dT%H%M%S)"
RUN_ID="forestformer3d__for-instance__published-pretrained__not-applicable__development__${TIMESTAMP}"
RUN_ROOT="$RUNTIME_ROOT/runs/development/$RUN_ID"
STATE_FILE="$RUNTIME_ROOT/state/${RUN_ID}.env"
test ! -e "$RUN_ROOT"
test ! -e "$STATE_FILE"
mkdir -p "$RUN_ROOT/logs" "$RUN_ROOT/tasks" "$(dirname "$STATE_FILE")"
cp "$FF3D_PREFLIGHT_MANIFEST" "$RUN_ROOT/development_manifest.json"
MANIFEST="$RUN_ROOT/development_manifest.json"
sha256sum "$MANIFEST" > "$RUN_ROOT/development_manifest.sha256"
cp "$SMOKE_CONFIRMATION" "$RUN_ROOT/manual_alignment_confirmation.json"

ARRAY_JOB="$(
  sbatch --parsable \
    --array=0-20%2 \
    --output="$RUN_ROOT/logs/development_%A_%a.out" \
    --error="$RUN_ROOT/logs/development_%A_%a.err" \
    --export=ALL,FF3D_DEVELOPMENT_CONFIRMED=1,FF3D_BENCHMARK_ROOT="$BENCHMARK_ROOT",FF3D_BENCHMARK_COMMIT="$BENCHMARK_COMMIT",FF3D_BASE_SIF="$BASE_SIF",FF3D_ENV_ROOT="$ENV_ROOT",FF3D_CHECKPOINT="$CHECKPOINT",FF3D_TREEBENCH_PYTHON="$TREEBENCH_PYTHON",FF3D_RUN_ROOT="$RUN_ROOT",FF3D_MANIFEST="$MANIFEST" \
    "$METHOD_ROOT/slurm/run_published_pretrained_development.sbatch"
)"
SUMMARY_JOB="$(
  sbatch --parsable \
    --dependency="afterany:$ARRAY_JOB" \
    --output="$RUN_ROOT/logs/summary_%j.out" \
    --error="$RUN_ROOT/logs/summary_%j.err" \
    --export=ALL,FF3D_BENCHMARK_ROOT="$BENCHMARK_ROOT",FF3D_BENCHMARK_COMMIT="$BENCHMARK_COMMIT",FF3D_TREEBENCH_PYTHON="$TREEBENCH_PYTHON",FF3D_RUN_ROOT="$RUN_ROOT",FF3D_MANIFEST="$MANIFEST" \
    "$METHOD_ROOT/slurm/summarise_published_pretrained_development.sbatch"
)"
{
  printf 'FF3D_WORKFLOW=%q\n' "published_pretrained_development"
  printf 'FF3D_RUN_ID=%q\n' "$RUN_ID"
  printf 'FF3D_RUN_ROOT=%q\n' "$RUN_ROOT"
  printf 'FF3D_JOB_IDS=%q\n' "$ARRAY_JOB,$SUMMARY_JOB"
  printf 'FF3D_ARRAY_JOB=%q\n' "$ARRAY_JOB"
  printf 'FF3D_SUMMARY_JOB=%q\n' "$SUMMARY_JOB"
  printf 'FF3D_EXPECTED_TASKS_ROOT=%q\n' "$RUN_ROOT/tasks"
  printf 'FF3D_EXPECTED_TASK_COUNT=%q\n' "21"
  printf 'FF3D_EXPECTED_FILES=%q\n' "$RUN_ROOT/development_manifest.json|$RUN_ROOT/manual_alignment_confirmation.json|$RUN_ROOT/summary/summary.json|$RUN_ROOT/summary/per_plot_metrics.csv|$RUN_ROOT/summary/retention_manifest.json|$RUN_ROOT/summary/summary.complete|$RUN_ROOT/development.complete"
  printf 'FF3D_BENCHMARK_COMMIT=%q\n' "$BENCHMARK_COMMIT"
} > "$STATE_FILE"
echo "run_id=$RUN_ID"
echo "state_file=$STATE_FILE"
echo "array_job=$ARRAY_JOB"
echo "summary_job=$SUMMARY_JOB"
echo "cancel_command=scancel $ARRAY_JOB $SUMMARY_JOB"
echo "scope=21 development plots; held-out access false"
echo "array_concurrency=2"
if [[ "${FF3D_NO_WATCH:-0}" != "1" ]]; then
  exec bash "$METHOD_ROOT/slurm/monitor_workflow.sh" \
    "$STATE_FILE" --watch "${FF3D_MONITOR_SECONDS:-30}"
fi
