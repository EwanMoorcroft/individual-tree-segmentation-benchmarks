#!/usr/bin/env bash

set -euo pipefail

if [[ "${FF3D_FINETUNE_VALIDATION_CONFIRMED:-0}" != "1" ]]; then
  echo "Set FF3D_FINETUNE_VALIDATION_CONFIRMED=1 after training inventory passes." >&2
  exit 2
fi
: "${FF3D_RUN_ROOT:?Set FF3D_RUN_ROOT to the completed fine-tuning run}"

BENCHMARK_ROOT="${FF3D_BENCHMARK_ROOT:-$(pwd)}"
RUNTIME_ROOT="${FF3D_RUNTIME_ROOT:-$HOME/fastscratch/forestformer3d}"
BASE_SIF="${FF3D_BASE_SIF:-$RUNTIME_ROOT/containers/pytorch_1.13.1_cuda11.6_cudnn8_devel.sif}"
ENV_ROOT="${FF3D_ENV_ROOT:-$RUNTIME_ROOT/environments/forestformer3d_6a75c3735e4a_741e13d08e51_20260723T191340}"
TREEBENCH_PYTHON="${FF3D_TREEBENCH_PYTHON:-$HOME/fastscratch/venvs/treebench/bin/python}"
METHOD_ROOT="$BENCHMARK_ROOT/methods/forestformer3d"
BASE_SIF_SHA256="4a35d5a57c1d57061f899b514329ad8ec2bf74a9ff31d103c0a53a289e07c84f"

cd "$BENCHMARK_ROOT"
test "$(git branch --show-current)" = "method/forestformer3d"
test -z "$(git status --porcelain)"
BENCHMARK_COMMIT="$(git rev-parse HEAD)"
echo "$BASE_SIF_SHA256  $BASE_SIF" | sha256sum --check --status
test -f "$ENV_ROOT/environment_build.complete"
test -f "$FF3D_RUN_ROOT/fine_tune_training.complete"
test ! -f "$FF3D_RUN_ROOT/fine_tune_training.failed"
"$TREEBENCH_PYTHON" -c \
  'import json,sys; f=json.load(open(sys.argv[1])); i=json.load(open(sys.argv[2])); assert f["benchmark_commit"] == sys.argv[3]; assert f["split"]["held_out_access"] is False; assert i["status"] == "complete"; assert i["epochs"] == [7,14,21,28,35]; assert i["held_out_access"] is False' \
  "$FF3D_RUN_ROOT/fine_tune_freeze.json" \
  "$FF3D_RUN_ROOT/checkpoint_inventory.json" \
  "$BENCHMARK_COMMIT"

SOURCE_RUN_ID="$("$TREEBENCH_PYTHON" -c \
  'import json,sys; print(json.load(open(sys.argv[1]))["source_development_run_id"])' \
  "$FF3D_RUN_ROOT/fine_tune_freeze.json")"
SOURCE_RUN_ROOT="$RUNTIME_ROOT/runs/development/$SOURCE_RUN_ID"
test -f "$SOURCE_RUN_ROOT/development.complete"

TIMESTAMP="$(date -u +%Y%m%dT%H%M%S)"
VALIDATION_ID="$(basename "$FF3D_RUN_ROOT")__checkpoint-sweep__development-validation__${TIMESTAMP}"
VALIDATION_ROOT="$RUNTIME_ROOT/runs/development-finetune-validation/$VALIDATION_ID"
STATE_FILE="$RUNTIME_ROOT/state/${VALIDATION_ID}.env"
test ! -e "$VALIDATION_ROOT"
test ! -e "$STATE_FILE"
mkdir -p "$VALIDATION_ROOT/logs" "$VALIDATION_ROOT/tasks" "$(dirname "$STATE_FILE")"
cp "$FF3D_RUN_ROOT/fine_tune_freeze.json" "$VALIDATION_ROOT/fine_tune_freeze.json"
cp "$FF3D_RUN_ROOT/fine_tune_split.csv" "$VALIDATION_ROOT/fine_tune_split.csv"
cp "$FF3D_RUN_ROOT/checkpoint_inventory.json" "$VALIDATION_ROOT/checkpoint_inventory.json"

ARRAY_JOB="$(
  sbatch --parsable \
    --array=0-24%2 \
    --output="$VALIDATION_ROOT/logs/validation_%A_%a.out" \
    --error="$VALIDATION_ROOT/logs/validation_%A_%a.err" \
    --export=ALL,FF3D_BENCHMARK_ROOT="$BENCHMARK_ROOT",FF3D_BENCHMARK_COMMIT="$BENCHMARK_COMMIT",FF3D_BASE_SIF="$BASE_SIF",FF3D_ENV_ROOT="$ENV_ROOT",FF3D_TREEBENCH_PYTHON="$TREEBENCH_PYTHON",FF3D_RUN_ROOT="$FF3D_RUN_ROOT",FF3D_VALIDATION_ROOT="$VALIDATION_ROOT",FF3D_SOURCE_RUN_ROOT="$SOURCE_RUN_ROOT" \
    "$METHOD_ROOT/slurm/run_finetune_validation.sbatch"
)"
SUMMARY_JOB="$(
  sbatch --parsable \
    --dependency="afterany:$ARRAY_JOB" \
    --output="$VALIDATION_ROOT/logs/selection_%j.out" \
    --error="$VALIDATION_ROOT/logs/selection_%j.err" \
    --export=ALL,FF3D_BENCHMARK_ROOT="$BENCHMARK_ROOT",FF3D_BENCHMARK_COMMIT="$BENCHMARK_COMMIT",FF3D_TREEBENCH_PYTHON="$TREEBENCH_PYTHON",FF3D_RUN_ROOT="$FF3D_RUN_ROOT",FF3D_VALIDATION_ROOT="$VALIDATION_ROOT",FF3D_SOURCE_RUN_ROOT="$SOURCE_RUN_ROOT" \
    "$METHOD_ROOT/slurm/summarise_finetune_validation.sbatch"
)"
{
  printf 'FF3D_WORKFLOW=%q\n' "fine_tune_validation"
  printf 'FF3D_RUN_ID=%q\n' "$VALIDATION_ID"
  printf 'FF3D_RUN_ROOT=%q\n' "$VALIDATION_ROOT"
  printf 'FF3D_JOB_IDS=%q\n' "$ARRAY_JOB,$SUMMARY_JOB"
  printf 'FF3D_ARRAY_JOB=%q\n' "$ARRAY_JOB"
  printf 'FF3D_SUMMARY_JOB=%q\n' "$SUMMARY_JOB"
  printf 'FF3D_EXPECTED_TASKS_ROOT=%q\n' "$VALIDATION_ROOT/tasks"
  printf 'FF3D_EXPECTED_TASK_COUNT=%q\n' "25"
  printf 'FF3D_EXPECTED_FILES=%q\n' "$VALIDATION_ROOT/selection/selected_checkpoint.json|$VALIDATION_ROOT/selection/checkpoint_metrics.csv|$VALIDATION_ROOT/selection/per_plot_metrics.csv|$VALIDATION_ROOT/selection/retention_manifest.json|$VALIDATION_ROOT/selection/selection.complete|$VALIDATION_ROOT/fine_tune_validation.complete"
  printf 'FF3D_BENCHMARK_COMMIT=%q\n' "$BENCHMARK_COMMIT"
} > "$STATE_FILE"
echo "validation_id=$VALIDATION_ID"
echo "state_file=$STATE_FILE"
echo "array_job=$ARRAY_JOB"
echo "summary_job=$SUMMARY_JOB"
echo "cancel_command=scancel $ARRAY_JOB $SUMMARY_JOB"
echo "scope=5 checkpoints x 5 frozen development-validation plots; held-out access false"
echo "array_concurrency=2"
if [[ "${FF3D_NO_WATCH:-0}" != "1" ]]; then
  exec bash "$METHOD_ROOT/slurm/monitor_workflow.sh" \
    "$STATE_FILE" --watch "${FF3D_MONITOR_SECONDS:-30}"
fi
