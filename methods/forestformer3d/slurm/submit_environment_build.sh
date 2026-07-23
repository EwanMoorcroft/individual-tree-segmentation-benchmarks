#!/usr/bin/env bash

set -euo pipefail

if [[ "${FF3D_ENVIRONMENT_BUILD_CONFIRMED:-0}" != "1" ]]; then
  echo "Refusing ForestFormer3D environment build." >&2
  echo "Set FF3D_ENVIRONMENT_BUILD_CONFIRMED=1 after reviewing the recipe." >&2
  exit 2
fi

BENCHMARK_ROOT="${FF3D_BENCHMARK_ROOT:-$(pwd)}"
CHECKPOINT="${FF3D_CHECKPOINT:-$HOME/fastscratch/forestformer3d/checkpoints/clean_forestformer/clean_forestformer/epoch_3000_fix.pth}"
CHECKPOINT_SHA256="01037a648596832238ac72ea2f5eef87ceaf5aeb399e56ff4b760ba1ed1c777e"
METHOD_ROOT="$BENCHMARK_ROOT/methods/forestformer3d"
RECIPE="$METHOD_ROOT/container/forestformer3d.def"
ROOT="${FF3D_RUNTIME_ROOT:-$HOME/fastscratch/forestformer3d}"
CONTAINER_DIR="$ROOT/containers"
CACHE_DIR="$ROOT/apptainer_cache"
TMP_DIR="$ROOT/apptainer_tmp"
RUNS_DIR="$ROOT/runs/environment"
STATE_DIR="$ROOT/state"

cd "$BENCHMARK_ROOT"
test "$(git branch --show-current)" = "method/forestformer3d"
test -z "$(git status --porcelain)"
BENCHMARK_COMMIT="$(git rev-parse HEAD)"
RECIPE_SHA256="$(sha256sum "$RECIPE" | cut -d ' ' -f 1)"
RECIPE_SHORT="${RECIPE_SHA256:0:12}"
SOURCE_SHORT="6a75c3735e4a"
TIMESTAMP="$(date -u +%Y%m%dT%H%M%S)"
RUN_ID="forestformer3d__for-instance__published-pretrained__not-applicable__environment__${TIMESTAMP}"
RUN_ROOT="$RUNS_DIR/$RUN_ID"
IMAGE="$CONTAINER_DIR/forestformer3d_${SOURCE_SHORT}_${RECIPE_SHORT}.sif"
STATE_FILE="$STATE_DIR/${RUN_ID}.env"

test -f "$CHECKPOINT"
echo "$CHECKPOINT_SHA256  $CHECKPOINT" | sha256sum --check --status
test ! -e "$RUN_ROOT"
test ! -e "$IMAGE"
test ! -e "$STATE_FILE"

mkdir -p "$CONTAINER_DIR" "$CACHE_DIR" "$TMP_DIR" "$RUNS_DIR" "$STATE_DIR"
mkdir "$RUN_ROOT"
mkdir "$RUN_ROOT/logs"

BUILD_JOB="$(
  sbatch --parsable \
    --output="$RUN_ROOT/logs/build_%j.out" \
    --error="$RUN_ROOT/logs/build_%j.err" \
    --export=ALL,FF3D_BUILD_CONFIRMED=1,FF3D_BENCHMARK_ROOT="$BENCHMARK_ROOT",FF3D_BENCHMARK_COMMIT="$BENCHMARK_COMMIT",FF3D_RECIPE="$RECIPE",FF3D_RECIPE_SHA256="$RECIPE_SHA256",FF3D_IMAGE="$IMAGE",FF3D_RUN_ROOT="$RUN_ROOT",APPTAINER_CACHEDIR="$CACHE_DIR",APPTAINER_TMPDIR="$TMP_DIR" \
    "$METHOD_ROOT/slurm/build_environment.sbatch"
)"

VALIDATE_JOB="$(
  sbatch --parsable \
    --dependency="afterok:$BUILD_JOB" \
    --output="$RUN_ROOT/logs/validate_%j.out" \
    --error="$RUN_ROOT/logs/validate_%j.err" \
    --export=ALL,FF3D_BENCHMARK_ROOT="$BENCHMARK_ROOT",FF3D_BENCHMARK_COMMIT="$BENCHMARK_COMMIT",FF3D_IMAGE="$IMAGE",FF3D_CHECKPOINT="$CHECKPOINT",FF3D_CHECKPOINT_SHA256="$CHECKPOINT_SHA256",FF3D_RUN_ROOT="$RUN_ROOT" \
    "$METHOD_ROOT/slurm/validate_environment.sbatch"
)"

EXPECTED_FILES="$IMAGE|$RUN_ROOT/image_sha256.txt|$RUN_ROOT/image_inspect.json|$RUN_ROOT/environment_validation.json|$RUN_ROOT/pip_freeze.txt|$RUN_ROOT/environment_validation.complete"

{
  printf 'FF3D_WORKFLOW=%q\n' "environment_build"
  printf 'FF3D_RUN_ID=%q\n' "$RUN_ID"
  printf 'FF3D_RUN_ROOT=%q\n' "$RUN_ROOT"
  printf 'FF3D_JOB_IDS=%q\n' "$BUILD_JOB,$VALIDATE_JOB"
  printf 'FF3D_BUILD_JOB=%q\n' "$BUILD_JOB"
  printf 'FF3D_VALIDATE_JOB=%q\n' "$VALIDATE_JOB"
  printf 'FF3D_IMAGE=%q\n' "$IMAGE"
  printf 'FF3D_EXPECTED_FILES=%q\n' "$EXPECTED_FILES"
  printf 'FF3D_BENCHMARK_COMMIT=%q\n' "$BENCHMARK_COMMIT"
  printf 'FF3D_RECIPE_SHA256=%q\n' "$RECIPE_SHA256"
  printf 'FF3D_CREATED_AT=%q\n' "$(date -Is)"
} > "$STATE_FILE"

echo "run_id=$RUN_ID"
echo "state_file=$STATE_FILE"
echo "build_job=$BUILD_JOB"
echo "validate_job=$VALIDATE_JOB"
echo "image=$IMAGE"
echo "cancel_command=scancel $BUILD_JOB $VALIDATE_JOB"

if [[ "${FF3D_NO_WATCH:-0}" != "1" ]]; then
  exec bash "$METHOD_ROOT/slurm/monitor_workflow.sh" \
    "$STATE_FILE" --watch "${FF3D_MONITOR_SECONDS:-30}"
fi
