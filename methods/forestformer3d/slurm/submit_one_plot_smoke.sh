#!/usr/bin/env bash

set -euo pipefail

if [[ "${FF3D_SMOKE_CONFIRMED:-0}" != "1" ]]; then
  echo "Refusing ForestFormer3D development smoke submission." >&2
  echo "Set FF3D_SMOKE_CONFIRMED=1 after reviewing this one-plot route." >&2
  exit 2
fi

BENCHMARK_ROOT="${FF3D_BENCHMARK_ROOT:-$(pwd)}"
RUNTIME_ROOT="${FF3D_RUNTIME_ROOT:-$HOME/fastscratch/forestformer3d}"
BASE_SIF="${FF3D_BASE_SIF:-$RUNTIME_ROOT/containers/pytorch_1.13.1_cuda11.6_cudnn8_devel.sif}"
ENV_ROOT="${FF3D_ENV_ROOT:-$RUNTIME_ROOT/environments/forestformer3d_6a75c3735e4a_741e13d08e51_20260723T191340}"
CHECKPOINT="${FF3D_CHECKPOINT:-$RUNTIME_ROOT/checkpoints/clean_forestformer/clean_forestformer/epoch_3000_fix.pth}"
DATASET_ROOT="${FF3D_DATASET_ROOT:-$HOME/data/datasets/for_instance/FORinstance_dataset}"
TREEBENCH_PYTHON="${FF3D_TREEBENCH_PYTHON:-$HOME/fastscratch/venvs/treebench/bin/python}"
METHOD_ROOT="$BENCHMARK_ROOT/methods/forestformer3d"
JOB_FILE="$METHOD_ROOT/slurm/run_one_plot_smoke.sbatch"
RUNS_DIR="$RUNTIME_ROOT/runs/development-smoke"
STATE_DIR="$RUNTIME_ROOT/state"

BASE_SIF_SHA256="4a35d5a57c1d57061f899b514329ad8ec2bf74a9ff31d103c0a53a289e07c84f"
CHECKPOINT_SHA256="01037a648596832238ac72ea2f5eef87ceaf5aeb399e56ff4b760ba1ed1c777e"
PIP_FREEZE_SHA256="b48648ead7fe20afad5af9cc1a2a272277dd34988d82a5a1159fd0ac78578456"
CONDA_EXPLICIT_SHA256="2ed0298fc8dbeae38dc0d431c614647af5acde80b2c878b1d12858042c850f71"
SPLIT_METADATA_SHA256="dd64aa338681f8f4166f8d175879a2b0b0158ecf222497ec6f7f0b23bc4fce94"

cd "$BENCHMARK_ROOT"
test "$(git branch --show-current)" = "method/forestformer3d"
test -z "$(git status --porcelain)"
BENCHMARK_COMMIT="$(git rev-parse HEAD)"

test -f "$BASE_SIF"
echo "$BASE_SIF_SHA256  $BASE_SIF" | sha256sum --check --status
test -f "$CHECKPOINT"
echo "$CHECKPOINT_SHA256  $CHECKPOINT" | sha256sum --check --status
test -f "$ENV_ROOT/environment_build.complete"
test ! -e "$ENV_ROOT/environment_build.incomplete"
echo "$PIP_FREEZE_SHA256  $ENV_ROOT/pip_freeze.txt" \
  | sha256sum --check --status
echo "$CONDA_EXPLICIT_SHA256  $ENV_ROOT/conda_explicit.txt" \
  | sha256sum --check --status
test -x "$TREEBENCH_PYTHON"
test -f "$DATASET_ROOT/CULS/plot_1_annotated.las"
echo "$SPLIT_METADATA_SHA256  $DATASET_ROOT/data_split_metadata.csv" \
  | sha256sum --check --status

TIMESTAMP="$(date -u +%Y%m%dT%H%M%S)"
RUN_ID="forestformer3d__for-instance__published-pretrained__not-applicable__development-smoke__${TIMESTAMP}"
RUN_ROOT="$RUNS_DIR/$RUN_ID"
STATE_FILE="$STATE_DIR/${RUN_ID}.env"
test ! -e "$RUN_ROOT"
test ! -e "$STATE_FILE"
mkdir -p "$RUNS_DIR" "$STATE_DIR"
mkdir "$RUN_ROOT"
mkdir "$RUN_ROOT/logs"

SMOKE_JOB="$(
  sbatch --parsable \
    --output="$RUN_ROOT/logs/smoke_%j.out" \
    --error="$RUN_ROOT/logs/smoke_%j.err" \
    --export=ALL,FF3D_SMOKE_CONFIRMED=1,FF3D_BENCHMARK_ROOT="$BENCHMARK_ROOT",FF3D_BENCHMARK_COMMIT="$BENCHMARK_COMMIT",FF3D_BASE_SIF="$BASE_SIF",FF3D_ENV_ROOT="$ENV_ROOT",FF3D_CHECKPOINT="$CHECKPOINT",FF3D_DATASET_ROOT="$DATASET_ROOT",FF3D_TREEBENCH_PYTHON="$TREEBENCH_PYTHON",FF3D_RUN_ROOT="$RUN_ROOT" \
    "$JOB_FILE"
)"

EXPECTED_FILES="$RUN_ROOT/staged_input/input_manifest.json|$RUN_ROOT/staged_input/input_preparation.complete|$RUN_ROOT/raw/reference/forestformer3d_smoke_test.ply|$RUN_ROOT/raw/dummy/forestformer3d_smoke_test.ply|$RUN_ROOT/raw/reference/model_input_fingerprint.json|$RUN_ROOT/raw/dummy/model_input_fingerprint.json|$RUN_ROOT/raw/reference/effective_predict_audit.json|$RUN_ROOT/raw/dummy/effective_predict_audit.json|$RUN_ROOT/validation/smoke_validation.json|$RUN_ROOT/validation/forestformer3d_smoke_predictions.npz|$RUN_ROOT/validation/smoke_validation.complete|$RUN_ROOT/artifact_sha256.txt|$RUN_ROOT/one_plot_smoke.complete"

{
  printf 'FF3D_WORKFLOW=%q\n' "development_one_plot_smoke"
  printf 'FF3D_RUN_ID=%q\n' "$RUN_ID"
  printf 'FF3D_RUN_ROOT=%q\n' "$RUN_ROOT"
  printf 'FF3D_JOB_IDS=%q\n' "$SMOKE_JOB"
  printf 'FF3D_SMOKE_JOB=%q\n' "$SMOKE_JOB"
  printf 'FF3D_BASE_SIF=%q\n' "$BASE_SIF"
  printf 'FF3D_ENV_ROOT=%q\n' "$ENV_ROOT"
  printf 'FF3D_EXPECTED_FILES=%q\n' "$EXPECTED_FILES"
  printf 'FF3D_BENCHMARK_COMMIT=%q\n' "$BENCHMARK_COMMIT"
  printf 'FF3D_CREATED_AT=%q\n' "$(date -Is)"
} > "$STATE_FILE"

echo "run_id=$RUN_ID"
echo "state_file=$STATE_FILE"
echo "smoke_job=$SMOKE_JOB"
echo "cancel_command=scancel $SMOKE_JOB"
echo "scope=development-only CULS/plot_1_annotated.las"

if [[ "${FF3D_NO_WATCH:-0}" != "1" ]]; then
  exec bash "$METHOD_ROOT/slurm/monitor_workflow.sh" \
    "$STATE_FILE" --watch "${FF3D_MONITOR_SECONDS:-30}"
fi
