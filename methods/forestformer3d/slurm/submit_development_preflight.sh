#!/usr/bin/env bash

set -euo pipefail

if [[ "${FF3D_PREFLIGHT_CONFIRMED:-0}" != "1" ]]; then
  echo "Set FF3D_PREFLIGHT_CONFIRMED=1 to freeze the development-only manifest." >&2
  exit 2
fi
BENCHMARK_ROOT="${FF3D_BENCHMARK_ROOT:-$(pwd)}"
RUNTIME_ROOT="${FF3D_RUNTIME_ROOT:-$HOME/fastscratch/forestformer3d}"
DATASET_ROOT="${FF3D_DATASET_ROOT:-$HOME/data/datasets/for_instance/FORinstance_dataset}"
TREEBENCH_PYTHON="${FF3D_TREEBENCH_PYTHON:-$HOME/fastscratch/venvs/treebench/bin/python}"
METHOD_ROOT="$BENCHMARK_ROOT/methods/forestformer3d"
TIMESTAMP="$(date -u +%Y%m%dT%H%M%S)"
RUN_ID="forestformer3d__for-instance__published-pretrained__not-applicable__development-preflight__${TIMESTAMP}"
PREFLIGHT_ROOT="$RUNTIME_ROOT/runs/development-preflight/$RUN_ID"
STATE_FILE="$RUNTIME_ROOT/state/${RUN_ID}.env"

cd "$BENCHMARK_ROOT"
test "$(git branch --show-current)" = "method/forestformer3d"
test -z "$(git status --porcelain)"
BENCHMARK_COMMIT="$(git rev-parse HEAD)"
test -x "$TREEBENCH_PYTHON"
test -f "$DATASET_ROOT/data_split_metadata.csv"
test ! -e "$PREFLIGHT_ROOT"
test ! -e "$STATE_FILE"
mkdir -p "$PREFLIGHT_ROOT/logs" "$(dirname "$STATE_FILE")"
JOB="$(
  sbatch --parsable \
    --output="$PREFLIGHT_ROOT/logs/preflight_%j.out" \
    --error="$PREFLIGHT_ROOT/logs/preflight_%j.err" \
    --export=ALL,FF3D_BENCHMARK_ROOT="$BENCHMARK_ROOT",FF3D_BENCHMARK_COMMIT="$BENCHMARK_COMMIT",FF3D_DATASET_ROOT="$DATASET_ROOT",FF3D_TREEBENCH_PYTHON="$TREEBENCH_PYTHON",FF3D_PREFLIGHT_ROOT="$PREFLIGHT_ROOT" \
    "$METHOD_ROOT/slurm/prepare_development_preflight.sbatch"
)"
{
  printf 'FF3D_WORKFLOW=%q\n' "development_preflight"
  printf 'FF3D_RUN_ID=%q\n' "$RUN_ID"
  printf 'FF3D_RUN_ROOT=%q\n' "$PREFLIGHT_ROOT"
  printf 'FF3D_JOB_IDS=%q\n' "$JOB"
  printf 'FF3D_EXPECTED_FILES=%q\n' "$PREFLIGHT_ROOT/development_manifest.json|$PREFLIGHT_ROOT/development_manifest.sha256|$PREFLIGHT_ROOT/preflight.complete"
  printf 'FF3D_BENCHMARK_COMMIT=%q\n' "$BENCHMARK_COMMIT"
} > "$STATE_FILE"
echo "run_id=$RUN_ID"
echo "state_file=$STATE_FILE"
echo "preflight_job=$JOB"
echo "cancel_command=scancel $JOB"
echo "scope=21 development plots; held-out access false"
if [[ "${FF3D_NO_WATCH:-0}" != "1" ]]; then
  exec bash "$METHOD_ROOT/slurm/monitor_workflow.sh" \
    "$STATE_FILE" --watch "${FF3D_MONITOR_SECONDS:-30}"
fi
