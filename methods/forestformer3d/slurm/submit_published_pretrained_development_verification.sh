#!/usr/bin/env bash

set -euo pipefail

if [[ "${FF3D_DEVELOPMENT_VERIFICATION_CONFIRMED:-0}" != "1" ]]; then
  echo "Set FF3D_DEVELOPMENT_VERIFICATION_CONFIRMED=1 to re-hash the completed development run." >&2
  exit 2
fi
: "${FF3D_SOURCE_RUN_ROOT:?Set FF3D_SOURCE_RUN_ROOT to the completed development run}"

BENCHMARK_ROOT="${FF3D_BENCHMARK_ROOT:-$(pwd)}"
RUNTIME_ROOT="${FF3D_RUNTIME_ROOT:-$HOME/fastscratch/forestformer3d}"
TREEBENCH_PYTHON="${FF3D_TREEBENCH_PYTHON:-$HOME/fastscratch/venvs/treebench/bin/python}"
METHOD_ROOT="$BENCHMARK_ROOT/methods/forestformer3d"

cd "$BENCHMARK_ROOT"
test "$(git branch --show-current)" = "method/forestformer3d"
test -z "$(git status --porcelain)"
BENCHMARK_COMMIT="$(git rev-parse HEAD)"
test -x "$TREEBENCH_PYTHON"
test -f "$FF3D_SOURCE_RUN_ROOT/development.complete"
SOURCE_RUN_ID="$(basename "$FF3D_SOURCE_RUN_ROOT")"
TIMESTAMP="$(date -u +%Y%m%dT%H%M%S)"
RUN_ID="${SOURCE_RUN_ID}__verification__${TIMESTAMP}"
VERIFICATION_ROOT="$RUNTIME_ROOT/runs/development-verification/$RUN_ID"
STATE_FILE="$RUNTIME_ROOT/state/${RUN_ID}.env"
test ! -e "$VERIFICATION_ROOT"
test ! -e "$STATE_FILE"
mkdir -p "$(dirname "$VERIFICATION_ROOT")" "$(dirname "$STATE_FILE")" \
  "$RUNTIME_ROOT/logs"

JOB="$(
  sbatch --parsable \
    --output="$RUNTIME_ROOT/logs/${RUN_ID}_%j.out" \
    --error="$RUNTIME_ROOT/logs/${RUN_ID}_%j.err" \
    --export=ALL,FF3D_BENCHMARK_ROOT="$BENCHMARK_ROOT",FF3D_BENCHMARK_COMMIT="$BENCHMARK_COMMIT",FF3D_TREEBENCH_PYTHON="$TREEBENCH_PYTHON",FF3D_SOURCE_RUN_ROOT="$FF3D_SOURCE_RUN_ROOT",FF3D_VERIFICATION_ROOT="$VERIFICATION_ROOT" \
    "$METHOD_ROOT/slurm/verify_published_pretrained_development.sbatch"
)"
{
  printf 'FF3D_WORKFLOW=%q\n' "published_pretrained_development_verification"
  printf 'FF3D_RUN_ID=%q\n' "$RUN_ID"
  printf 'FF3D_RUN_ROOT=%q\n' "$VERIFICATION_ROOT"
  printf 'FF3D_SOURCE_RUN_ROOT=%q\n' "$FF3D_SOURCE_RUN_ROOT"
  printf 'FF3D_JOB_IDS=%q\n' "$JOB"
  printf 'FF3D_EXPECTED_FILES=%q\n' "$VERIFICATION_ROOT/verification.json|$VERIFICATION_ROOT/verification.sha256|$VERIFICATION_ROOT/verification.complete"
  printf 'FF3D_BENCHMARK_COMMIT=%q\n' "$BENCHMARK_COMMIT"
} > "$STATE_FILE"
echo "run_id=$RUN_ID"
echo "source_run_id=$SOURCE_RUN_ID"
echo "state_file=$STATE_FILE"
echo "verification_job=$JOB"
echo "cancel_command=scancel $JOB"
echo "scope=immutable 21-plot development run; held-out access false"
echo "resources=nodes partition, 4 CPUs, 32 GiB, 4 hours"
echo "runtime_estimate=10-60 minutes for 16.6 GB re-hash; queue waiting separate"
if [[ "${FF3D_NO_WATCH:-0}" != "1" ]]; then
  exec bash "$METHOD_ROOT/slurm/monitor_workflow.sh" \
    "$STATE_FILE" --watch "${FF3D_MONITOR_SECONDS:-30}"
fi
