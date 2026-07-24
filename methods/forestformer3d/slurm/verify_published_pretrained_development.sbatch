#!/bin/bash -l
#SBATCH --job-name=ff3d_dev_verify
#SBATCH --partition=nodes
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

set -euo pipefail

: "${FF3D_BENCHMARK_ROOT:?Missing FF3D_BENCHMARK_ROOT}"
: "${FF3D_BENCHMARK_COMMIT:?Missing FF3D_BENCHMARK_COMMIT}"
: "${FF3D_TREEBENCH_PYTHON:?Missing FF3D_TREEBENCH_PYTHON}"
: "${FF3D_SOURCE_RUN_ROOT:?Missing FF3D_SOURCE_RUN_ROOT}"
: "${FF3D_VERIFICATION_ROOT:?Missing FF3D_VERIFICATION_ROOT}"

cd "$FF3D_BENCHMARK_ROOT"
test "$(git rev-parse HEAD)" = "$FF3D_BENCHMARK_COMMIT"
test "$(git branch --show-current)" = "method/forestformer3d"
test -z "$(git status --porcelain)"
test -f "$FF3D_SOURCE_RUN_ROOT/development.complete"
test ! -e "$FF3D_VERIFICATION_ROOT"

"$FF3D_TREEBENCH_PYTHON" \
  methods/forestformer3d/scripts/evaluation/verify_development_run.py \
  --run-root "$FF3D_SOURCE_RUN_ROOT" \
  --output-root "$FF3D_VERIFICATION_ROOT"
echo "status=forestformer3d-development-verification-complete"
echo "held_out_access=false"
