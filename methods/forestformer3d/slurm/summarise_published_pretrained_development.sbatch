#!/bin/bash -l
#SBATCH --job-name=ff3d_dev_summary
#SBATCH --partition=nodes
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

set -euo pipefail

: "${FF3D_BENCHMARK_ROOT:?Missing FF3D_BENCHMARK_ROOT}"
: "${FF3D_BENCHMARK_COMMIT:?Missing FF3D_BENCHMARK_COMMIT}"
: "${FF3D_TREEBENCH_PYTHON:?Missing FF3D_TREEBENCH_PYTHON}"
: "${FF3D_RUN_ROOT:?Missing FF3D_RUN_ROOT}"
: "${FF3D_MANIFEST:?Missing FF3D_MANIFEST}"

cd "$FF3D_BENCHMARK_ROOT"
test "$(git rev-parse HEAD)" = "$FF3D_BENCHMARK_COMMIT"
test -z "$(git status --porcelain)"
"$FF3D_TREEBENCH_PYTHON" \
  methods/forestformer3d/scripts/evaluation/summarise_development.py \
  --manifest "$FF3D_MANIFEST" \
  --run-root "$FF3D_RUN_ROOT" \
  --output-root "$FF3D_RUN_ROOT/summary" \
  --run-id "$(basename "$FF3D_RUN_ROOT")" \
  --benchmark-commit "$FF3D_BENCHMARK_COMMIT"
touch "$FF3D_RUN_ROOT/development.complete"
echo "status=forestformer3d-published-pretrained-development-complete"
echo "held_out_access=false"
