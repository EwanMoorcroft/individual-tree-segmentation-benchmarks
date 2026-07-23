#!/bin/bash -l
#SBATCH --job-name=ff3d_dev_preflight
#SBATCH --partition=nodes
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

set -euo pipefail

: "${FF3D_BENCHMARK_ROOT:?Missing FF3D_BENCHMARK_ROOT}"
: "${FF3D_BENCHMARK_COMMIT:?Missing FF3D_BENCHMARK_COMMIT}"
: "${FF3D_DATASET_ROOT:?Missing FF3D_DATASET_ROOT}"
: "${FF3D_TREEBENCH_PYTHON:?Missing FF3D_TREEBENCH_PYTHON}"
: "${FF3D_PREFLIGHT_ROOT:?Missing FF3D_PREFLIGHT_ROOT}"

cd "$FF3D_BENCHMARK_ROOT"
test "$(git rev-parse HEAD)" = "$FF3D_BENCHMARK_COMMIT"
test "$(git branch --show-current)" = "method/forestformer3d"
test -z "$(git status --porcelain)"
test ! -e "$FF3D_PREFLIGHT_ROOT/development_manifest.json"

"$FF3D_TREEBENCH_PYTHON" \
  methods/forestformer3d/scripts/data/prepare_development_manifest.py \
  --dataset-root "$FF3D_DATASET_ROOT" \
  --output "$FF3D_PREFLIGHT_ROOT/development_manifest.json"
sha256sum "$FF3D_PREFLIGHT_ROOT/development_manifest.json" \
  > "$FF3D_PREFLIGHT_ROOT/development_manifest.sha256"
touch "$FF3D_PREFLIGHT_ROOT/preflight.complete"
echo "status=forestformer3d-development-preflight-complete"
echo "held_out_access=false"
