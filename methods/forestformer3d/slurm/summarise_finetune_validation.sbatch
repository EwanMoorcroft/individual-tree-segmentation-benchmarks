#!/bin/bash -l
#SBATCH --job-name=ff3d_ft_select
#SBATCH --partition=nodes
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

set -Eeuo pipefail

: "${FF3D_BENCHMARK_ROOT:?Missing FF3D_BENCHMARK_ROOT}"
: "${FF3D_BENCHMARK_COMMIT:?Missing FF3D_BENCHMARK_COMMIT}"
: "${FF3D_TREEBENCH_PYTHON:?Missing FF3D_TREEBENCH_PYTHON}"
: "${FF3D_RUN_ROOT:?Missing FF3D_RUN_ROOT}"
: "${FF3D_VALIDATION_ROOT:?Missing FF3D_VALIDATION_ROOT}"
: "${FF3D_SOURCE_RUN_ROOT:?Missing FF3D_SOURCE_RUN_ROOT}"

cd "$FF3D_BENCHMARK_ROOT"
test "$(git rev-parse HEAD)" = "$FF3D_BENCHMARK_COMMIT"
test "$(git branch --show-current)" = "method/forestformer3d"
test -z "$(git status --porcelain)"
"$FF3D_TREEBENCH_PYTHON" \
  methods/forestformer3d/scripts/evaluation/summarise_finetune_validation.py \
  --run-root "$FF3D_RUN_ROOT" \
  --validation-root "$FF3D_VALIDATION_ROOT" \
  --source-run-root "$FF3D_SOURCE_RUN_ROOT" \
  --output-root "$FF3D_VALIDATION_ROOT/selection" \
  --benchmark-commit "$FF3D_BENCHMARK_COMMIT"
touch "$FF3D_VALIDATION_ROOT/fine_tune_validation.complete"
echo "status=forestformer3d-finetune-validation-selection-complete"
echo "held_out_access=false"
