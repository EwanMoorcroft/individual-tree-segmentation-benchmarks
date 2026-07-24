#!/bin/bash -l
#SBATCH --job-name=ff3d_ft_prepare
#SBATCH --partition=nodes
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

set -euo pipefail

: "${FF3D_FINETUNE_PREPARATION_CONFIRMED:?Missing FF3D_FINETUNE_PREPARATION_CONFIRMED}"
: "${FF3D_BENCHMARK_ROOT:?Missing FF3D_BENCHMARK_ROOT}"
: "${FF3D_BENCHMARK_COMMIT:?Missing FF3D_BENCHMARK_COMMIT}"
: "${FF3D_BASE_SIF:?Missing FF3D_BASE_SIF}"
: "${FF3D_ENV_ROOT:?Missing FF3D_ENV_ROOT}"
: "${FF3D_CHECKPOINT:?Missing FF3D_CHECKPOINT}"
: "${FF3D_CHECKPOINT_SHA256:?Missing FF3D_CHECKPOINT_SHA256}"
: "${FF3D_TREEBENCH_PYTHON:?Missing FF3D_TREEBENCH_PYTHON}"
: "${FF3D_SOURCE_RUN_ROOT:?Missing FF3D_SOURCE_RUN_ROOT}"
: "${FF3D_VERIFICATION_JSON:?Missing FF3D_VERIFICATION_JSON}"
: "${FF3D_RUN_ROOT:?Missing FF3D_RUN_ROOT}"

[[ "$FF3D_FINETUNE_PREPARATION_CONFIRMED" == "1" ]]
cd "$FF3D_BENCHMARK_ROOT"
test "$(git rev-parse HEAD)" = "$FF3D_BENCHMARK_COMMIT"
test "$(git branch --show-current)" = "method/forestformer3d"
test -z "$(git status --porcelain)"
test -f "$FF3D_ENV_ROOT/environment_build.complete"
test -f "$FF3D_SOURCE_RUN_ROOT/development.complete"
test -f "$FF3D_VERIFICATION_JSON"
test ! -e "$FF3D_RUN_ROOT"

PREPARATION_STDOUT="$(mktemp "${TMPDIR:-/tmp}/ff3d_finetune_prepare.XXXXXX")"
trap 'rm -f "$PREPARATION_STDOUT"' EXIT
"$FF3D_TREEBENCH_PYTHON" \
  methods/forestformer3d/scripts/data/prepare_finetune.py \
  --source-run-root "$FF3D_SOURCE_RUN_ROOT" \
  --verification-json "$FF3D_VERIFICATION_JSON" \
  --output-root "$FF3D_RUN_ROOT" \
  --benchmark-commit "$FF3D_BENCHMARK_COMMIT" \
  --checkpoint-sha256 "$FF3D_CHECKPOINT_SHA256" \
  > "$PREPARATION_STDOUT"
mv "$PREPARATION_STDOUT" "$FF3D_RUN_ROOT/preparation_stdout.json"
trap - EXIT

module purge
module load apptainer/1.3.6
apptainer exec --cleanenv \
  --bind "$FF3D_BENCHMARK_ROOT:/benchmark:ro" \
  --bind "$FF3D_ENV_ROOT:/ff3d_environment:ro" \
  --bind "$FF3D_CHECKPOINT:/inputs/epoch_3000_fix.pth:ro" \
  --bind "$FF3D_RUN_ROOT:/ff3d_run" \
  --env PATH="/ff3d_environment/venv/bin:/ff3d_environment/toolchain/bin:/opt/conda/bin:/usr/bin:/bin" \
  --env LD_LIBRARY_PATH="/ff3d_environment/toolchain/lib:/usr/local/cuda/lib64" \
  --env PYTHONPATH="/ff3d_environment/source/ForestFormer3D" \
  --env PYTHONDONTWRITEBYTECODE=1 \
  "$FF3D_BASE_SIF" \
  /usr/bin/env \
  LD_LIBRARY_PATH="/ff3d_environment/toolchain/lib:/usr/local/cuda/lib64:/.singularity.d/libs" \
  /ff3d_environment/venv/bin/python \
  /benchmark/methods/forestformer3d/scripts/runtime/build_finetune_configs.py \
    --upstream-config /ff3d_environment/source/ForestFormer3D/configs/oneformer3d_qs_radius16_qp300_2many.py \
    --output-root /ff3d_run/configs \
    --data-root /ff3d_run/data/ForAINetV2/ \
    --checkpoint /inputs/epoch_3000_fix.pth \
    --run-root /ff3d_run \
  > "$FF3D_RUN_ROOT/config_build_stdout.json"

sha256sum \
  "$FF3D_RUN_ROOT/fine_tune_freeze.json" \
  "$FF3D_RUN_ROOT/fine_tune_split.csv" \
  "$FF3D_RUN_ROOT/configs/effective_smoke.py" \
  "$FF3D_RUN_ROOT/configs/effective_full.py" \
  "$FF3D_RUN_ROOT/configs/config_manifest.json" \
  > "$FF3D_RUN_ROOT/frozen_inputs.sha256"
touch "$FF3D_RUN_ROOT/fine_tune_preparation.complete"
echo "status=forestformer3d-fine-tune-preparation-complete"
echo "training_plots=16 validation_plots=5 held_out_access=false"
