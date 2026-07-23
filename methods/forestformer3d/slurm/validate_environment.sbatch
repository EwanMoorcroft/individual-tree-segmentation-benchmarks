#!/bin/bash -l
#SBATCH --job-name=ff3d_validate_env
#SBATCH --partition=gpu-a100-lowbig
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:a100:1

set -euo pipefail

: "${FF3D_BENCHMARK_ROOT:?Missing FF3D_BENCHMARK_ROOT}"
: "${FF3D_BENCHMARK_COMMIT:?Missing FF3D_BENCHMARK_COMMIT}"
: "${FF3D_BASE_SIF:?Missing FF3D_BASE_SIF}"
: "${FF3D_BASE_SIF_SHA256:?Missing FF3D_BASE_SIF_SHA256}"
: "${FF3D_ENV_ROOT:?Missing FF3D_ENV_ROOT}"
: "${FF3D_CHECKPOINT:?Missing FF3D_CHECKPOINT}"
: "${FF3D_CHECKPOINT_SHA256:?Missing FF3D_CHECKPOINT_SHA256}"
: "${FF3D_RUN_ROOT:?Missing FF3D_RUN_ROOT}"

cd "$FF3D_BENCHMARK_ROOT"
test "$(git rev-parse HEAD)" = "$FF3D_BENCHMARK_COMMIT"
test "$(git branch --show-current)" = "method/forestformer3d"
test -z "$(git status --porcelain)"
test -f "$FF3D_BASE_SIF"
test -f "$FF3D_ENV_ROOT/environment_build.complete"
test ! -e "$FF3D_ENV_ROOT/environment_build.incomplete"
test -f "$FF3D_CHECKPOINT"
echo "$FF3D_BASE_SIF_SHA256  $FF3D_BASE_SIF" \
  | sha256sum --check --status
echo "$FF3D_CHECKPOINT_SHA256  $FF3D_CHECKPOINT" \
  | sha256sum --check --status

module purge
module load apptainer/1.3.6

printf 'host=%s\n' "$(hostname)"
nvidia-smi -L
sha256sum "$FF3D_BASE_SIF"
sha256sum "$FF3D_CHECKPOINT"

apptainer inspect --json "$FF3D_BASE_SIF" \
  > "$FF3D_RUN_ROOT/runtime_base_sif_inspect.json"
apptainer exec --nv --cleanenv \
  --bind "$FF3D_BENCHMARK_ROOT:/benchmark:ro" \
  --bind "$FF3D_ENV_ROOT:/ff3d_environment:ro" \
  --bind "$FF3D_CHECKPOINT:/inputs/epoch_3000_fix.pth:ro" \
  --bind "$FF3D_RUN_ROOT:/run" \
  --env PATH="/ff3d_environment/venv/bin:/ff3d_environment/toolchain/bin:/usr/local/cuda/bin:/opt/conda/bin:/usr/bin:/bin" \
  --env LD_LIBRARY_PATH="/ff3d_environment/toolchain/lib:/usr/local/cuda/lib64" \
  --env PYTHONPATH="/ff3d_environment/source/ForestFormer3D" \
  --env PYTHONDONTWRITEBYTECODE=1 \
  "$FF3D_BASE_SIF" \
  /usr/bin/env \
  LD_LIBRARY_PATH="/ff3d_environment/toolchain/lib:/usr/local/cuda/lib64:/.singularity.d/libs" \
  /ff3d_environment/venv/bin/python \
  /benchmark/methods/forestformer3d/scripts/runtime/validate_environment.py \
    --environment-root /ff3d_environment \
    --source-root /ff3d_environment/source/ForestFormer3D \
    --checkpoint /inputs/epoch_3000_fix.pth \
    --output /run/environment_validation.json \
    --require-cuda

apptainer exec --nv --cleanenv \
  --bind "$FF3D_ENV_ROOT:/ff3d_environment:ro" \
  --env PATH="/ff3d_environment/venv/bin:/opt/conda/bin:/usr/bin:/bin" \
  --env PYTHONDONTWRITEBYTECODE=1 \
  "$FF3D_BASE_SIF" \
  /usr/bin/env \
  LD_LIBRARY_PATH="/ff3d_environment/toolchain/lib:/usr/local/cuda/lib64:/.singularity.d/libs" \
  /ff3d_environment/venv/bin/python -m pip freeze \
  > "$FF3D_RUN_ROOT/pip_freeze.txt"

touch "$FF3D_RUN_ROOT/environment_validation.complete"
echo "status=forestformer3d-environment-validated"
