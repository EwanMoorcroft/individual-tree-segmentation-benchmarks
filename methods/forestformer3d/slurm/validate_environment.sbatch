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
: "${FF3D_IMAGE:?Missing FF3D_IMAGE}"
: "${FF3D_CHECKPOINT:?Missing FF3D_CHECKPOINT}"
: "${FF3D_CHECKPOINT_SHA256:?Missing FF3D_CHECKPOINT_SHA256}"
: "${FF3D_RUN_ROOT:?Missing FF3D_RUN_ROOT}"

cd "$FF3D_BENCHMARK_ROOT"
test "$(git rev-parse HEAD)" = "$FF3D_BENCHMARK_COMMIT"
test "$(git branch --show-current)" = "method/forestformer3d"
test -z "$(git status --porcelain)"
test -f "$FF3D_IMAGE"
test -f "$FF3D_CHECKPOINT"
echo "$FF3D_CHECKPOINT_SHA256  $FF3D_CHECKPOINT" \
  | sha256sum --check --status

module purge
module load apptainer/1.3.6

printf 'host=%s\n' "$(hostname)"
nvidia-smi -L
sha256sum "$FF3D_IMAGE"
sha256sum "$FF3D_CHECKPOINT"

apptainer inspect --json "$FF3D_IMAGE" \
  > "$FF3D_RUN_ROOT/runtime_image_inspect.json"
apptainer exec --nv --cleanenv \
  --bind "$FF3D_BENCHMARK_ROOT:/benchmark:ro" \
  --bind "$FF3D_CHECKPOINT:/inputs/epoch_3000_fix.pth:ro" \
  --bind "$FF3D_RUN_ROOT:/run" \
  "$FF3D_IMAGE" \
  python /benchmark/methods/forestformer3d/scripts/runtime/validate_environment.py \
    --checkpoint /inputs/epoch_3000_fix.pth \
    --output /run/environment_validation.json \
    --require-cuda

apptainer exec --nv --cleanenv "$FF3D_IMAGE" \
  python -m pip freeze \
  > "$FF3D_RUN_ROOT/pip_freeze.txt"

touch "$FF3D_RUN_ROOT/environment_validation.complete"
echo "status=forestformer3d-environment-validated"
