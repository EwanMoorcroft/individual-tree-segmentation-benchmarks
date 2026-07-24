#!/bin/bash -l
#SBATCH --job-name=ff3d_ft_smoke
#SBATCH --partition=gpu-a100-lowbig
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=12
#SBATCH --mem=128G
#SBATCH --gres=gpu:a100:1

set -euo pipefail

: "${FF3D_FINETUNE_SMOKE_CONFIRMED:?Missing FF3D_FINETUNE_SMOKE_CONFIRMED}"
: "${FF3D_BENCHMARK_ROOT:?Missing FF3D_BENCHMARK_ROOT}"
: "${FF3D_BENCHMARK_COMMIT:?Missing FF3D_BENCHMARK_COMMIT}"
: "${FF3D_BASE_SIF:?Missing FF3D_BASE_SIF}"
: "${FF3D_BASE_SIF_SHA256:?Missing FF3D_BASE_SIF_SHA256}"
: "${FF3D_ENV_ROOT:?Missing FF3D_ENV_ROOT}"
: "${FF3D_CHECKPOINT:?Missing FF3D_CHECKPOINT}"
: "${FF3D_RUN_ROOT:?Missing FF3D_RUN_ROOT}"

[[ "$FF3D_FINETUNE_SMOKE_CONFIRMED" == "1" ]]
cd "$FF3D_BENCHMARK_ROOT"
test "$(git rev-parse HEAD)" = "$FF3D_BENCHMARK_COMMIT"
test "$(git branch --show-current)" = "method/forestformer3d"
test -z "$(git status --porcelain)"
test -f "$FF3D_ENV_ROOT/environment_build.complete"
test -f "$FF3D_RUN_ROOT/fine_tune_preparation.complete"
test ! -e "$FF3D_RUN_ROOT/training_smoke"
test ! -e "$FF3D_RUN_ROOT/fine_tune_initialization_smoke.complete"
echo "$FF3D_BASE_SIF_SHA256  $FF3D_BASE_SIF" | sha256sum --check --status

echo "host=$(hostname)"
echo "benchmark_commit=$FF3D_BENCHMARK_COMMIT"
echo "split=development_training"
echo "smoke_examples=1 smoke_epochs=1 optimizer_steps=1"
echo "held_out_access=false"
nvidia-smi -L
touch "$FF3D_RUN_ROOT/fine_tune_initialization_smoke.started"
trap 'rc=$?; printf "%s\n" "$rc" > "$FF3D_RUN_ROOT/fine_tune_initialization_smoke.failed"; exit "$rc"' ERR

module purge
module load apptainer/1.3.6

run_in_environment() {
  apptainer exec --nv --cleanenv \
    --bind "$FF3D_BENCHMARK_ROOT:/benchmark:ro" \
    --bind "$FF3D_ENV_ROOT:/ff3d_environment:ro" \
    --bind "$FF3D_CHECKPOINT:/inputs/epoch_3000_fix.pth:ro" \
    --bind "$FF3D_RUN_ROOT:/ff3d_run" \
    --env PATH="/ff3d_environment/venv/bin:/ff3d_environment/toolchain/bin:/usr/local/cuda/bin:/opt/conda/bin:/usr/bin:/bin" \
    --env LD_LIBRARY_PATH="/ff3d_environment/toolchain/lib:/usr/local/cuda/lib64" \
    --env PYTHONPATH="/ff3d_environment/source/ForestFormer3D" \
    --env PYTHONDONTWRITEBYTECODE=1 \
    --env PYTHONHASHSEED=42 \
    --env OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-12}" \
    --env CUBLAS_WORKSPACE_CONFIG=:4096:8 \
    "$FF3D_BASE_SIF" \
    /usr/bin/env \
    LD_LIBRARY_PATH="/ff3d_environment/toolchain/lib:/usr/local/cuda/lib64:/.singularity.d/libs" \
    "$@"
}

run_in_environment \
  /ff3d_environment/venv/bin/python \
  /benchmark/methods/forestformer3d/scripts/runtime/validate_finetune_initialization.py \
    --config /ff3d_run/configs/effective_smoke.py \
    --checkpoint /inputs/epoch_3000_fix.pth \
    --output /ff3d_run/initialization_validation.json

run_in_environment \
  /ff3d_environment/venv/bin/python \
  /ff3d_environment/source/ForestFormer3D/tools/train.py \
    /ff3d_run/configs/effective_smoke.py \
    --work-dir /ff3d_run/training_smoke

test -f "$FF3D_RUN_ROOT/training_smoke/epoch_1.pth"
run_in_environment \
  /ff3d_environment/venv/bin/python \
  /benchmark/methods/forestformer3d/scripts/runtime/validate_finetune_smoke_checkpoint.py \
    --initial-checkpoint /inputs/epoch_3000_fix.pth \
    --trained-checkpoint /ff3d_run/training_smoke/epoch_1.pth \
    --output /ff3d_run/smoke_checkpoint_validation.json

sha256sum \
  "$FF3D_RUN_ROOT/initialization_validation.json" \
  "$FF3D_RUN_ROOT/training_smoke/epoch_1.pth" \
  "$FF3D_RUN_ROOT/smoke_checkpoint_validation.json" \
  > "$FF3D_RUN_ROOT/fine_tune_initialization_smoke.sha256"
touch "$FF3D_RUN_ROOT/fine_tune_initialization_smoke.complete"
trap - ERR
echo "status=forestformer3d-fine-tune-initialization-smoke-complete"
echo "held_out_access=false"
