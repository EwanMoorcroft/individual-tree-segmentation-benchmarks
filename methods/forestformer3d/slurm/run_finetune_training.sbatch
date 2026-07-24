#!/bin/bash -l
#SBATCH --job-name=ff3d_ft_train
#SBATCH --partition=gpu-a100-lowbig
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=12
#SBATCH --mem=128G
#SBATCH --gres=gpu:a100:1

set -Eeuo pipefail

: "${FF3D_FINETUNE_TRAINING_CONFIRMED:?Missing FF3D_FINETUNE_TRAINING_CONFIRMED}"
: "${FF3D_BENCHMARK_ROOT:?Missing FF3D_BENCHMARK_ROOT}"
: "${FF3D_BENCHMARK_COMMIT:?Missing FF3D_BENCHMARK_COMMIT}"
: "${FF3D_BASE_SIF:?Missing FF3D_BASE_SIF}"
: "${FF3D_BASE_SIF_SHA256:?Missing FF3D_BASE_SIF_SHA256}"
: "${FF3D_ENV_ROOT:?Missing FF3D_ENV_ROOT}"
: "${FF3D_CHECKPOINT:?Missing FF3D_CHECKPOINT}"
: "${FF3D_RUN_ROOT:?Missing FF3D_RUN_ROOT}"

[[ "$FF3D_FINETUNE_TRAINING_CONFIRMED" == "1" ]]
cd "$FF3D_BENCHMARK_ROOT"
test "$(git rev-parse HEAD)" = "$FF3D_BENCHMARK_COMMIT"
test "$(git branch --show-current)" = "method/forestformer3d"
test -z "$(git status --porcelain)"
test -f "$FF3D_ENV_ROOT/environment_build.complete"
test -f "$FF3D_RUN_ROOT/fine_tune_preparation.complete"
test -f "$FF3D_RUN_ROOT/fine_tune_initialization_smoke.complete"
test ! -e "$FF3D_RUN_ROOT/training_full"
test ! -e "$FF3D_RUN_ROOT/fine_tune_training.complete"
echo "$FF3D_BASE_SIF_SHA256  $FF3D_BASE_SIF" | sha256sum --check --status

echo "host=$(hostname)"
echo "benchmark_commit=$FF3D_BENCHMARK_COMMIT"
echo "split=development_training"
echo "training_plots=16 validation_plots_read=0 held_out_access=false"
echo "epochs=35 examples_per_epoch=16 micro_batch_size=1 gradient_accumulation=2 effective_batch_size=2 optimizer_steps=280"
echo "checkpoint_epochs=7,14,21,28,35"
nvidia-smi -L
touch "$FF3D_RUN_ROOT/fine_tune_training.started"
trap 'rc=$?; printf "%s\n" "$rc" > "$FF3D_RUN_ROOT/fine_tune_training.failed"; exit "$rc"' ERR

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
  /ff3d_environment/source/ForestFormer3D/tools/train.py \
    /ff3d_run/configs/effective_full.py \
    --work-dir /ff3d_run/training_full

run_in_environment \
  /ff3d_environment/venv/bin/python \
  /benchmark/methods/forestformer3d/scripts/runtime/inventory_finetune_checkpoints.py \
    --initial-checkpoint /inputs/epoch_3000_fix.pth \
    --training-root /ff3d_run/training_full \
    --output /ff3d_run/checkpoint_inventory.json

sha256sum \
  "$FF3D_RUN_ROOT/configs/effective_full.py" \
  "$FF3D_RUN_ROOT/checkpoint_inventory.json" \
  > "$FF3D_RUN_ROOT/fine_tune_training.sha256"
touch "$FF3D_RUN_ROOT/fine_tune_training.complete"
trap - ERR
echo "status=forestformer3d-fine-tune-training-complete"
echo "held_out_access=false"
