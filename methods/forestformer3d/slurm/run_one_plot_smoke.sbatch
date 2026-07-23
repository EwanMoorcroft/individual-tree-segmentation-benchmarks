#!/bin/bash -l
#SBATCH --job-name=ff3d_dev_smoke
#SBATCH --partition=gpu-a100-lowbig
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=12
#SBATCH --mem=128G
#SBATCH --gres=gpu:a100:1

set -euo pipefail

: "${FF3D_SMOKE_CONFIRMED:?Missing FF3D_SMOKE_CONFIRMED}"
: "${FF3D_BENCHMARK_ROOT:?Missing FF3D_BENCHMARK_ROOT}"
: "${FF3D_BENCHMARK_COMMIT:?Missing FF3D_BENCHMARK_COMMIT}"
: "${FF3D_BASE_SIF:?Missing FF3D_BASE_SIF}"
: "${FF3D_ENV_ROOT:?Missing FF3D_ENV_ROOT}"
: "${FF3D_CHECKPOINT:?Missing FF3D_CHECKPOINT}"
: "${FF3D_DATASET_ROOT:?Missing FF3D_DATASET_ROOT}"
: "${FF3D_TREEBENCH_PYTHON:?Missing FF3D_TREEBENCH_PYTHON}"
: "${FF3D_RUN_ROOT:?Missing FF3D_RUN_ROOT}"

[[ "$FF3D_SMOKE_CONFIRMED" == "1" ]]
cd "$FF3D_BENCHMARK_ROOT"
test "$(git rev-parse HEAD)" = "$FF3D_BENCHMARK_COMMIT"
test "$(git branch --show-current)" = "method/forestformer3d"
test -z "$(git status --porcelain)"
test -f "$FF3D_ENV_ROOT/environment_build.complete"
test ! -e "$FF3D_ENV_ROOT/environment_build.incomplete"
test -f "$FF3D_DATASET_ROOT/data_split_metadata.csv"
test -f "$FF3D_DATASET_ROOT/CULS/plot_1_annotated.las"

echo "host=$(hostname)"
echo "benchmark_commit=$FF3D_BENCHMARK_COMMIT"
echo "split=development"
echo "plot=CULS/plot_1_annotated.las"
echo "held_out_access=false"
nvidia-smi -L

"$FF3D_TREEBENCH_PYTHON" - <<'PY'
import laspy
if laspy.__version__ != "2.7.0":
    raise RuntimeError(f"Expected laspy 2.7.0, found {laspy.__version__}")
PY
"$FF3D_TREEBENCH_PYTHON" -m pip freeze \
  > "$FF3D_RUN_ROOT/treebench_pip_freeze.txt"
"$FF3D_TREEBENCH_PYTHON" \
  methods/forestformer3d/scripts/data/prepare_one_plot_smoke.py \
  --dataset-root "$FF3D_DATASET_ROOT" \
  --output-root "$FF3D_RUN_ROOT/staged_input" \
  --expected-point-count 1816672 \
  --expected-reference-tree-count 6 \
  > "$FF3D_RUN_ROOT/input_preparation_stdout.json"

module purge
module load apptainer/1.3.6

run_case() {
  local case_name="$1"
  apptainer exec --nv --cleanenv \
    --bind "$FF3D_BENCHMARK_ROOT:/benchmark:ro" \
    --bind "$FF3D_ENV_ROOT:/ff3d_environment:ro" \
    --bind "$FF3D_CHECKPOINT:/inputs/epoch_3000_fix.pth:ro" \
    --bind "$FF3D_RUN_ROOT/staged_input:/ff3d_input:ro" \
    --bind "$FF3D_RUN_ROOT/raw:/ff3d_output" \
    --env PATH="/ff3d_environment/venv/bin:/ff3d_environment/toolchain/bin:/usr/local/cuda/bin:/opt/conda/bin:/usr/bin:/bin" \
    --env LD_LIBRARY_PATH="/ff3d_environment/toolchain/lib:/usr/local/cuda/lib64" \
    --env PYTHONPATH="/ff3d_environment/source/ForestFormer3D" \
    --env PYTHONDONTWRITEBYTECODE=1 \
    --env PYTHONHASHSEED=3407 \
    --env OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-12}" \
    --env CUBLAS_WORKSPACE_CONFIG=:4096:8 \
    "$FF3D_BASE_SIF" \
    /usr/bin/env \
    LD_LIBRARY_PATH="/ff3d_environment/toolchain/lib:/usr/local/cuda/lib64:/.singularity.d/libs" \
    /ff3d_environment/venv/bin/python \
    /benchmark/methods/forestformer3d/scripts/runtime/run_official_test.py \
      --source-root /ff3d_environment/source/ForestFormer3D \
      --config /ff3d_environment/source/ForestFormer3D/configs/oneformer3d_qs_radius16_qp300_2many.py \
      --checkpoint /inputs/epoch_3000_fix.pth \
      --data-root /ff3d_input \
      --ann-file "${case_name}.pkl" \
      --work-dir "/ff3d_output/$case_name"
}

mkdir "$FF3D_RUN_ROOT/raw"
run_case reference
run_case dummy

apptainer exec --cleanenv \
  --bind "$FF3D_BENCHMARK_ROOT:/benchmark:ro" \
  --bind "$FF3D_ENV_ROOT:/ff3d_environment:ro" \
  --bind "$FF3D_RUN_ROOT/staged_input:/ff3d_input:ro" \
  --bind "$FF3D_RUN_ROOT/raw:/ff3d_output:ro" \
  --bind "$FF3D_RUN_ROOT:/run" \
  --env PATH="/ff3d_environment/venv/bin:/opt/conda/bin:/usr/bin:/bin" \
  --env PYTHONDONTWRITEBYTECODE=1 \
  "$FF3D_BASE_SIF" \
  /usr/bin/env \
  LD_LIBRARY_PATH="/ff3d_environment/toolchain/lib:/usr/local/cuda/lib64:/.singularity.d/libs" \
  /ff3d_environment/venv/bin/python \
  /benchmark/methods/forestformer3d/scripts/evaluation/validate_one_plot_smoke.py \
    --reference-ply /ff3d_output/reference/forestformer3d_smoke_test_final_results.ply \
    --dummy-ply /ff3d_output/dummy/forestformer3d_smoke_test_final_results.ply \
    --sidecar /ff3d_input/evaluation_sidecar.npz \
    --output-root /run/validation

sha256sum \
  "$FF3D_RUN_ROOT/raw/reference/forestformer3d_smoke_test_final_results.ply" \
  "$FF3D_RUN_ROOT/raw/dummy/forestformer3d_smoke_test_final_results.ply" \
  "$FF3D_RUN_ROOT/validation/forestformer3d_smoke_predictions.npz" \
  > "$FF3D_RUN_ROOT/artifact_sha256.txt"
touch "$FF3D_RUN_ROOT/one_plot_smoke.complete"
echo "status=forestformer3d-development-one-plot-smoke-passed"
