#!/bin/bash -l
#SBATCH --job-name=ff3d_dev
#SBATCH --partition=gpu-a100-lowbig
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=12
#SBATCH --mem=128G
#SBATCH --gres=gpu:a100:1

set -euo pipefail

: "${FF3D_DEVELOPMENT_CONFIRMED:?Missing FF3D_DEVELOPMENT_CONFIRMED}"
: "${FF3D_BENCHMARK_ROOT:?Missing FF3D_BENCHMARK_ROOT}"
: "${FF3D_BENCHMARK_COMMIT:?Missing FF3D_BENCHMARK_COMMIT}"
: "${FF3D_BASE_SIF:?Missing FF3D_BASE_SIF}"
: "${FF3D_ENV_ROOT:?Missing FF3D_ENV_ROOT}"
: "${FF3D_CHECKPOINT:?Missing FF3D_CHECKPOINT}"
: "${FF3D_TREEBENCH_PYTHON:?Missing FF3D_TREEBENCH_PYTHON}"
: "${FF3D_RUN_ROOT:?Missing FF3D_RUN_ROOT}"
: "${FF3D_MANIFEST:?Missing FF3D_MANIFEST}"
: "${SLURM_ARRAY_TASK_ID:?Missing SLURM_ARRAY_TASK_ID}"

[[ "$FF3D_DEVELOPMENT_CONFIRMED" == "1" ]]
cd "$FF3D_BENCHMARK_ROOT"
test "$(git rev-parse HEAD)" = "$FF3D_BENCHMARK_COMMIT"
test "$(git branch --show-current)" = "method/forestformer3d"
test -z "$(git status --porcelain)"
test -f "$FF3D_ENV_ROOT/environment_build.complete"
test -f "$FF3D_MANIFEST"

TASK_JSON="$("$FF3D_TREEBENCH_PYTHON" -c \
  'import json,sys; p=json.load(open(sys.argv[1])); print(json.dumps(p["plots"][int(sys.argv[2])]))' \
  "$FF3D_MANIFEST" "$SLURM_ARRAY_TASK_ID")"
SAFE_PLOT_ID="$("$FF3D_TREEBENCH_PYTHON" -c \
  'import json,sys; print(json.loads(sys.argv[1])["safe_plot_id"])' "$TASK_JSON")"
RELATIVE_PATH="$("$FF3D_TREEBENCH_PYTHON" -c \
  'import json,sys; print(json.loads(sys.argv[1])["relative_path"])' "$TASK_JSON")"
TASK_ROOT="$FF3D_RUN_ROOT/tasks/$SAFE_PLOT_ID"
test ! -e "$TASK_ROOT"
mkdir "$TASK_ROOT"
trap 'rc=$?; printf "%s\n" "$rc" > "$TASK_ROOT/task.failed"; exit "$rc"' ERR

echo "host=$(hostname)"
echo "benchmark_commit=$FF3D_BENCHMARK_COMMIT"
echo "task_index=$SLURM_ARRAY_TASK_ID"
echo "plot=$RELATIVE_PATH"
echo "split=development"
echo "held_out_access=false"
nvidia-smi -L

"$FF3D_TREEBENCH_PYTHON" \
  methods/forestformer3d/scripts/data/prepare_development_plot.py \
  --manifest "$FF3D_MANIFEST" \
  --task-index "$SLURM_ARRAY_TASK_ID" \
  --output-root "$TASK_ROOT/staged_input" \
  > "$TASK_ROOT/input_preparation_stdout.json"

module purge
module load apptainer/1.3.6
apptainer exec --nv --cleanenv \
  --bind "$FF3D_BENCHMARK_ROOT:/benchmark:ro" \
  --bind "$FF3D_ENV_ROOT:/ff3d_environment:ro" \
  --bind "$FF3D_CHECKPOINT:/inputs/epoch_3000_fix.pth:ro" \
  --bind "$TASK_ROOT/staged_input:/ff3d_input:ro" \
  --bind "$TASK_ROOT:/ff3d_task" \
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
    --ann-file reference.pkl \
    --work-dir /ff3d_task/raw

apptainer exec --cleanenv \
  --bind "$FF3D_BENCHMARK_ROOT:/benchmark:ro" \
  --bind "$FF3D_ENV_ROOT:/ff3d_environment:ro" \
  --bind "$TASK_ROOT:/task" \
  --env PATH="/ff3d_environment/venv/bin:/opt/conda/bin:/usr/bin:/bin" \
  --env PYTHONDONTWRITEBYTECODE=1 \
  "$FF3D_BASE_SIF" \
  /usr/bin/env \
  LD_LIBRARY_PATH="/ff3d_environment/toolchain/lib:/usr/local/cuda/lib64:/.singularity.d/libs" \
  /ff3d_environment/venv/bin/python \
  /benchmark/methods/forestformer3d/scripts/evaluation/validate_development_plot.py \
    --raw-ply /task/raw/forestformer3d_development_test.ply \
    --sidecar /task/staged_input/evaluation_sidecar.npz \
    --input-manifest /task/staged_input/input_manifest.json \
    --fingerprint /task/raw/model_input_fingerprint.json \
    --audit /task/raw/effective_predict_audit.json \
    --output-root /task/validation

"$FF3D_TREEBENCH_PYTHON" \
  methods/forestformer3d/scripts/evaluation/evaluate_development_plot.py \
  --prediction "$TASK_ROOT/validation/predictions.npz" \
  --validation "$TASK_ROOT/validation/validation.json" \
  --resource "$TASK_ROOT/raw/resource_usage.json" \
  --output-root "$TASK_ROOT/evaluation" \
  --run-id "$(basename "$FF3D_RUN_ROOT")" \
  --benchmark-commit "$FF3D_BENCHMARK_COMMIT"

find "$TASK_ROOT" -type f ! -name task.complete ! -name task.failed \
  -print0 | sort -z | xargs -0 sha256sum > "$TASK_ROOT/artifact_sha256.txt"
touch "$TASK_ROOT/task.complete"
trap - ERR
echo "status=forestformer3d-development-task-complete"
