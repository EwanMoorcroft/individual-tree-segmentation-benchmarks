#!/usr/bin/env bash
#SBATCH --job-name=forai_dev
#SBATCH --partition=gpu-a-lowsmall
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --output=logs/forai_dev_%A_%a.out
#SBATCH --error=logs/forai_dev_%A_%a.err

set -euo pipefail
trap 'status=$?; printf "ForAINet development task failed at line %s: %s\n" "$LINENO" "$BASH_COMMAND" >&2; exit "$status"' ERR

: "${FORAINET_BENCHMARK_ROOT:?set FORAINET_BENCHMARK_ROOT}"
: "${FORAINET_EXPECTED_BENCHMARK_COMMIT:?set FORAINET_EXPECTED_BENCHMARK_COMMIT}"
: "${FORAINET_UPSTREAM_ROOT:?set FORAINET_UPSTREAM_ROOT}"
: "${FORAINET_CHECKPOINT:?set FORAINET_CHECKPOINT}"
: "${FORAINET_IMAGE:?set FORAINET_IMAGE}"
: "${FORAINET_QUALIFICATION_ROOT:?set FORAINET_QUALIFICATION_ROOT}"
: "${FORAINET_DATASET_ROOT:?set FORAINET_DATASET_ROOT}"
: "${FORAINET_DEVELOPMENT_ROOT:?set FORAINET_DEVELOPMENT_ROOT}"
: "${FORAINET_DEVELOPMENT_RUN_ID:?set FORAINET_DEVELOPMENT_RUN_ID}"
: "${SLURM_ARRAY_TASK_ID:?development route requires a Slurm array task}"

if [[ "${FORAINET_DEVELOPMENT_CONFIRMED:-0}" != "1" ]]; then
  echo "Refusing full development inference without confirmation."
  exit 2
fi
if ((SLURM_ARRAY_TASK_ID < 0 || SLURM_ARRAY_TASK_ID > 20)); then
  echo "development task index is outside 0..20."
  exit 2
fi

test "$(git -C "$FORAINET_BENCHMARK_ROOT" rev-parse HEAD)" = \
  "$FORAINET_EXPECTED_BENCHMARK_COMMIT"
test -z "$(git -C "$FORAINET_BENCHMARK_ROOT" status --porcelain)"
manifest_csv="$FORAINET_DEVELOPMENT_ROOT/manifest/development.csv"
manifest_json="$FORAINET_DEVELOPMENT_ROOT/manifest/development.json"
test -s "$manifest_csv"
test -s "$manifest_json"

task_environment="$(
  python3 \
    "$FORAINET_BENCHMARK_ROOT/methods/forainet/scripts/provenance/resolve_development_task.py" \
    --manifest-csv "$manifest_csv" \
    --task-index "$SLURM_ARRAY_TASK_ID"
)"
eval "$task_environment"
: "${FORAINET_TASK_RELATIVE_PATH:?task path did not resolve}"
: "${FORAINET_TASK_SOURCE_SHA256:?task hash did not resolve}"
: "${FORAINET_TASK_POINT_COUNT:?task point count did not resolve}"
source_las="$FORAINET_DATASET_ROOT/$FORAINET_TASK_RELATIVE_PATH"
test -r "$source_las"

plot_name="$(printf 'task_%03d' "$SLURM_ARRAY_TASK_ID")"
plot_parent="$FORAINET_DEVELOPMENT_ROOT/plots"
plot_root="$plot_parent/$plot_name"
mkdir -p "$plot_parent"
mkdir "$plot_root"

expected_upstream="5fe600ae8f2fe913ae8740f475f0261a702f2a72"
expected_checkpoint="97c03ce81621dc4193e55d2ca2294861b1f4421c94d192799e5fe031f9d35861"
expected_image="ad0df684209014c52421dc213cd0e15ddbb47214c00fac264e829f68dc17812d"
test "$(git -C "$FORAINET_UPSTREAM_ROOT" rev-parse HEAD)" = "$expected_upstream"
test -z "$(git -C "$FORAINET_UPSTREAM_ROOT" status --porcelain)"
test "$(sha256sum "$FORAINET_CHECKPOINT" | awk '{print $1}')" = \
  "$expected_checkpoint"
actual_image_sha256="$(sha256sum "$FORAINET_IMAGE" | awk '{print $1}')"
test "$actual_image_sha256" = "$expected_image"
test -f "$FORAINET_QUALIFICATION_ROOT/checkpoint_load.json"

module load apptainer/1.3.6
benchmark_root="$(readlink -f "$FORAINET_BENCHMARK_ROOT")"
benchmark_git_common_raw="$(
  git -C "$benchmark_root" rev-parse --git-common-dir
)"
case "$benchmark_git_common_raw" in
  /*) benchmark_git_common="$benchmark_git_common_raw" ;;
  *) benchmark_git_common="$(readlink -f "$benchmark_root/$benchmark_git_common_raw")" ;;
esac
upstream_root="$(readlink -f "$FORAINET_UPSTREAM_ROOT")"
checkpoint="$(readlink -f "$FORAINET_CHECKPOINT")"
image="$(readlink -f "$FORAINET_IMAGE")"
dataset_root="$(readlink -f "$FORAINET_DATASET_ROOT")"
qualification_root="$(readlink -f "$FORAINET_QUALIFICATION_ROOT")"
plot_root="$(readlink -f "$plot_root")"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-12}"

apptainer exec --nv --cleanenv \
  --env "OMP_NUM_THREADS=$OMP_NUM_THREADS" \
  --bind "$benchmark_root:$benchmark_root:ro" \
  --bind "$benchmark_git_common:$benchmark_git_common:ro" \
  --bind "$upstream_root:$upstream_root:ro" \
  --bind "$checkpoint:$checkpoint:ro" \
  --bind "$dataset_root:$dataset_root:ro" \
  --bind "$qualification_root:$qualification_root:ro" \
  --bind "$plot_root:$plot_root" \
  --pwd "$plot_root" \
  "$image" python3.8 \
    "$benchmark_root/methods/forainet/scripts/runtime/run_for_instance_smoke.py" \
    --benchmark-root "$benchmark_root" \
    --benchmark-commit "$FORAINET_EXPECTED_BENCHMARK_COMMIT" \
    --upstream-root "$upstream_root" \
    --checkpoint "$checkpoint" \
    --source-las "$dataset_root/$FORAINET_TASK_RELATIVE_PATH" \
    --split-metadata "$dataset_root/data_split_metadata.csv" \
    --relative-path "$FORAINET_TASK_RELATIVE_PATH" \
    --run-root "$plot_root" \
    --run-id "$FORAINET_DEVELOPMENT_RUN_ID" \
    --image-sha256 "$actual_image_sha256" \
    --route development \
    --development-task-index "$SLURM_ARRAY_TASK_ID" \
    --expected-source-sha256 "$FORAINET_TASK_SOURCE_SHA256" \
    --expected-point-count "$FORAINET_TASK_POINT_COUNT"

test -f "$plot_root/final_gate.json"
test -f "$plot_root/evaluation/metrics.json"
test -f "$plot_root/retention/manifest.json"
