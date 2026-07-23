#!/usr/bin/env bash
#SBATCH --job-name=forai_ftval
#SBATCH --partition=gpu-a-lowsmall
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --output=logs/forai_ftval_%A_%a.out
#SBATCH --error=logs/forai_ftval_%A_%a.err

set -euo pipefail
trap 'status=$?; printf "ForAINet fine-tune validation failed at line %s: %s\n" "$LINENO" "$BASH_COMMAND" >&2; exit "$status"' ERR

: "${FORAINET_BENCHMARK_ROOT:?set FORAINET_BENCHMARK_ROOT}"
: "${FORAINET_EXPECTED_BENCHMARK_COMMIT:?set FORAINET_EXPECTED_BENCHMARK_COMMIT}"
: "${FORAINET_UPSTREAM_ROOT:?set FORAINET_UPSTREAM_ROOT}"
: "${FORAINET_IMAGE:?set FORAINET_IMAGE}"
: "${FORAINET_QUALIFICATION_ROOT:?set FORAINET_QUALIFICATION_ROOT}"
: "${FORAINET_DATASET_ROOT:?set FORAINET_DATASET_ROOT}"
: "${FORAINET_FINETUNE_ROOT:?set FORAINET_FINETUNE_ROOT}"
: "${FORAINET_FINETUNE_VALIDATION_ROOT:?set FORAINET_FINETUNE_VALIDATION_ROOT}"
: "${FORAINET_FINETUNE_VALIDATION_RUN_ID:?set FORAINET_FINETUNE_VALIDATION_RUN_ID}"
: "${SLURM_ARRAY_TASK_ID:?validation route requires a Slurm array task}"

if [[ "${FORAINET_FINETUNE_VALIDATION_CONFIRMED:-0}" != "1" ]]; then
  echo "Refusing fine-tune validation without development-only confirmation."
  exit 2
fi
if ((SLURM_ARRAY_TASK_ID < 0 || SLURM_ARRAY_TASK_ID > 24)); then
  echo "fine-tune validation task index is outside 0..24."
  exit 2
fi
test "$(git -C "$FORAINET_BENCHMARK_ROOT" rev-parse HEAD)" = \
  "$FORAINET_EXPECTED_BENCHMARK_COMMIT"
test -z "$(git -C "$FORAINET_BENCHMARK_ROOT" status --porcelain)"
test -f "$FORAINET_FINETUNE_ROOT/full/final_gate.json"
manifest="$FORAINET_FINETUNE_ROOT/finetune_data_manifest.json"
candidate_index="$FORAINET_FINETUNE_ROOT/full/candidates/index.json"
test -s "$manifest"
test -s "$candidate_index"

task_environment="$(
  python3 \
    "$FORAINET_BENCHMARK_ROOT/methods/forainet/scripts/provenance/resolve_finetune_validation_task.py" \
    --finetune-manifest "$manifest" \
    --candidate-index "$candidate_index" \
    --task-index "$SLURM_ARRAY_TASK_ID"
)"
eval "$task_environment"
: "${FORAINET_CANDIDATE_EPOCH:?candidate epoch did not resolve}"
: "${FORAINET_CANDIDATE_FILENAME:?candidate filename did not resolve}"
: "${FORAINET_CANDIDATE_SHA256:?candidate hash did not resolve}"
: "${FORAINET_TASK_RELATIVE_PATH:?validation path did not resolve}"
: "${FORAINET_TASK_SOURCE_SHA256:?validation source hash did not resolve}"
: "${FORAINET_TASK_POINT_COUNT:?validation point count did not resolve}"

checkpoint="$FORAINET_FINETUNE_ROOT/full/candidates/$FORAINET_CANDIDATE_FILENAME"
source_las="$FORAINET_DATASET_ROOT/$FORAINET_TASK_RELATIVE_PATH"
test -r "$checkpoint"
test -r "$source_las"
test "$(sha256sum "$checkpoint" | awk '{print $1}')" = \
  "$FORAINET_CANDIDATE_SHA256"

plot_parent="$FORAINET_FINETUNE_VALIDATION_ROOT/candidates/$(printf 'epoch_%03d' "$FORAINET_CANDIDATE_EPOCH")"
plot_root="$plot_parent/$(printf 'plot_%03d' "$FORAINET_VALIDATION_PLOT_OFFSET")"
mkdir -p "$plot_parent"
mkdir "$plot_root"

expected_upstream="5fe600ae8f2fe913ae8740f475f0261a702f2a72"
expected_image="ad0df684209014c52421dc213cd0e15ddbb47214c00fac264e829f68dc17812d"
test "$(git -C "$FORAINET_UPSTREAM_ROOT" rev-parse HEAD)" = "$expected_upstream"
test -z "$(git -C "$FORAINET_UPSTREAM_ROOT" status --porcelain)"
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
checkpoint="$(readlink -f "$checkpoint")"
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
    --checkpoint-kind fine_tuned_on_dev \
    --expected-checkpoint-sha256 "$FORAINET_CANDIDATE_SHA256" \
    --expected-checkpoint-epoch "$FORAINET_CANDIDATE_EPOCH" \
    --source-las "$dataset_root/$FORAINET_TASK_RELATIVE_PATH" \
    --split-metadata "$dataset_root/data_split_metadata.csv" \
    --relative-path "$FORAINET_TASK_RELATIVE_PATH" \
    --run-root "$plot_root" \
    --run-id "$FORAINET_FINETUNE_VALIDATION_RUN_ID" \
    --image-sha256 "$actual_image_sha256" \
    --route finetune_validation \
    --development-task-index "$FORAINET_TASK_DEVELOPMENT_INDEX" \
    --expected-source-sha256 "$FORAINET_TASK_SOURCE_SHA256" \
    --expected-point-count "$FORAINET_TASK_POINT_COUNT"

test -f "$plot_root/final_gate.json"
test -f "$plot_root/evaluation/metrics.json"
test -f "$plot_root/retention/manifest.json"
