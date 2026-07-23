#!/bin/bash -l
#SBATCH --job-name=ff3d_build_env
#SBATCH --partition=nodes
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

set -euo pipefail

: "${FF3D_BUILD_CONFIRMED:?Set FF3D_BUILD_CONFIRMED=1 through the submit wrapper}"
: "${FF3D_BENCHMARK_ROOT:?Missing FF3D_BENCHMARK_ROOT}"
: "${FF3D_BENCHMARK_COMMIT:?Missing FF3D_BENCHMARK_COMMIT}"
: "${FF3D_ROOTLESS_BUILDER:?Missing FF3D_ROOTLESS_BUILDER}"
: "${FF3D_BUILD_INPUT_SHA256:?Missing FF3D_BUILD_INPUT_SHA256}"
: "${FF3D_BASE_SIF:?Missing FF3D_BASE_SIF}"
: "${FF3D_BASE_SIF_SHA256:?Missing FF3D_BASE_SIF_SHA256}"
: "${FF3D_ENV_ROOT:?Missing FF3D_ENV_ROOT}"
: "${FF3D_RUN_ROOT:?Missing FF3D_RUN_ROOT}"

if [[ "$FF3D_BUILD_CONFIRMED" != "1" ]]; then
  echo "ForestFormer3D environment build was not confirmed." >&2
  exit 2
fi

cd "$FF3D_BENCHMARK_ROOT"
test "$(git rev-parse HEAD)" = "$FF3D_BENCHMARK_COMMIT"
test "$(git branch --show-current)" = "method/forestformer3d"
test -z "$(git status --porcelain)"
test "$(sha256sum "$FF3D_ROOTLESS_BUILDER" | cut -d ' ' -f 1)" = \
  "$FF3D_BUILD_INPUT_SHA256"

module purge
module load apptainer/1.3.6

test -f "$FF3D_BASE_SIF"
echo "$FF3D_BASE_SIF_SHA256  $FF3D_BASE_SIF" | sha256sum --check --status
test ! -e "$FF3D_ENV_ROOT"
mkdir "$FF3D_ENV_ROOT"
touch "$FF3D_ENV_ROOT/environment_build.incomplete"

printf 'host=%s\n' "$(hostname)"
printf 'apptainer=%s\n' "$(apptainer --version)"
printf 'benchmark_commit=%s\n' "$FF3D_BENCHMARK_COMMIT"
printf 'build_input_sha256=%s\n' "$FF3D_BUILD_INPUT_SHA256"
printf 'base_sif_sha256=%s\n' "$FF3D_BASE_SIF_SHA256"

apptainer exec --cleanenv \
  --bind "$FF3D_BENCHMARK_ROOT:/benchmark:ro" \
  --bind "$FF3D_ENV_ROOT:$FF3D_ENV_ROOT" \
  --env FF3D_ENV_ROOT="$FF3D_ENV_ROOT" \
  "$FF3D_BASE_SIF" \
  bash /benchmark/methods/forestformer3d/scripts/runtime/build_rootless_environment.sh

test -f "$FF3D_ENV_ROOT/pip_freeze.txt"
test -f "$FF3D_ENV_ROOT/conda_explicit.txt"
test -f "$FF3D_ENV_ROOT/forestformer3d_commit.txt"
rm "$FF3D_ENV_ROOT/environment_build.incomplete"
touch "$FF3D_ENV_ROOT/environment_build.complete"

sha256sum "$FF3D_BASE_SIF" > "$FF3D_RUN_ROOT/base_sif_sha256.txt"
apptainer inspect --json "$FF3D_BASE_SIF" \
  > "$FF3D_RUN_ROOT/base_sif_inspect.json"
stat -c '%n|%s|%y' "$FF3D_BASE_SIF" \
  > "$FF3D_RUN_ROOT/base_sif_metadata.txt"
cp "$FF3D_ENV_ROOT/pip_freeze.txt" "$FF3D_RUN_ROOT/pip_freeze_build.txt"
cp "$FF3D_ENV_ROOT/conda_explicit.txt" "$FF3D_RUN_ROOT/conda_explicit.txt"
printf '%s\n' "$FF3D_BENCHMARK_COMMIT" \
  > "$FF3D_RUN_ROOT/benchmark_commit.txt"
printf '%s\n' "$FF3D_BUILD_INPUT_SHA256" \
  > "$FF3D_RUN_ROOT/build_input_sha256.txt"
printf '%s\n' "$FF3D_ENV_ROOT" \
  > "$FF3D_RUN_ROOT/environment_root.txt"
echo "status=forestformer3d-rootless-environment-built"
