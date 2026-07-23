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
: "${FF3D_RECIPE:?Missing FF3D_RECIPE}"
: "${FF3D_RECIPE_SHA256:?Missing FF3D_RECIPE_SHA256}"
: "${FF3D_IMAGE:?Missing FF3D_IMAGE}"
: "${FF3D_RUN_ROOT:?Missing FF3D_RUN_ROOT}"
: "${APPTAINER_CACHEDIR:?Missing APPTAINER_CACHEDIR}"
: "${APPTAINER_TMPDIR:?Missing APPTAINER_TMPDIR}"

if [[ "$FF3D_BUILD_CONFIRMED" != "1" ]]; then
  echo "ForestFormer3D environment build was not confirmed." >&2
  exit 2
fi

cd "$FF3D_BENCHMARK_ROOT"
test "$(git rev-parse HEAD)" = "$FF3D_BENCHMARK_COMMIT"
test "$(git branch --show-current)" = "method/forestformer3d"
test -z "$(git status --porcelain)"
echo "$FF3D_RECIPE_SHA256  $FF3D_RECIPE" | sha256sum --check --status

module purge
module load apptainer/1.3.6

test ! -e "$FF3D_IMAGE"
partial_image="${FF3D_IMAGE}.partial.${SLURM_JOB_ID}"
test ! -e "$partial_image"

mkdir -p "$APPTAINER_CACHEDIR" "$APPTAINER_TMPDIR"
printf 'host=%s\n' "$(hostname)"
printf 'apptainer=%s\n' "$(apptainer --version)"
printf 'benchmark_commit=%s\n' "$FF3D_BENCHMARK_COMMIT"
printf 'recipe_sha256=%s\n' "$FF3D_RECIPE_SHA256"

apptainer build --fakeroot "$partial_image" "$FF3D_RECIPE"
mv "$partial_image" "$FF3D_IMAGE"

sha256sum "$FF3D_IMAGE" | tee "$FF3D_RUN_ROOT/image_sha256.txt"
apptainer inspect --json "$FF3D_IMAGE" \
  > "$FF3D_RUN_ROOT/image_inspect.json"
stat -c '%n|%s|%y' "$FF3D_IMAGE" \
  > "$FF3D_RUN_ROOT/image_metadata.txt"
printf '%s\n' "$FF3D_BENCHMARK_COMMIT" \
  > "$FF3D_RUN_ROOT/benchmark_commit.txt"
printf '%s\n' "$FF3D_RECIPE_SHA256" \
  > "$FF3D_RUN_ROOT/recipe_sha256.txt"
echo "status=forestformer3d-image-built"
