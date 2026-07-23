#!/usr/bin/env bash
#SBATCH --job-name=forai_build
#SBATCH --partition=nodes
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=logs/forai_build_%j.out
#SBATCH --error=logs/forai_build_%j.err

set -euo pipefail

: "${FORAINET_BENCHMARK_ROOT:?set FORAINET_BENCHMARK_ROOT}"
: "${FORAINET_DEFINITION:?set FORAINET_DEFINITION}"
: "${FORAINET_IMAGE:?set FORAINET_IMAGE}"
: "${FORAINET_IMAGE_BUILD_ROOT:?set FORAINET_IMAGE_BUILD_ROOT}"
: "${FORAINET_TOOLCHAIN_ROOT:?set FORAINET_TOOLCHAIN_ROOT}"
: "${FORAINET_EXPECTED_BENCHMARK_COMMIT:?set FORAINET_EXPECTED_BENCHMARK_COMMIT}"
: "${APPTAINER_CACHEDIR:?set APPTAINER_CACHEDIR}"

expected_definition="$FORAINET_BENCHMARK_ROOT/methods/forainet/containers/forainet-cuda111-a100.def"
requirements="$FORAINET_BENCHMARK_ROOT/methods/forainet/containers/requirements.lock"
apptainer="$FORAINET_TOOLCHAIN_ROOT/bin/apptainer"
test "$FORAINET_DEFINITION" = "$expected_definition"
test -f "$FORAINET_DEFINITION"
test -f "$requirements"
test -x "$apptainer"
test "$("$apptainer" --version)" = "apptainer version 1.3.6-1"
test ! -e "$FORAINET_IMAGE"
test ! -e "$FORAINET_IMAGE_BUILD_ROOT"
test "$(git -C "$FORAINET_BENCHMARK_ROOT" rev-parse HEAD)" = \
  "$FORAINET_EXPECTED_BENCHMARK_COMMIT"
test -z "$(git -C "$FORAINET_BENCHMARK_ROOT" status --porcelain)"

mkdir -p \
  "$FORAINET_IMAGE_BUILD_ROOT" \
  "$APPTAINER_CACHEDIR" \
  "$(dirname "$FORAINET_IMAGE")"

node_tmp="$(mktemp -d "/tmp/forai-build-${SLURM_JOB_ID}-XXXXXXXX")"
cleanup_node_tmp() {
  case "$node_tmp" in
    "/tmp/forai-build-${SLURM_JOB_ID}-"*) rm -rf -- "$node_tmp" ;;
    *) return 1 ;;
  esac
}
trap cleanup_node_tmp EXIT
export APPTAINER_TMPDIR="$node_tmp"

partial_image="$(dirname "$FORAINET_IMAGE")/.forainet-${SLURM_JOB_ID}.partial.sif"
test ! -e "$partial_image"

git -C "$FORAINET_BENCHMARK_ROOT" status --short --branch \
  > "$FORAINET_IMAGE_BUILD_ROOT/benchmark_git_status.txt"
git -C "$FORAINET_BENCHMARK_ROOT" rev-parse HEAD \
  > "$FORAINET_IMAGE_BUILD_ROOT/benchmark_commit.txt"
sha256sum "$FORAINET_DEFINITION" \
  > "$FORAINET_IMAGE_BUILD_ROOT/definition.sha256"
sha256sum "$requirements" \
  > "$FORAINET_IMAGE_BUILD_ROOT/requirements.sha256"
"$apptainer" --version > "$FORAINET_IMAGE_BUILD_ROOT/apptainer_version.txt"

cd "$FORAINET_BENCHMARK_ROOT"
"$apptainer" build --fakeroot "$partial_image" "$FORAINET_DEFINITION"

"$apptainer" test "$partial_image" \
  > "$FORAINET_IMAGE_BUILD_ROOT/container_test.txt"
"$apptainer" inspect --json "$partial_image" \
  > "$FORAINET_IMAGE_BUILD_ROOT/container_inspect.json"
"$apptainer" exec "$partial_image" python3.8 -m pip freeze --all \
  > "$FORAINET_IMAGE_BUILD_ROOT/pip_freeze.txt"
"$apptainer" exec "$partial_image" python3.8 - <<'PY' \
  > "$FORAINET_IMAGE_BUILD_ROOT/import_probe.txt"
import json

import hdbscan
import hydra
import MinkowskiEngine
import numpy
import omegaconf
import torch
import torch_cluster
import torch_geometric
import torch_points_kernels
import torch_scatter
import torch_sparse
import torchsparse

print(json.dumps({
    "cuda_available_on_cpu_build_node": torch.cuda.is_available(),
    "numpy": numpy.__version__,
    "torch": torch.__version__,
}, sort_keys=True))
PY

image_sha256="$(sha256sum "$partial_image" | awk '{print $1}')"
image_size="$(stat -c '%s' "$partial_image")"
printf '%s  %s\n' "$image_sha256" "$FORAINET_IMAGE" \
  > "$FORAINET_IMAGE_BUILD_ROOT/image.sha256"
printf '%s\n' "$image_size" > "$FORAINET_IMAGE_BUILD_ROOT/image_size_bytes.txt"
mv "$partial_image" "$FORAINET_IMAGE"

python - \
  "$FORAINET_IMAGE_BUILD_ROOT/build.json" \
  "$FORAINET_EXPECTED_BENCHMARK_COMMIT" \
  "$FORAINET_IMAGE" \
  "$image_sha256" \
  "$image_size" \
  "$SLURM_JOB_ID" <<'PY'
import json
import sys
from pathlib import Path

output, commit, image, sha256, size, job_id = sys.argv[1:]
payload = {
    "benchmark_commit": commit,
    "build_status": "complete",
    "image": image,
    "image_sha256": sha256,
    "image_size_bytes": int(size),
    "slurm_job_id": job_id,
    "target_cuda_arch": "8.0",
}
Path(output).write_text(
    json.dumps(payload, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
PY
