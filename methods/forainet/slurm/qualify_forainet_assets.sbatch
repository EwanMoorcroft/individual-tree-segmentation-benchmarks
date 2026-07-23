#!/usr/bin/env bash
#SBATCH --job-name=forai_qualify
#SBATCH --partition=gpu-a-lowsmall
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=logs/forai_qualify_%j.out
#SBATCH --error=logs/forai_qualify_%j.err

set -euo pipefail

: "${FORAINET_BENCHMARK_ROOT:?set FORAINET_BENCHMARK_ROOT}"
: "${FORAINET_UPSTREAM_ROOT:?set FORAINET_UPSTREAM_ROOT}"
: "${FORAINET_CHECKPOINT:?set FORAINET_CHECKPOINT}"
: "${FORAINET_IMAGE:?set FORAINET_IMAGE}"
: "${FORAINET_QUALIFICATION_ROOT:?set FORAINET_QUALIFICATION_ROOT}"
: "${FORAINET_EXPECTED_BENCHMARK_COMMIT:?set FORAINET_EXPECTED_BENCHMARK_COMMIT}"

test "$(git -C "$FORAINET_BENCHMARK_ROOT" rev-parse HEAD)" = \
  "$FORAINET_EXPECTED_BENCHMARK_COMMIT"
test -z "$(git -C "$FORAINET_BENCHMARK_ROOT" status --porcelain)"
test -f "$FORAINET_IMAGE"
test ! -e "$FORAINET_QUALIFICATION_ROOT"

mkdir -p "$FORAINET_QUALIFICATION_ROOT"
module load apptainer/1.3.6

benchmark_root="$(readlink -f "$FORAINET_BENCHMARK_ROOT")"
upstream_root="$(readlink -f "$FORAINET_UPSTREAM_ROOT")"
checkpoint="$(readlink -f "$FORAINET_CHECKPOINT")"
image="$(readlink -f "$FORAINET_IMAGE")"
qualification_root="$(readlink -f "$FORAINET_QUALIFICATION_ROOT")"

python "$benchmark_root/methods/forainet/scripts/provenance/verify_forainet_assets.py" \
  --upstream-root "$upstream_root" \
  --checkpoint "$checkpoint" \
  --output-json "$qualification_root/assets.json"

sha256sum "$image" > "$qualification_root/image.sha256"

apptainer exec --nv \
  --bind "$upstream_root:$upstream_root:ro" \
  --bind "$checkpoint:$checkpoint:ro" \
  --bind "$benchmark_root:$benchmark_root:ro" \
  --bind "$qualification_root:$qualification_root" \
  "$image" \
  python3.8 "$benchmark_root/methods/forainet/scripts/provenance/probe_checkpoint_load.py" \
    --upstream-root "$upstream_root" \
    --checkpoint "$checkpoint" \
    --output-json "$qualification_root/checkpoint_load.json"

python - "$qualification_root" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
files = []
for path in sorted(root.iterdir()):
    if path.name == "retention.json" or not path.is_file():
        continue
    files.append(
        {
            "name": path.name,
            "size_bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    )
(root / "retention.json").write_text(
    json.dumps({"status": "complete", "files": files}, indent=2, sort_keys=True)
    + "\n",
    encoding="utf-8",
)
PY
