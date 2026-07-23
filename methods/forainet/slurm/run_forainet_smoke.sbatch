#!/usr/bin/env bash
#SBATCH --job-name=forai_smoke
#SBATCH --partition=gpu-a-lowsmall
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --output=logs/forai_smoke_%j.out
#SBATCH --error=logs/forai_smoke_%j.err

set -euo pipefail
trap 'status=$?; printf "ForAINet smoke failed at line %s: %s\n" "$LINENO" "$BASH_COMMAND" >&2; exit "$status"' ERR

: "${FORAINET_BENCHMARK_ROOT:?set FORAINET_BENCHMARK_ROOT}"
: "${FORAINET_EXPECTED_BENCHMARK_COMMIT:?set FORAINET_EXPECTED_BENCHMARK_COMMIT}"
: "${FORAINET_UPSTREAM_ROOT:?set FORAINET_UPSTREAM_ROOT}"
: "${FORAINET_CHECKPOINT:?set FORAINET_CHECKPOINT}"
: "${FORAINET_IMAGE:?set FORAINET_IMAGE}"
: "${FORAINET_QUALIFICATION_ROOT:?set FORAINET_QUALIFICATION_ROOT}"
: "${FORAINET_DATASET_ROOT:?set FORAINET_DATASET_ROOT}"
: "${FORAINET_RUN_ROOT:?set FORAINET_RUN_ROOT}"
: "${FORAINET_RUN_ID:?set FORAINET_RUN_ID}"

if [[ "${FORAINET_SMOKE_CONFIRMED:-0}" != "1" ]]; then
  echo "Refusing development smoke without FORAINET_SMOKE_CONFIRMED=1."
  exit 2
fi

expected_upstream="5fe600ae8f2fe913ae8740f475f0261a702f2a72"
expected_checkpoint="97c03ce81621dc4193e55d2ca2294861b1f4421c94d192799e5fe031f9d35861"
expected_image="4b8835107800c5a368e4073aade1fee5b94e436693e13ab351fe8c2a250d898e"
relative_path="CULS/plot_1_annotated.las"
source_las="$FORAINET_DATASET_ROOT/$relative_path"
split_metadata="$FORAINET_DATASET_ROOT/data_split_metadata.csv"

test "$(git -C "$FORAINET_BENCHMARK_ROOT" rev-parse HEAD)" = \
  "$FORAINET_EXPECTED_BENCHMARK_COMMIT"
test -z "$(git -C "$FORAINET_BENCHMARK_ROOT" status --porcelain)"
test "$(git -C "$FORAINET_UPSTREAM_ROOT" rev-parse HEAD)" = "$expected_upstream"
test -z "$(git -C "$FORAINET_UPSTREAM_ROOT" status --porcelain)"
test -r "$source_las"
test -r "$split_metadata"
test -f "$FORAINET_CHECKPOINT"
test -f "$FORAINET_IMAGE"
test -f "$FORAINET_QUALIFICATION_ROOT/assets.json"
test -f "$FORAINET_QUALIFICATION_ROOT/checkpoint_load.json"
test -f "$FORAINET_QUALIFICATION_ROOT/retention.json"
test ! -e "$FORAINET_RUN_ROOT"

test "$(sha256sum "$FORAINET_CHECKPOINT" | awk '{print $1}')" = \
  "$expected_checkpoint"
actual_image_sha256="$(sha256sum "$FORAINET_IMAGE" | awk '{print $1}')"
test "$actual_image_sha256" = "$expected_image"
python - "$FORAINET_QUALIFICATION_ROOT" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
assets = json.loads((root / "assets.json").read_text())
load = json.loads((root / "checkpoint_load.json").read_text())
retention = json.loads((root / "retention.json").read_text())
if assets.get("status") != "verified":
    raise SystemExit("asset qualification is not verified")
if load.get("status") != "verified" or load.get("compatible_fraction") != 1.0:
    raise SystemExit("checkpoint qualification is not complete")
if retention.get("status") != "complete":
    raise SystemExit("qualification retention is not complete")
PY

split="$(
  awk -F, -v target="$relative_path" '
    NR == 1 {
      for (i = 1; i <= NF; i++) {
        header = $i
        sub(/\r$/, "", header)
        if (header == "path" || header == "relative_path") path_column = i
        if (header == "split") split_column = i
      }
      next
    }
    path_column && split_column && $path_column == target {
      value = $split_column
      sub(/\r$/, "", value)
      print value
    }
  ' "$split_metadata"
)"
test "$split" = "dev"

mkdir "$FORAINET_RUN_ROOT"
module load apptainer/1.3.6

benchmark_root="$(readlink -f "$FORAINET_BENCHMARK_ROOT")"
upstream_root="$(readlink -f "$FORAINET_UPSTREAM_ROOT")"
checkpoint="$(readlink -f "$FORAINET_CHECKPOINT")"
image="$(readlink -f "$FORAINET_IMAGE")"
dataset_root="$(readlink -f "$FORAINET_DATASET_ROOT")"
run_root="$(readlink -f "$FORAINET_RUN_ROOT")"
qualification_root="$(readlink -f "$FORAINET_QUALIFICATION_ROOT")"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-12}"

apptainer exec --nv --cleanenv \
  --env "OMP_NUM_THREADS=$OMP_NUM_THREADS" \
  --bind "$benchmark_root:$benchmark_root:ro" \
  --bind "$upstream_root:$upstream_root:ro" \
  --bind "$checkpoint:$checkpoint:ro" \
  --bind "$dataset_root:$dataset_root:ro" \
  --bind "$qualification_root:$qualification_root:ro" \
  --bind "$run_root:$run_root" \
  --pwd "$run_root" \
  "$image" \
  python3.8 \
    "$benchmark_root/methods/forainet/scripts/runtime/run_for_instance_smoke.py" \
    --benchmark-root "$benchmark_root" \
    --benchmark-commit "$FORAINET_EXPECTED_BENCHMARK_COMMIT" \
    --upstream-root "$upstream_root" \
    --checkpoint "$checkpoint" \
    --source-las "$dataset_root/$relative_path" \
    --split-metadata "$dataset_root/data_split_metadata.csv" \
    --relative-path "$relative_path" \
    --run-root "$run_root" \
    --run-id "$FORAINET_RUN_ID" \
    --image-sha256 "$actual_image_sha256"

test -f "$FORAINET_RUN_ROOT/final_gate.json"
test -f "$FORAINET_RUN_ROOT/evaluation/metrics.json"
test -f "$FORAINET_RUN_ROOT/retention/manifest.json"
