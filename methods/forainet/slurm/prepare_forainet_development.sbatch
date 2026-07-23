#!/usr/bin/env bash
#SBATCH --job-name=forai_devprep
#SBATCH --partition=nodes
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=logs/forai_devprep_%j.out
#SBATCH --error=logs/forai_devprep_%j.err

set -euo pipefail
trap 'status=$?; printf "ForAINet development preparation failed at line %s: %s\n" "$LINENO" "$BASH_COMMAND" >&2; exit "$status"' ERR

: "${FORAINET_BENCHMARK_ROOT:?set FORAINET_BENCHMARK_ROOT}"
: "${FORAINET_EXPECTED_BENCHMARK_COMMIT:?set FORAINET_EXPECTED_BENCHMARK_COMMIT}"
: "${FORAINET_IMAGE:?set FORAINET_IMAGE}"
: "${FORAINET_DATASET_ROOT:?set FORAINET_DATASET_ROOT}"
: "${FORAINET_DEVELOPMENT_ROOT:?set FORAINET_DEVELOPMENT_ROOT}"

test "$(git -C "$FORAINET_BENCHMARK_ROOT" rev-parse HEAD)" = \
  "$FORAINET_EXPECTED_BENCHMARK_COMMIT"
test -z "$(git -C "$FORAINET_BENCHMARK_ROOT" status --porcelain)"
test -d "$FORAINET_DEVELOPMENT_ROOT"
test -z "$(find "$FORAINET_DEVELOPMENT_ROOT" -mindepth 1 -maxdepth 1 -print -quit)"
test -r "$FORAINET_DATASET_ROOT/data_split_metadata.csv"
test -f "$FORAINET_IMAGE"

benchmark_root="$(readlink -f "$FORAINET_BENCHMARK_ROOT")"
dataset_root="$(readlink -f "$FORAINET_DATASET_ROOT")"
development_root="$(readlink -f "$FORAINET_DEVELOPMENT_ROOT")"
image="$(readlink -f "$FORAINET_IMAGE")"
accepted_smoke="$benchmark_root/methods/forainet/examples/accepted_development_smoke_20260723.json"
test -f "$accepted_smoke"
mkdir "$development_root/manifest"

module load apptainer/1.3.6
apptainer exec --cleanenv \
  --bind "$benchmark_root:$benchmark_root:ro" \
  --bind "$dataset_root:$dataset_root:ro" \
  --bind "$development_root:$development_root" \
  "$image" python3.8 \
  "$benchmark_root/methods/forainet/scripts/provenance/prepare_development_manifest.py" \
  --dataset-root "$dataset_root" \
  --split-metadata "$dataset_root/data_split_metadata.csv" \
  --accepted-smoke "$accepted_smoke" \
  --output-csv "$development_root/manifest/development.csv" \
  --output-json "$development_root/manifest/development.json"

test -s "$development_root/manifest/development.csv"
test -s "$development_root/manifest/development.json"
