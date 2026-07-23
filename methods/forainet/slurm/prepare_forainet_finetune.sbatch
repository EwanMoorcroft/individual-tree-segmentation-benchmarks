#!/usr/bin/env bash
#SBATCH --job-name=forai_ftprep
#SBATCH --partition=nodes
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/forai_ftprep_%j.out
#SBATCH --error=logs/forai_ftprep_%j.err

set -euo pipefail
trap 'status=$?; printf "ForAINet fine-tune preparation failed at line %s: %s\n" "$LINENO" "$BASH_COMMAND" >&2; exit "$status"' ERR

: "${FORAINET_BENCHMARK_ROOT:?set FORAINET_BENCHMARK_ROOT}"
: "${FORAINET_EXPECTED_BENCHMARK_COMMIT:?set FORAINET_EXPECTED_BENCHMARK_COMMIT}"
: "${FORAINET_IMAGE:?set FORAINET_IMAGE}"
: "${FORAINET_DATASET_ROOT:?set FORAINET_DATASET_ROOT}"
: "${FORAINET_DEVELOPMENT_ROOT:?set FORAINET_DEVELOPMENT_ROOT}"
: "${FORAINET_FINETUNE_ROOT:?set FORAINET_FINETUNE_ROOT}"

test "$(git -C "$FORAINET_BENCHMARK_ROOT" rev-parse HEAD)" = \
  "$FORAINET_EXPECTED_BENCHMARK_COMMIT"
test -z "$(git -C "$FORAINET_BENCHMARK_ROOT" status --porcelain)"
test -f "$FORAINET_IMAGE"
test -f "$FORAINET_DEVELOPMENT_ROOT/manifest/development.json"
test -f "$FORAINET_DEVELOPMENT_ROOT/final_gate.json"
test ! -e "$FORAINET_FINETUNE_ROOT"

benchmark_root="$(readlink -f "$FORAINET_BENCHMARK_ROOT")"
dataset_root="$(readlink -f "$FORAINET_DATASET_ROOT")"
development_root="$(readlink -f "$FORAINET_DEVELOPMENT_ROOT")"
image="$(readlink -f "$FORAINET_IMAGE")"
finetune_parent="$(readlink -f "$(dirname "$FORAINET_FINETUNE_ROOT")")"
finetune_root="$finetune_parent/$(basename "$FORAINET_FINETUNE_ROOT")"

module load apptainer/1.3.6
apptainer exec --cleanenv \
  --bind "$benchmark_root:$benchmark_root:ro" \
  --bind "$dataset_root:$dataset_root:ro" \
  --bind "$development_root:$development_root:ro" \
  --bind "$finetune_parent:$finetune_parent" \
  "$image" python3.8 \
    "$benchmark_root/methods/forainet/scripts/data/prepare_finetune_data.py" \
    --development-manifest "$development_root/manifest/development.json" \
    --development-final-gate "$development_root/final_gate.json" \
    --dataset-root "$dataset_root" \
    --output-root "$finetune_root" \
    --seed 42

test -s "$finetune_root/finetune_data_manifest.json"
python3 - "$finetune_root/finetune_data_manifest.json" <<'PY'
import json
import sys
payload = json.load(open(sys.argv[1], encoding="utf-8"))
assert payload["status"] == "complete"
assert payload["role_counts"] == {"train": 16, "validation": 5}
assert payload["held_out_access"] is False
assert payload["held_out_paths_included"] is False
PY
