#!/usr/bin/env bash
#SBATCH --job-name=forai_ftvalsum
#SBATCH --partition=nodes
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=logs/forai_ftvalsum_%j.out
#SBATCH --error=logs/forai_ftvalsum_%j.err

set -euo pipefail
trap 'status=$?; printf "ForAINet fine-tune validation summary failed at line %s: %s\n" "$LINENO" "$BASH_COMMAND" >&2; exit "$status"' ERR

: "${FORAINET_BENCHMARK_ROOT:?set FORAINET_BENCHMARK_ROOT}"
: "${FORAINET_EXPECTED_BENCHMARK_COMMIT:?set FORAINET_EXPECTED_BENCHMARK_COMMIT}"
: "${FORAINET_FINETUNE_ROOT:?set FORAINET_FINETUNE_ROOT}"
: "${FORAINET_FINETUNE_VALIDATION_ROOT:?set FORAINET_FINETUNE_VALIDATION_ROOT}"
: "${FORAINET_FINETUNE_VALIDATION_RUN_ID:?set FORAINET_FINETUNE_VALIDATION_RUN_ID}"

test "$(git -C "$FORAINET_BENCHMARK_ROOT" rev-parse HEAD)" = \
  "$FORAINET_EXPECTED_BENCHMARK_COMMIT"
test -z "$(git -C "$FORAINET_BENCHMARK_ROOT" status --porcelain)"
python3 \
  "$FORAINET_BENCHMARK_ROOT/methods/forainet/scripts/provenance/summarise_finetune_validation.py" \
  --validation-root "$FORAINET_FINETUNE_VALIDATION_ROOT" \
  --validation-run-id "$FORAINET_FINETUNE_VALIDATION_RUN_ID" \
  --finetune-root "$FORAINET_FINETUNE_ROOT" \
  --benchmark-commit "$FORAINET_EXPECTED_BENCHMARK_COMMIT"

test -f "$FORAINET_FINETUNE_VALIDATION_ROOT/final_gate.json"
test -f "$FORAINET_FINETUNE_VALIDATION_ROOT/summary/selection.json"
test -f "$FORAINET_FINETUNE_VALIDATION_ROOT/retention/manifest.json"
