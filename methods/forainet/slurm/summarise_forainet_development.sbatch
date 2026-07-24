#!/usr/bin/env bash
#SBATCH --job-name=forai_devsum
#SBATCH --partition=nodes
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=logs/forai_devsum_%j.out
#SBATCH --error=logs/forai_devsum_%j.err

set -euo pipefail
trap 'status=$?; printf "ForAINet development summary failed at line %s: %s\n" "$LINENO" "$BASH_COMMAND" >&2; exit "$status"' ERR

: "${FORAINET_BENCHMARK_ROOT:?set FORAINET_BENCHMARK_ROOT}"
: "${FORAINET_EXPECTED_BENCHMARK_COMMIT:?set FORAINET_EXPECTED_BENCHMARK_COMMIT}"
: "${FORAINET_DEVELOPMENT_ROOT:?set FORAINET_DEVELOPMENT_ROOT}"
: "${FORAINET_DEVELOPMENT_RUN_ID:?set FORAINET_DEVELOPMENT_RUN_ID}"

test "$(git -C "$FORAINET_BENCHMARK_ROOT" rev-parse HEAD)" = \
  "$FORAINET_EXPECTED_BENCHMARK_COMMIT"
test -z "$(git -C "$FORAINET_BENCHMARK_ROOT" status --porcelain)"
original_commit="${FORAINET_DEVELOPMENT_ORIGINAL_BENCHMARK_COMMIT:-$FORAINET_EXPECTED_BENCHMARK_COMMIT}"
summary_args=(
  --run-root "$FORAINET_DEVELOPMENT_ROOT" \
  --run-id "$FORAINET_DEVELOPMENT_RUN_ID" \
  --manifest-csv "$FORAINET_DEVELOPMENT_ROOT/manifest/development.csv" \
  --manifest-json "$FORAINET_DEVELOPMENT_ROOT/manifest/development.json" \
  --benchmark-commit "$original_commit"
)
if [[ -n "${FORAINET_DEVELOPMENT_RECOVERY_ROOT:-}" ]]; then
  test -d "$FORAINET_DEVELOPMENT_RECOVERY_ROOT"
  summary_args+=(
    --recovery-root "$FORAINET_DEVELOPMENT_RECOVERY_ROOT"
    --recovery-benchmark-commit "$FORAINET_EXPECTED_BENCHMARK_COMMIT"
  )
fi
python3 \
  "$FORAINET_BENCHMARK_ROOT/methods/forainet/scripts/provenance/summarise_development_run.py" \
  "${summary_args[@]}"

test -f "$FORAINET_DEVELOPMENT_ROOT/final_gate.json"
test -f "$FORAINET_DEVELOPMENT_ROOT/summary/metrics.json"
test -f "$FORAINET_DEVELOPMENT_ROOT/retention/manifest.json"
