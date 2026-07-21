#!/bin/bash -l
#SBATCH --job-name=tls2trees_dt_probe_summary
#SBATCH --partition=nodes
#SBATCH --time=00:15:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --no-requeue
#SBATCH --output=logs/tls2trees_for_instance/%x_%j.out
#SBATCH --error=logs/tls2trees_for_instance/%x_%j.err

set -euo pipefail

PROJECT_ROOT="${TLS2TREES_PROJECT_ROOT:-$HOME/scratch/tree-seg-benchmark}"
cd "$PROJECT_ROOT"
test "${TLS2TREES_DEV_TUNED_PROBE_CONFIRMED:-0}" = "1"
test "$(git rev-parse HEAD)" = "$TLS2TREES_EXPECTED_BENCHMARK_COMMIT"
test -z "$(git status --porcelain)"
test "$(sha256sum "$TLS2TREES_CANDIDATE_MANIFEST" | awk '{print $1}')" = \
  "$TLS2TREES_CANDIDATE_MANIFEST_SHA256"

module purge
module load miniforge3/25.3.0-python3.12.10
# shellcheck disable=SC1090
source "$TLS2TREES_TREEBENCH_ENV/bin/activate"
export PYTHONNOUSERSITE=1

python methods/tls2trees/scripts/evaluation/summarise_tls2trees_compatibility_probe.py \
  --run-root "$TLS2TREES_PROBE_RUN_ROOT" \
  --candidate-manifest "$TLS2TREES_CANDIDATE_MANIFEST" \
  --output-json "$TLS2TREES_PROBE_SUMMARY_JSON" \
  --output-csv "$TLS2TREES_PROBE_SUMMARY_CSV"
