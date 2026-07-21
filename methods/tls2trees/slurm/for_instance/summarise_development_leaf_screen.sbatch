#!/bin/bash -l
#SBATCH --job-name=tls2trees_dt_leaf_summary
#SBATCH --partition=nodes
#SBATCH --time=00:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --no-requeue
#SBATCH --output=logs/tls2trees_for_instance/%x_%j.out
#SBATCH --error=logs/tls2trees_for_instance/%x_%j.err

set -euo pipefail

cd "${TLS2TREES_PROJECT_ROOT:-$HOME/scratch/tree-seg-benchmark}"
test "${TLS2TREES_LEAF_SCREEN_CONFIRMED:-0}" = "1"
test "${TLS2TREES_REQUESTED_SPLIT:-}" = "development"
test "$(git rev-parse HEAD)" = "$TLS2TREES_EXPECTED_BENCHMARK_COMMIT"
test -z "$(git status --porcelain)"
test "$(sha256sum "$TLS2TREES_LEAF_SCREEN_CONFIG" | awk '{print $1}')" = \
  "$TLS2TREES_LEAF_SCREEN_CONFIG_SHA256"
test "$(sha256sum "$TLS2TREES_DEVELOPMENT_EVIDENCE_JSON" | awk '{print $1}')" = \
  "$TLS2TREES_DEVELOPMENT_EVIDENCE_SHA256"

module purge
module load miniforge3/25.3.0-python3.12.10
# shellcheck disable=SC1090
source "$TLS2TREES_TREEBENCH_ENV/bin/activate"
export PYTHONNOUSERSITE=1

python methods/tls2trees/scripts/evaluation/summarise_tls2trees_development_leaf_screen.py \
  --output-root "$TLS2TREES_OUTPUT_ROOT" \
  --workflow-run-id "$TLS2TREES_LEAF_SCREEN_RUN_ID" \
  --manifest-json "$TLS2TREES_MANIFEST_JSON" \
  --candidate-config "$TLS2TREES_LEAF_SCREEN_CONFIG" \
  --development-evidence-json "$TLS2TREES_DEVELOPMENT_EVIDENCE_JSON" \
  --development-evidence-sha256 "$TLS2TREES_DEVELOPMENT_EVIDENCE_SHA256" \
  --output-json "$TLS2TREES_LEAF_SCREEN_SUMMARY_JSON" \
  --plot-csv "$TLS2TREES_LEAF_SCREEN_PLOT_CSV" \
  --aggregate-csv "$TLS2TREES_LEAF_SCREEN_AGGREGATE_CSV"
