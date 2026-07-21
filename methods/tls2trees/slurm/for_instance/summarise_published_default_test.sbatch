#!/bin/bash -l
#SBATCH --job-name=tls2trees_pd_test_summary
#SBATCH --partition=nodes
#SBATCH --time=00:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --no-requeue
#SBATCH --output=logs/tls2trees_for_instance/%x_%j.out
#SBATCH --error=logs/tls2trees_for_instance/%x_%j.err

set -euo pipefail
cd "${TLS2TREES_PD_TEST_PROJECT_ROOT:-$HOME/scratch/tree-seg-benchmark}"
# shellcheck disable=SC1091
source methods/tls2trees/slurm/for_instance/published_default_test_common.sh
pd_test_load_contract
pd_test_require_manifest
pd_test_load_treebench
MANIFEST_SHA256=$(awk '{print $1}' "$TLS2TREES_PD_TEST_MANIFEST_SHA256_FILE")
python methods/tls2trees/scripts/evaluation/summarise_tls2trees_published_default_test.py \
  --project-root "$TLS2TREES_PD_TEST_PROJECT_ROOT" \
  --output-root "$TLS2TREES_PD_TEST_OUTPUT_ROOT" \
  --run-id "$TLS2TREES_PD_TEST_RUN_ID" \
  --manifest-json "$TLS2TREES_PD_TEST_MANIFEST_JSON" \
  --manifest-sha256 "$MANIFEST_SHA256" \
  --workflow-config "$TLS2TREES_PD_TEST_WORKFLOW_CONFIG" \
  --workflow-config-sha256 "$TLS2TREES_PD_TEST_WORKFLOW_CONFIG_SHA256" \
  --published-config "$TLS2TREES_PD_TEST_PUBLISHED_CONFIG" \
  --published-config-sha256 "$TLS2TREES_PD_TEST_PUBLISHED_CONFIG_SHA256" \
  --benchmark-config "$TLS2TREES_PD_TEST_BENCHMARK_CONFIG" \
  --benchmark-config-sha256 "$TLS2TREES_PD_TEST_BENCHMARK_CONFIG_SHA256" \
  --output-json "$TLS2TREES_PD_TEST_SUMMARY_JSON" \
  --plot-csv "$TLS2TREES_PD_TEST_PLOT_CSV" \
  --aggregate-csv "$TLS2TREES_PD_TEST_AGGREGATE_CSV" \
  --retention-json "$TLS2TREES_PD_TEST_RETENTION_JSON"
