#!/bin/bash -l
#SBATCH --job-name=tls2trees_pd_finalise
#SBATCH --partition=nodes
#SBATCH --time=00:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --no-requeue
#SBATCH --output=logs/tls2trees_for_instance/%x_%j.out
#SBATCH --error=logs/tls2trees_for_instance/%x_%j.err

set -euo pipefail

cd "${TLS2TREES_PD_FINALISE_PROJECT_ROOT:-$HOME/scratch/tree-seg-benchmark}"
test "${TLS2TREES_PUBLISHED_DEFAULT_RESULTS_CONFIRMED:-0}" = "1"
RECOVERY_CONFIRMED="${TLS2TREES_PUBLISHED_DEFAULT_RESULTS_RECOVERY_CONFIRMED:-0}"
if [[ "$RECOVERY_CONFIRMED" != "0" && "$RECOVERY_CONFIRMED" != "1" ]]; then
  echo "TLS2TREES_PUBLISHED_DEFAULT_RESULTS_RECOVERY_CONFIRMED must be 0 or 1." >&2
  exit 2
fi
# shellcheck disable=SC1091
source methods/tls2trees/slurm/for_instance/published_default_finalisation_worktree_gate.sh
tls2trees_validate_published_default_finalisation_worktree \
  "$PWD" "$RECOVERY_CONFIRMED"
test "$(git rev-parse HEAD)" = \
  "$TLS2TREES_PD_FINALISE_PUBLICATION_BENCHMARK_COMMIT"
git cat-file -e "${TLS2TREES_PD_FINALISE_BENCHMARK_COMMIT}^{commit}"
git merge-base --is-ancestor \
  "$TLS2TREES_PD_FINALISE_BENCHMARK_COMMIT" \
  "$TLS2TREES_PD_FINALISE_PUBLICATION_BENCHMARK_COMMIT"

module purge
module load miniforge3/25.3.0-python3.12.10
# shellcheck disable=SC1090
source "$TLS2TREES_PD_FINALISE_TREEBENCH_ENV/bin/activate"
export PYTHONNOUSERSITE=1

MANIFEST_SHA256=$(awk '{print $1}' "$TLS2TREES_PD_FINALISE_MANIFEST_SHA256_FILE")

python methods/tls2trees/scripts/evaluation/finalise_tls2trees_published_default_results.py \
  --project-root "$PWD" \
  --run-id "$TLS2TREES_PD_FINALISE_RUN_ID" \
  --benchmark-commit "$TLS2TREES_PD_FINALISE_BENCHMARK_COMMIT" \
  --publication-benchmark-commit \
  "$TLS2TREES_PD_FINALISE_PUBLICATION_BENCHMARK_COMMIT" \
  --upstream-commit "$TLS2TREES_PD_FINALISE_UPSTREAM_COMMIT" \
  --model-sha256 "$TLS2TREES_PD_FINALISE_MODEL_SHA256" \
  --workflow-config "$TLS2TREES_PD_FINALISE_WORKFLOW_CONFIG" \
  --workflow-config-sha256 "$TLS2TREES_PD_FINALISE_WORKFLOW_CONFIG_SHA256" \
  --published-config "$TLS2TREES_PD_FINALISE_PUBLISHED_CONFIG" \
  --published-config-sha256 "$TLS2TREES_PD_FINALISE_PUBLISHED_CONFIG_SHA256" \
  --benchmark-config "$TLS2TREES_PD_FINALISE_BENCHMARK_CONFIG" \
  --benchmark-config-sha256 "$TLS2TREES_PD_FINALISE_BENCHMARK_CONFIG_SHA256" \
  --manifest-json "$TLS2TREES_PD_FINALISE_MANIFEST_JSON" \
  --manifest-sha256 "$MANIFEST_SHA256" \
  --summary-json "$TLS2TREES_PD_FINALISE_SUMMARY_JSON" \
  --plot-csv "$TLS2TREES_PD_FINALISE_PLOT_CSV" \
  --target-csv "$TLS2TREES_PD_FINALISE_TARGET_CSV" \
  --source-retention-json "$TLS2TREES_PD_FINALISE_SOURCE_RETENTION_JSON" \
  --examples-dir methods/tls2trees/examples \
  --results-csv outputs/for_instance_benchmark_metrics/for_instance_method_benchmark_results.csv \
  --diagnostics-csv outputs/for_instance_benchmark_metrics/for_instance_method_development_diagnostics.csv \
  --retention-registry outputs/for_instance_benchmark_metrics/for_instance_prediction_retention_registry.csv \
  --recovery-confirmed "$RECOVERY_CONFIRMED" \
  --receipt-json "$TLS2TREES_PD_FINALISE_RECEIPT_JSON"

echo "status=TLS2TREES_PUBLISHED_DEFAULT_RESULTS_READY_TO_COMMIT"
echo "receipt=$TLS2TREES_PD_FINALISE_RECEIPT_JSON"
