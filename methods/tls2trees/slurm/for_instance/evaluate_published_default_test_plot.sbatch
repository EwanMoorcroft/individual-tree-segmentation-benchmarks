#!/bin/bash -l
#SBATCH --job-name=tls2trees_pd_test_evaluate
#SBATCH --partition=nodes
#SBATCH --time=08:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --no-requeue
#SBATCH --output=logs/tls2trees_for_instance/%x_%A_%a.out
#SBATCH --error=logs/tls2trees_for_instance/%x_%A_%a.err

set -euo pipefail
cd "${TLS2TREES_PD_TEST_PROJECT_ROOT:-$HOME/scratch/tree-seg-benchmark}"
# shellcheck disable=SC1091
source methods/tls2trees/slurm/for_instance/published_default_test_common.sh
pd_test_load_contract
pd_test_require_manifest
TASK_INDEX="$SLURM_ARRAY_TASK_ID"
pd_test_load_treebench
pd_test_resolve_plot "$TASK_INDEX"
PLOT_ROOT="$TLS2TREES_PD_TEST_OUTPUT_ROOT/tls2trees/for_instance/published_default/test/$TLS2TREES_PD_TEST_RUN_ID/$TLS2TREES_PD_TEST_SAFE_PLOT_ID"
test -f "$PLOT_ROOT/metadata/semantic_run.json"

pd_test_load_method
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
python "$TLS2TREES_PD_TEST_ENV_VALIDATOR" \
  --tls2trees-repo "$TLS2TREES_PD_TEST_UPSTREAM_REPO" \
  --setup-marker-json "$TLS2TREES_PD_TEST_METHOD_ENV_MARKER" \
  --skip-model-load
python "$TLS2TREES_PD_TEST_INSTANCE_CLI" \
  --manifest-json "$TLS2TREES_PD_TEST_MANIFEST_JSON" \
  --task-index "$TASK_INDEX" \
  --output-root "$TLS2TREES_PD_TEST_OUTPUT_ROOT" \
  --run-id "$TLS2TREES_PD_TEST_RUN_ID" \
  --tls2trees-repo "$TLS2TREES_PD_TEST_UPSTREAM_REPO" \
  --config "$TLS2TREES_PD_TEST_PUBLISHED_CONFIG" \
  --variant published_default --split test --allow-held-out-test

conda deactivate
pd_test_load_treebench
python "$TLS2TREES_PD_TEST_ADAPTER_CLI" \
  --manifest-json "$TLS2TREES_PD_TEST_MANIFEST_JSON" \
  --task-index "$TASK_INDEX" \
  --output-root "$TLS2TREES_PD_TEST_OUTPUT_ROOT" \
  --run-id "$TLS2TREES_PD_TEST_RUN_ID" \
  --variant published_default --split test \
  --coordinate-tolerance-m 0.001 --allow-held-out-test

for TARGET in leaf_off leaf_on; do
  ALIGNED_ROOT="$PLOT_ROOT/predictions/aligned/$TARGET"
  EVALUATION_ROOT="$PLOT_ROOT/evaluation/$TARGET"
  test ! -e "$EVALUATION_ROOT"
  mkdir -p "$EVALUATION_ROOT"
  python "$TLS2TREES_PD_TEST_EVALUATE_CLI" \
    --target "$TARGET" \
    --aligned-predictions-npz "$ALIGNED_ROOT/source_row_predictions.npz" \
    --reference-labelled-point-cloud "$TLS2TREES_PD_TEST_INPUT_LAS" \
    --alignment-metadata-json "$ALIGNED_ROOT/alignment_metadata.json" \
    --plot-id "$TLS2TREES_PD_TEST_SAFE_PLOT_ID" \
    --relative-path "$TLS2TREES_PD_TEST_RELATIVE_PATH" \
    --split test --iou-threshold 0.5 \
    --output-json "$EVALUATION_ROOT/plot_metrics.json" \
    --output-matches-csv "$EVALUATION_ROOT/matches.csv" \
    --output-unmatched-predictions-csv "$EVALUATION_ROOT/unmatched_predictions.csv" \
    --output-unmatched-references-csv "$EVALUATION_ROOT/unmatched_references.csv"
done

echo "status=published_default_test_plot_evaluated"
echo "task_index=$TASK_INDEX"
echo "safe_plot_id=$TLS2TREES_PD_TEST_SAFE_PLOT_ID"
echo "metrics_written=2"
echo "configuration_changed_after_test=false"
