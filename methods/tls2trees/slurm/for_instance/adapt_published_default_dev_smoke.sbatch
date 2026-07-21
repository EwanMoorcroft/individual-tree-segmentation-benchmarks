#!/bin/bash -l
#SBATCH --job-name=tls2trees_smoke_adapter
#SBATCH --partition=nodes
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --output=logs/tls2trees_for_instance/%x_%j.out
#SBATCH --error=logs/tls2trees_for_instance/%x_%j.err

set -euo pipefail

cd "${TLS2TREES_PROJECT_ROOT:-$HOME/scratch/tree-seg-benchmark}"
# shellcheck disable=SC1091
source methods/tls2trees/slurm/for_instance/published_default_dev_smoke_common.sh
tls2trees_load_contract
tls2trees_validate_contract
tls2trees_load_treebench_env
tls2trees_resolve_stage0_context
tls2trees_stage_banner adapter

tls2trees_require_file "$TLS2TREES_ADAPTER_CLI" "prediction adapter"
tls2trees_require_file "$PLOT_ROOT/metadata/instance_run.json" "instance metadata"
tls2trees_require_dir "$PLOT_ROOT/predictions/raw" "raw predictions"
if [[ -e "$PLOT_ROOT/predictions/aligned" || -e "$PLOT_ROOT/metadata/adapter_run.json" ]]; then
  tls2trees_die "refusing existing aligned prediction output under $PLOT_ROOT"
fi

python "$TLS2TREES_ADAPTER_CLI" \
  --manifest-json "$MANIFEST_JSON" \
  --task-index "$TASK_INDEX" \
  --output-root "$OUTPUT_ROOT" \
  --run-id "$RUN_ID" \
  --variant published_default \
  --split development \
  --coordinate-tolerance-m 0.001

test -f "$PLOT_ROOT/metadata/adapter_run.json"
for target in leaf_off leaf_on; do
  test -f "$PLOT_ROOT/predictions/aligned/$target/source_row_predictions.npz"
  test -f "$PLOT_ROOT/predictions/aligned/$target/alignment_metadata.json"
done
echo "plot_root=$PLOT_ROOT"
echo "coordinate_tolerance_m=0.001"
echo "success_criteria=recorded raw local frame, unique source-row assignment, multiplicity audit, aligned NPZ for both targets"
echo "failure_indicators=unmatched coordinate, many-to-one contention, frame mismatch, missing target"
echo "held_out_test_accessed=false"
echo "finished_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
