#!/bin/bash -l
#SBATCH --job-name=tls2trees_smoke_convert
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
tls2trees_stage_banner conversion

tls2trees_require_file "$TLS2TREES_CONVERT_CLI" "published-input adapter"
if [[ -e "$PLOT_ROOT" ]]; then
  tls2trees_die "refusing existing immutable plot root: $PLOT_ROOT"
fi

python "$TLS2TREES_CONVERT_CLI" \
  --manifest-json "$MANIFEST_JSON" \
  --task-index "$TASK_INDEX" \
  --output-root "$OUTPUT_ROOT" \
  --run-id "$RUN_ID" \
  --variant published_default \
  --split development

test -f "$PLOT_ROOT/converted/conversion_metadata.json"
test -f "$PLOT_ROOT/converted/source_map.npz"
test -f "$PLOT_ROOT/converted/tile_index.dat"
echo "plot_root=$PLOT_ROOT"
echo "success_criteria=label-stripped 10 m tiles, 0.02 m representatives, reversible local shift, source-row map"
echo "failure_indicators=reference fields passed, point-count mismatch, coordinate round-trip error, existing output root"
echo "held_out_test_accessed=false"
echo "finished_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
