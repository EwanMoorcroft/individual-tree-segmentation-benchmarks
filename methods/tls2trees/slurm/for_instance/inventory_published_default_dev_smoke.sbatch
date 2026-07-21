#!/bin/bash -l
#SBATCH --job-name=tls2trees_smoke_inventory
#SBATCH --partition=nodes
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --output=logs/tls2trees_for_instance/%x_%j.out
#SBATCH --error=logs/tls2trees_for_instance/%x_%j.err

set -euo pipefail

cd "${TLS2TREES_PROJECT_ROOT:-$HOME/scratch/tree-seg-benchmark}"
# shellcheck disable=SC1091
source methods/tls2trees/slurm/for_instance/published_default_dev_smoke_common.sh
tls2trees_load_contract
tls2trees_validate_contract
tls2trees_load_treebench_env
tls2trees_stage_banner inventory

tls2trees_require_file "$TLS2TREES_MANIFEST_CLI" "manifest CLI"
MANIFEST_CSV="$TLS2TREES_WORKFLOW_METADATA_ROOT/development_manifest.csv"
STAGE0_JSON="$TLS2TREES_WORKFLOW_METADATA_ROOT/stage0_selection.json"
STAGE0_CSV="$TLS2TREES_WORKFLOW_METADATA_ROOT/stage0_selection.csv"
for path in "$MANIFEST_JSON" "$MANIFEST_CSV" "$STAGE0_JSON" "$STAGE0_CSV"; do
  if [[ -e "$path" ]]; then
    tls2trees_die "refusing existing inventory output: $path"
  fi
done

mkdir -p "$TLS2TREES_WORKFLOW_METADATA_ROOT"
python "$TLS2TREES_MANIFEST_CLI" build \
  --dataset-root "$TLS2TREES_DATASET_ROOT" \
  --metadata-csv "$TLS2TREES_DATASET_ROOT/data_split_metadata.csv" \
  --split development \
  --output-json "$MANIFEST_JSON" \
  --output-csv "$MANIFEST_CSV"

python "$TLS2TREES_MANIFEST_CLI" select-stage0 \
  --manifest-json "$MANIFEST_JSON" \
  --output-json "$STAGE0_JSON" \
  --output-csv "$STAGE0_CSV"

tls2trees_resolve_stage0_context
test -f "$INPUT_LAS"
echo "selected_safe_plot_id=$SAFE_PLOT_ID"
echo "selected_relative_path=$RELATIVE_PATH"
echo "success_criteria=exact 21-plot development manifest, frozen five-site Stage 0 selection, selected source present"
echo "failure_indicators=split hash/count mismatch, missing development plot, duplicate path, any held-out access"
echo "held_out_test_accessed=false"
echo "finished_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
