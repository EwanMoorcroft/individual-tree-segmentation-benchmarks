#!/bin/bash -l
#SBATCH --job-name=tls2trees_smoke_gate
#SBATCH --partition=nodes
#SBATCH --time=00:30:00
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
tls2trees_resolve_stage0_context
tls2trees_stage_banner gate

tls2trees_require_file "$TLS2TREES_GATE_CLI" "automated smoke-gate CLI"
PRIMARY_GATE="$PLOT_ROOT/metadata/smoke_gate.json"
if [[ -e "$PRIMARY_GATE" || -e "$TLS2TREES_GATE_METADATA" ]]; then
  tls2trees_die "refusing existing smoke-gate output"
fi

python "$TLS2TREES_GATE_CLI" \
  --manifest-json "$MANIFEST_JSON" \
  --task-index "$TASK_INDEX" \
  --output-root "$OUTPUT_ROOT" \
  --run-id "$RUN_ID" \
  --variant published_default \
  --split development

test -f "$PRIMARY_GATE"
python -c 'import json,sys; p=json.load(open(sys.argv[1])); assert p.get("status") == "passed_automated_gates"; assert p.get("source_row_alignment") == "passed"; assert p.get("manual_alignment_review_required") is True; assert p.get("full_development_authorised") is False; assert p.get("held_out_test_authorised") is False; assert set(p.get("target_results", {})) == {"leaf_off", "leaf_on"}; assert all(int(v.get("prediction_instance_count", 0)) > 0 for v in p["target_results"].values())' "$PRIMARY_GATE"
mkdir -p "$TLS2TREES_WORKFLOW_METADATA_ROOT"
cp "$PRIMARY_GATE" "$TLS2TREES_GATE_METADATA"

test -f "$TLS2TREES_GATE_METADATA"
echo "status=passed_automated_gates"
echo "primary_gate=$PRIMARY_GATE"
echo "gate_copy=$TLS2TREES_GATE_METADATA"
echo "manual_alignment_review_required=true"
echo "full_development_authorised=false"
echo "held_out_test_authorised=false"
