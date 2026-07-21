#!/bin/bash -l
#SBATCH --job-name=tls2trees_smoke_summary
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
tls2trees_stage_banner summary

SEMANTIC="$PLOT_ROOT/metadata/semantic_run.json"
INSTANCE="$PLOT_ROOT/metadata/instance_run.json"
ADAPTER="$PLOT_ROOT/metadata/adapter_run.json"
LEAF_OFF="$PLOT_ROOT/evaluation/leaf_off/plot_metrics.json"
LEAF_ON="$PLOT_ROOT/evaluation/leaf_on/plot_metrics.json"
TARGET_CSV="$TLS2TREES_WORKFLOW_TABLE_ROOT/target_summary.csv"
for path in "$TLS2TREES_GATE_METADATA" "$SEMANTIC" "$INSTANCE" "$ADAPTER" "$LEAF_OFF" "$LEAF_ON"; do
  tls2trees_require_file "$path" "summary input"
done
for path in "$TLS2TREES_RUN_SUMMARY" "$TARGET_CSV"; do
  if [[ -e "$path" ]]; then
    tls2trees_die "refusing existing summary output: $path"
  fi
done

python -c '
import csv
import hashlib
import json
import pathlib
import sys

gate_path, semantic_path, instance_path, adapter_path, leaf_off_path, leaf_on_path, output_path, csv_path = map(pathlib.Path, sys.argv[1:])

def load(path):
    return json.loads(path.read_text(encoding="utf-8"))

gate = load(gate_path)
semantic = load(semantic_path)
instance = load(instance_path)
adapter = load(adapter_path)
metrics = {"leaf_off": load(leaf_off_path), "leaf_on": load(leaf_on_path)}
if gate.get("status") != "passed_automated_gates" or gate.get("source_row_alignment") != "passed":
    raise SystemExit("smoke gate has not passed")
if gate.get("manual_alignment_review_required") is not True or gate.get("full_development_authorised") is not False or gate.get("held_out_test_authorised") is not False:
    raise SystemExit("smoke gate did not preserve manual-review and split-isolation boundaries")
metric_fields = (
    "prediction_instance_count", "reference_instance_count", "true_positives",
    "false_positives", "false_negatives", "precision", "recall", "f1",
    "mean_matched_iou", "mean_unweighted_coverage", "mean_weighted_coverage",
    "evaluated_point_count", "oversegmented_reference_count",
    "undersegmented_prediction_count",
)
targets = {
    target: {field: payload.get(field) for field in metric_fields}
    for target, payload in metrics.items()
}
payload = {
    "schema_version": 1,
    "status": "development_smoke_completed",
    "safe_for_scoring": True,
    "dataset": "FOR-instance",
    "method": "TLS2trees",
    "variant": "published_default",
    "split": "development",
    "run_id": semantic["run_id"],
    "plot_id": gate["safe_plot_id"],
    "relative_path": gate["relative_path"],
    "held_out_test_accessed": False,
    "targets": targets,
    "stage_resources": {
        "semantic": {
            "runtime_seconds": semantic.get("runtime_seconds"),
            "peak_rss_gb": semantic.get("peak_rss_gb"),
        },
        "instance": {
            "runtime_seconds": instance.get("runtime_seconds"),
            "peak_rss_gb": instance.get("peak_rss_gb"),
        },
        "adapter": {
            "runtime_seconds": adapter.get("runtime_seconds"),
            "peak_rss_gb": adapter.get("peak_rss_gb"),
        },
    },
    "gate": {
        "path": str(gate_path),
        "sha256": hashlib.sha256(gate_path.read_bytes()).hexdigest(),
    },
    "next_gate": "manual_alignment_and_resource_review_before_any_full_development_route",
}
output_path.parent.mkdir(parents=True, exist_ok=True)
temporary = output_path.with_suffix(output_path.suffix + ".partial")
temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
temporary.replace(output_path)
with csv_path.open("x", encoding="utf-8", newline="") as handle:
    fieldnames = ["target", *metric_fields]
    writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    for target in ("leaf_off", "leaf_on"):
        writer.writerow({"target": target, **targets[target]})
' "$TLS2TREES_GATE_METADATA" "$SEMANTIC" "$INSTANCE" "$ADAPTER" "$LEAF_OFF" "$LEAF_ON" "$TLS2TREES_RUN_SUMMARY" "$TARGET_CSV"

test -f "$TLS2TREES_RUN_SUMMARY"
test -f "$TARGET_CSV"
echo "status=development-smoke-summarised"
echo "run_summary=$TLS2TREES_RUN_SUMMARY"
echo "target_summary=$TARGET_CSV"
echo "next_gate=manual_alignment_and_resource_review_before_any_full_development_route"
echo "No tuning, full-development array or held-out-test job was enabled."
