#!/usr/bin/env bash

set -euo pipefail

if [[ "${TLS2TREES_FINAL_FREEZE_CONFIRMED:-0}" != "1" ]]; then
  echo "Refusing TLS2trees final development-configuration freeze." >&2
  echo "Set TLS2TREES_FINAL_FREEZE_CONFIRMED=1 after reviewing all 84 Stage 2 metrics." >&2
  exit 2
fi

STATE_FILE="${1:?Usage: freeze_development_tuned_configuration.sh <completed-stage2-state-file>}"
test -f "$STATE_FILE"
# shellcheck disable=SC1090
source "$STATE_FILE"

RUN_ID="${TLS2TREES_STAGE2_RUN_ID:?State has no Stage 2 run ID}"
SUMMARY_JOB="${TLS2TREES_STAGE2_SUMMARY_JOB:?State has no summary job}"
SUMMARY_JSON="${TLS2TREES_STAGE2_SUMMARY_JSON:?State has no summary JSON}"
STAGE1_CONFIG="${TLS2TREES_STAGE2_STAGE1_CONFIG:?State has no Stage 1 config}"
TREEBENCH_ENV="${TLS2TREES_STAGE2_TREEBENCH_ENV:-$HOME/fastscratch/venvs/treebench}"
PROJECT_ROOT="${TLS2TREES_PROJECT_ROOT:-$HOME/scratch/tree-seg-benchmark}"
FINAL_CONFIG="methods/tls2trees/configs/for_instance_development_tuned_final.yml"
FREEZE_CLI="methods/tls2trees/scripts/evaluation/freeze_tls2trees_development_final.py"

cd "$PROJECT_ROOT"
test -z "$(git status --porcelain)"
BENCHMARK_COMMIT=$(git rev-parse HEAD)
test "$RUN_ID" = \
  "tls2trees_for-instance_development_tuned_stage2_20260718_202002"
test "$SUMMARY_JOB" = "9838957"
test -x "$TREEBENCH_ENV/bin/python"
for path in "$SUMMARY_JSON" "$STAGE1_CONFIG" "$FINAL_CONFIG" "$FREEZE_CLI"; do
  test -f "$path"
done

SUMMARY_STATE=$(sacct -X -n -P -j "$SUMMARY_JOB" --format=JobIDRaw,State | \
  awk -F'|' -v id="$SUMMARY_JOB" '$1 == id {sub(/[+ ].*$/, "", $2); print $2; exit}')
test "$SUMMARY_STATE" = "COMPLETED"
"$TREEBENCH_ENV/bin/python" -c '
import json,sys
p=json.load(open(sys.argv[1]))
assert p["workflow_run_id"] == sys.argv[2]
assert p["status"] == "stage2_completed"
assert p["valid_metric_count"] == 84
assert p["expected_metric_count"] == 84
assert p["held_out_test_accessed"] is False
assert p["final_configuration_selected"] is False
' "$SUMMARY_JSON" "$RUN_ID"

OUTPUT_ROOT="$PROJECT_ROOT/results/metadata/tls2trees/for_instance/development_tuned/stage3/$RUN_ID"
OUTPUT_JSON="$OUTPUT_ROOT/final_selection.json"
test ! -e "$OUTPUT_ROOT"

"$TREEBENCH_ENV/bin/python" "$FREEZE_CLI" \
  --stage2-summary-json "$SUMMARY_JSON" \
  --stage1-config "$STAGE1_CONFIG" \
  --final-config "$FINAL_CONFIG" \
  --benchmark-commit "$BENCHMARK_COMMIT" \
  --output-json "$OUTPUT_JSON"

OUTPUT_SHA256=$(sha256sum "$OUTPUT_JSON" | awk '{print $1}')
printf '%s\n' "$OUTPUT_JSON" > \
  logs/tls2trees_for_instance/latest_final_selection_file.txt
echo "final_selection=$OUTPUT_JSON"
echo "final_selection_sha256=$OUTPUT_SHA256"
echo "next_gate=review_frozen_configuration_before_one_time_held_out_test"
