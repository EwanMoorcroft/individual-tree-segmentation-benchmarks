#!/usr/bin/env bash

set -euo pipefail
STATE_FILE="${1:?Usage: monitor_held_out_test.sh <held-out-test-state-file>}"
test -f "$STATE_FILE"
# shellcheck disable=SC1090
source "$STATE_FILE"

echo "time=$(date --iso-8601=seconds)"
echo "run_id=$TLS2TREES_TEST_RUN_ID"
echo "submission_status=$TLS2TREES_TEST_SUBMISSION_STATUS"
echo "final_selection_sha256=$TLS2TREES_TEST_FINAL_SELECTION_SHA256"
echo
JOBS="$TLS2TREES_TEST_INVENTORY_JOB,$TLS2TREES_TEST_SEMANTIC_JOB,$TLS2TREES_TEST_CANDIDATE_JOB,$TLS2TREES_TEST_SUMMARY_JOB"
squeue -j "$JOBS" -o "%.18i %.34j %.10T %.10M %.10L %.28R" 2>/dev/null || true
echo

array_status() {
  local label=$1 job=$2 expected=$3 records counts failed
  records=$(sacct -X -n -P -j "$job" --format=JobID%30,State 2>/dev/null | awk -F'|' -v id="$job" '
    {gsub(/^[[:space:]]+|[[:space:]]+$/, "", $1)}
    $1 ~ ("^" id "_[0-9]+$") {sub(/[+ ].*$/, "", $2); print $1 " " $2}')
  counts=$(printf '%s\n' "$records" | awk 'NF==2 {c[$2]++} END {for (s in c) printf "%s%s=%d", sep,s,c[s]; sep=","; if (!sep) printf "not_started=0"}')
  failed=$(printf '%s\n' "$records" | awk '$2 ~ /^(FAILED|CANCELLED|TIMEOUT|OUT_OF_MEMORY|NODE_FAIL|PREEMPTED)$/ {sub(/^.*_/,"",$1); printf "%s%s",sep,$1; sep=","}')
  echo "$label job=$job expected=$expected states=${counts:-not_started=0} failed_tasks=${failed:-NONE}"
}
job_state() {
  sacct -X -n -P -j "$1" --format=JobIDRaw,State 2>/dev/null | awk -F'|' -v id="$1" '$1==id {sub(/[+ ].*$/, "", $2); print $2; exit}'
}
echo "inventory job=$TLS2TREES_TEST_INVENTORY_JOB state=$(job_state "$TLS2TREES_TEST_INVENTORY_JOB" || echo NOT_STARTED)"
array_status semantic "$TLS2TREES_TEST_SEMANTIC_JOB" 11
array_status candidate "$TLS2TREES_TEST_CANDIDATE_JOB" 22
echo "summary job=$TLS2TREES_TEST_SUMMARY_JOB state=$(job_state "$TLS2TREES_TEST_SUMMARY_JOB" || echo NOT_STARTED)"
echo
if [[ -f "$TLS2TREES_TEST_SUMMARY_JSON" ]]; then
  "$TLS2TREES_TEST_TREEBENCH_ENV/bin/python" -c '
import json,sys
p=json.load(open(sys.argv[1])); print("status="+p["status"])
print("valid_metrics={}/{}".format(p["valid_metric_count"],p["expected_metric_count"]))
for a in p["aggregates"]:
 print("{}={} micro_f1={:.6f} mean_plot_f1={:.6f} precision={:.6f} recall={:.6f} invalid={}".format(a["target"],a["candidate_id"],a["micro_f1"],a["mean_plot_f1"],a["precision"],a["recall"],a["failed_or_invalid_plot_count"]))
print("configuration_changed_after_test=false")
' "$TLS2TREES_TEST_SUMMARY_JSON"
else
  echo "status=held_out_test_in_progress_or_awaiting_summary"
  echo "accuracy_metrics_not_displayed_until_summary=true"
fi
