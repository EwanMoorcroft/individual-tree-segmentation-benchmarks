#!/usr/bin/env bash

set -euo pipefail
STATE_FILE="${1:?Usage: monitor_published_default_held_out_test.sh <state-file>}"
test -f "$STATE_FILE"
# shellcheck disable=SC1090
source "$STATE_FILE"

echo "time=$(date --iso-8601=seconds)"
echo "run_id=$TLS2TREES_PD_TEST_RUN_ID"
echo "submission_status=$TLS2TREES_PD_TEST_SUBMISSION_STATUS"
echo "published_config_sha256=$TLS2TREES_PD_TEST_PUBLISHED_CONFIG_SHA256"
echo "semantic_cache_available=$TLS2TREES_PD_TEST_CACHE_AVAILABLE"
echo
JOBS="$TLS2TREES_PD_TEST_MANIFEST_JOB,$TLS2TREES_PD_TEST_SEMANTIC_JOB,$TLS2TREES_PD_TEST_EVALUATE_JOB,$TLS2TREES_PD_TEST_SUMMARY_JOB"
squeue -j "$JOBS" -o "%.18i %.34j %.10T %.10M %.10L %.28R" 2>/dev/null || true
echo

job_state() {
  sacct -X -n -P -j "$1" --format=JobID%30,State 2>/dev/null |
    awk -F'|' -v id="$1" '
      {gsub(/^[[:space:]]+|[[:space:]]+$/, "", $1)}
      $1 == id {sub(/[+ ].*$/, "", $2); print $2; exit}'
}
array_status() {
  local label=$1 job=$2 expected=$3 records counts failed
  records=$(sacct -X -n -P -j "$job" --format=JobID%30,State 2>/dev/null |
    awk -F'|' -v id="$job" '
      {gsub(/^[[:space:]]+|[[:space:]]+$/, "", $1)}
      $1 ~ ("^" id "_[0-9]+$") {sub(/[+ ].*$/, "", $2); print $1 " " $2}')
  counts=$(printf '%s\n' "$records" | awk '
    NF==2 {count[$2]++}
    END {for (state in count) printf "%s%s=%d", separator,state,count[state]; separator=",";
         if (!separator) printf "not_started=0"}')
  failed=$(printf '%s\n' "$records" | awk '
    $2 ~ /^(FAILED|CANCELLED|TIMEOUT|OUT_OF_MEMORY|NODE_FAIL|PREEMPTED)$/ {
      sub(/^.*_/,"",$1); printf "%s%s",separator,$1; separator=","}')
  echo "$label job=$job expected=$expected states=${counts:-not_started=0} failed_tasks=${failed:-NONE}"
}

echo "manifest job=$TLS2TREES_PD_TEST_MANIFEST_JOB state=$(job_state "$TLS2TREES_PD_TEST_MANIFEST_JOB" || echo NOT_STARTED)"
array_status semantic "$TLS2TREES_PD_TEST_SEMANTIC_JOB" 11
array_status evaluate "$TLS2TREES_PD_TEST_EVALUATE_JOB" 11
echo "summary job=$TLS2TREES_PD_TEST_SUMMARY_JOB state=$(job_state "$TLS2TREES_PD_TEST_SUMMARY_JOB" || echo NOT_STARTED)"
echo

if [[ -s "$TLS2TREES_PD_TEST_SUMMARY_JSON" ]]; then
  "$TLS2TREES_PD_TEST_TREEBENCH_ENV/bin/python" -c '
import json,sys
p=json.load(open(sys.argv[1]))
print("status="+p["status"])
print("valid_metrics={}/{}".format(p["valid_metric_count"],p["expected_metric_count"]))
for row in p["aggregates"]:
 print("{}: micro_f1={:.6f} precision={:.6f} recall={:.6f} predictions={} references={}".format(row["target"],row["micro_f1"],row["precision"],row["recall"],row["prediction_instance_count"],row["reference_instance_count"]))
print("semantic_cache_reused_plots={}".format(p["semantic_cache_reused_plot_count"]))
print("dedicated_semantic_plots={}".format(p["dedicated_semantic_plot_count"]))
print("configuration_changed_after_test=false")
' "$TLS2TREES_PD_TEST_SUMMARY_JSON"
  test -s "$TLS2TREES_PD_TEST_RETENTION_JSON"
  echo "retention_manifest_sha256=$(sha256sum "$TLS2TREES_PD_TEST_RETENTION_JSON" | awk '{print $1}')"
else
  echo "status=published_default_test_in_progress_or_failed"
  echo "accuracy_metrics_not_available_until_summary=true"
fi
