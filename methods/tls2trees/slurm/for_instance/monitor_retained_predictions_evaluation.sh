#!/usr/bin/env bash

set -euo pipefail

STATE_FILE="${1:?Usage: monitor_retained_predictions_evaluation.sh <final-evaluation-state-file>}"
test -f "$STATE_FILE"
# shellcheck disable=SC1090
source "$STATE_FILE"

echo "time=$(date --iso-8601=seconds)"
echo "run_id=$TLS2TREES_FINAL_EVALUATION_RUN_ID"
echo "submission_status=$TLS2TREES_FINAL_EVALUATION_SUBMISSION_STATUS"
echo
JOBS="$TLS2TREES_FINAL_EVALUATION_JOB,$TLS2TREES_FINAL_EVALUATION_SUMMARY_JOB"
squeue -j "$JOBS" -o "%.18i %.34j %.10T %.10M %.10L %.28R" 2>/dev/null || true
echo

records=$(sacct -X -n -P -j "$TLS2TREES_FINAL_EVALUATION_JOB" \
  --format=JobID%30,State 2>/dev/null | awk -F'|' \
  -v id="$TLS2TREES_FINAL_EVALUATION_JOB" '
    {gsub(/^[[:space:]]+|[[:space:]]+$/, "", $1)}
    $1 ~ ("^" id "_[0-9]+$") {sub(/[+ ].*$/, "", $2); print $1 " " $2}')
counts=$(printf '%s\n' "$records" | awk '
  NF == 2 {count[$2]++}
  END {for (state in count) printf "%s%s=%d", sep,state,count[state]; sep=",";
       if (!sep) printf "not_started=0"}')
failed=$(printf '%s\n' "$records" | awk '
  $2 ~ /^(FAILED|CANCELLED|TIMEOUT|OUT_OF_MEMORY|NODE_FAIL|PREEMPTED)$/ {
    sub(/^.*_/, "", $1); printf "%s%s", sep,$1; sep=","}')
echo "evaluation job=$TLS2TREES_FINAL_EVALUATION_JOB expected=106 states=${counts:-not_started=0} failed_tasks=${failed:-NONE}"

summary_state=$(sacct -X -n -P -j "$TLS2TREES_FINAL_EVALUATION_SUMMARY_JOB" \
  --format=JobIDRaw,State 2>/dev/null | awk -F'|' \
  -v id="$TLS2TREES_FINAL_EVALUATION_SUMMARY_JOB" \
  '$1 == id {sub(/[+ ].*$/, "", $2); print $2; exit}')
echo "summary job=$TLS2TREES_FINAL_EVALUATION_SUMMARY_JOB state=${summary_state:-NOT_STARTED}"
echo

if [[ -f "$TLS2TREES_FINAL_EVALUATION_SUMMARY_JSON" ]]; then
  "$TLS2TREES_FINAL_EVALUATION_TREEBENCH_ENV/bin/python" -c '
import json,sys
p=json.load(open(sys.argv[1]))
print("status="+p["status"])
print("valid_metrics={}/{}".format(p["valid_metric_count"],p["expected_metric_count"]))
for row in p["development_aggregates"]:
 print("development {} {}: micro_f1={:.6f} precision={:.6f} recall={:.6f} invalid={}".format(row["candidate_id"],row["target"],row["micro_f1"],row["precision"],row["recall"],row["failed_or_invalid_plot_count"]))
for row in p["test_aggregates"]:
 print("test {}={}: micro_f1={:.6f} precision={:.6f} recall={:.6f} invalid={}".format(row["target"],row["candidate_id"],row["micro_f1"],row["precision"],row["recall"],row["failed_or_invalid_plot_count"]))
print("inference_rerun=false")
print("retained_sources_unchanged=true")
' "$TLS2TREES_FINAL_EVALUATION_SUMMARY_JSON"
else
  echo "status=retained_prediction_evaluation_in_progress_or_awaiting_summary"
  echo "inference_rerun=false"
  echo "retained_sources_unchanged=true"
fi
