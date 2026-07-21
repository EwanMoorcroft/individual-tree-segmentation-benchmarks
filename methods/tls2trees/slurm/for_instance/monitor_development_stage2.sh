#!/usr/bin/env bash

set -euo pipefail

STATE_FILE="${1:?Usage: monitor_development_stage2.sh <stage2-state-file>}"
test -f "$STATE_FILE"
# shellcheck disable=SC1090
source "$STATE_FILE"

echo "time=$(date --iso-8601=seconds)"
echo "run_id=$TLS2TREES_STAGE2_RUN_ID"
echo "submission_status=$TLS2TREES_STAGE2_SUBMISSION_STATUS"
echo "source_stage1_run_id=$TLS2TREES_STAGE2_SOURCE_STAGE1_RUN_ID"
echo "selected_candidate_ids=p04_min_points_50_lower_band,p02_min_points_50"
echo

JOBS="$TLS2TREES_STAGE2_SEMANTIC_JOB,$TLS2TREES_STAGE2_CANDIDATE_JOB,$TLS2TREES_STAGE2_SUMMARY_JOB"
squeue -j "$JOBS" \
  -o "%.18i %.32j %.10T %.10M %.10L %.28R" 2>/dev/null || true
echo

summarise_array() {
  local label=$1
  local job=$2
  local expected=$3
  local records
  records=$(sacct -X -n -P -j "$job" --format=JobID%30,State 2>/dev/null | \
    awk -F'|' -v id="$job" '
      {gsub(/^[[:space:]]+|[[:space:]]+$/, "", $1)}
      $1 ~ ("^" id "_[0-9]+$") {
        sub(/[+ ].*$/, "", $2); print $1 " " $2
      }')
  local counts
  counts=$(printf '%s\n' "$records" | awk '
    NF == 2 {count[$2]++}
    END {
      first=1
      for (state in count) {
        if (!first) printf ","
        printf "%s=%d", state, count[state]
        first=0
      }
      if (first) printf "not_started=0"
    }')
  local failed
  failed=$(printf '%s\n' "$records" | awk '$2 ~ /^(FAILED|CANCELLED|TIMEOUT|OUT_OF_MEMORY|NODE_FAIL|PREEMPTED)$/ {sub(/^.*_/, "", $1); value=value (value ? "," : "") $1} END {print value}')
  echo "$label job=$job expected=$expected states=$counts failed_tasks=${failed:-NONE}"
}

summarise_array semantic "$TLS2TREES_STAGE2_SEMANTIC_JOB" 21
CANDIDATE_TASK_COUNT="${TLS2TREES_STAGE2_EXPECTED_CANDIDATE_TASKS:-42}"
summarise_array candidate "$TLS2TREES_STAGE2_CANDIDATE_JOB" "$CANDIDATE_TASK_COUNT"
if [[ -n "${TLS2TREES_STAGE2_RECOVERY_FROM_CANDIDATE_JOB:-}" ]]; then
  echo "recovery_from_candidate_job=$TLS2TREES_STAGE2_RECOVERY_FROM_CANDIDATE_JOB tasks=$TLS2TREES_STAGE2_RECOVERY_TASKS"
fi
SUMMARY_STATE=$(sacct -X -n -P -j "$TLS2TREES_STAGE2_SUMMARY_JOB" \
  --format=JobIDRaw,State 2>/dev/null | awk -F'|' \
  -v id="$TLS2TREES_STAGE2_SUMMARY_JOB" \
  '$1 == id {sub(/[+ ].*$/, "", $2); print $2; exit}')
echo "summary job=$TLS2TREES_STAGE2_SUMMARY_JOB state=${SUMMARY_STATE:-NOT_STARTED}"
echo

if [[ -f "$TLS2TREES_STAGE2_SUMMARY_JSON" ]]; then
  "$TLS2TREES_STAGE2_TREEBENCH_ENV/bin/python" -c '
import json,sys
p=json.load(open(sys.argv[1]))
print("status="+p["status"])
print("valid_metrics={}/{}".format(p["valid_metric_count"], p["expected_metric_count"]))
for a in p["aggregates"]:
    print("{} {}: micro_f1={:.6f} mean_plot_f1={:.6f} precision={:.6f} recall={:.6f} invalid={}".format(a["candidate_id"], a["target"], a["micro_f1"], a["mean_plot_f1"], a["precision"], a["recall"], a["failed_or_invalid_plot_count"]))
for target, ranking in p["candidate_rankings_for_review"].items():
    print(target+"_ranking="+",".join(ranking))
print("final_configuration_selected=false")
print("held_out_test_accessed=false")
' "$TLS2TREES_STAGE2_SUMMARY_JSON"
else
  echo "status=stage2_in_progress_or_awaiting_summary"
  echo "final_configuration_selected=false"
  echo "held_out_test_accessed=false"
fi
