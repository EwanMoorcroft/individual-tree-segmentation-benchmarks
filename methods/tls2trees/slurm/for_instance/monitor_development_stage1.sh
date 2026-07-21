#!/usr/bin/env bash

set -euo pipefail
STATE_FILE="${1:?Usage: monitor_development_stage1.sh <state-file>}"
test -f "$STATE_FILE"
# shellcheck disable=SC1090
source "$STATE_FILE"

echo "time=$(date --iso-8601=seconds)"
echo "run_id=$TLS2TREES_STAGE1_RUN_ID"
echo "submission_status=$TLS2TREES_STAGE1_SUBMISSION_STATUS"
echo "probe_run_id=$TLS2TREES_STAGE1_PROBE_RUN_ID"
echo
JOBS="$TLS2TREES_STAGE1_SEMANTIC_JOB,$TLS2TREES_STAGE1_CANDIDATE_JOB,$TLS2TREES_STAGE1_SUMMARY_JOB"
squeue -j "$JOBS" -o "%.18i %.30j %.10T %.10M %.9L %.19e %R" 2>/dev/null || true
echo
sacct -X -j "$JOBS" \
  --format=JobID,JobName%30,State,Submit,Start,Elapsed,TotalCPU,AllocCPUS,MaxRSS,ExitCode 2>/dev/null || true

if [[ -f "$TLS2TREES_STAGE1_SUMMARY_JSON" ]]; then
  echo
  "$TLS2TREES_STAGE1_TREEBENCH_ENV/bin/python" -c '
import json,sys
p=json.load(open(sys.argv[1]))
print("status="+p["status"])
print("valid_metrics={}/{}".format(p["valid_metric_count"],p["expected_metric_count"]))
for r in p["aggregates"]:
 print("{candidate_id} {target}: micro_f1={micro_f1:.6f} precision={micro_precision:.6f} recall={micro_recall:.6f} mean_plot_f1={mean_plot_f1:.6f} invalid={failed_or_invalid_plot_count}".format(**r))
for target,ranking in p["candidate_rankings_for_review"].items():
 print(target+"_ranking="+",".join(ranking))
print("final_configuration_selected=false")
print("next_gate="+p["next_gate"])
' "$TLS2TREES_STAGE1_SUMMARY_JSON"
else
  echo
  echo "status=stage1-in-progress-or-awaiting-summary"
  echo "On failure inspect only the matching logs/tls2trees_for_instance/*_<jobid>_<task>.err file."
fi
echo "held_out_test_accessed=false"
