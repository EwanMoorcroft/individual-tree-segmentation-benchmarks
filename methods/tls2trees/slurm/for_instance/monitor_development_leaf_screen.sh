#!/usr/bin/env bash

set -euo pipefail
STATE_FILE="${1:?Usage: monitor_development_leaf_screen.sh <leaf-screen-state-file>}"
test -f "$STATE_FILE"
# shellcheck disable=SC1090
source "$STATE_FILE"

echo "time=$(date --iso-8601=seconds)"
echo "run_id=$TLS2TREES_LEAF_SCREEN_RUN_ID"
echo "submission_status=$TLS2TREES_LEAF_SCREEN_SUBMISSION_STATUS"
echo "source_stage1_run_id=$TLS2TREES_LEAF_SCREEN_SOURCE_RUN_ID"
echo "semantic_cache_reused=true"
echo
JOBS="$TLS2TREES_LEAF_SCREEN_CANDIDATE_JOB,$TLS2TREES_LEAF_SCREEN_SUMMARY_JOB"
squeue -j "$JOBS" -o "%.18i %.30j %.10T %.10M %.9L %.19e %R" 2>/dev/null || true
echo
sacct -X -j "$JOBS" \
  --format=JobID,JobName%30,State,Submit,Start,Elapsed,TotalCPU,AllocCPUS,MaxRSS,ExitCode \
  2>/dev/null || true

if [[ -f "$TLS2TREES_LEAF_SCREEN_SUMMARY_JSON" ]]; then
  echo
  "$TLS2TREES_LEAF_SCREEN_TREEBENCH_ENV/bin/python" -c '
import json,sys
p=json.load(open(sys.argv[1]))
print("status="+p["status"])
print("valid_metrics={}/{}".format(p["valid_metric_count"],p["expected_metric_count"]))
for row in p["aggregates"]:
 print("{candidate_id}: micro_f1={micro_f1:.6f} mean_plot_f1={mean_plot_f1:.6f} precision={micro_precision:.6f} recall={micro_recall:.6f} invalid={failed_or_invalid_plot_count}".format(**row))
print("leaf_on_ranking="+",".join(p["candidate_ranking_for_review"]))
print("top_three_for_review="+",".join(p["top_three_candidate_ids_for_review"]))
print("final_configuration_selected=false")
print("next_gate="+p["next_gate"])
' "$TLS2TREES_LEAF_SCREEN_SUMMARY_JSON"
else
  echo
  echo "status=development-leaf-screen-in-progress-or-awaiting-summary"
  echo "On failure inspect only logs/tls2trees_for_instance/tls2trees_dt_leaf_screen_<array-job>_<task>.err"
fi
echo "held_out_test_accessed=false"
