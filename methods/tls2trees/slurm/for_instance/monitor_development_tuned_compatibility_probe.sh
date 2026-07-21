#!/usr/bin/env bash

set -euo pipefail

STATE_FILE="${1:?Usage: monitor_development_tuned_compatibility_probe.sh <state-file>}"
test -f "$STATE_FILE"
# shellcheck disable=SC1090
source "$STATE_FILE"

echo "time=$(date --iso-8601=seconds)"
echo "run_id=$TLS2TREES_PROBE_RUN_ID"
echo "submission_status=$TLS2TREES_PROBE_SUBMISSION_STATUS"
echo "source_run_id=$TLS2TREES_PROBE_SOURCE_RUN_ID"
echo

JOBS="$TLS2TREES_PROBE_ARRAY_JOB,$TLS2TREES_PROBE_SUMMARY_JOB"
squeue -j "$JOBS" \
  -o "%.18i %.30j %.10T %.10M %.9L %.19e %R" 2>/dev/null || true
echo
sacct -X -j "$JOBS" \
  --format=JobID,JobName%30,State,Submit,Start,Elapsed,TotalCPU,AllocCPUS,MaxRSS,ExitCode 2>/dev/null || true

if [[ -f "$TLS2TREES_PROBE_SUMMARY_JSON" ]]; then
  echo
  "$TLS2TREES_PROBE_TREEBENCH_ENV/bin/python" -c '
import json,sys
p=json.load(open(sys.argv[1]))
print("status=" + p["status"])
print("viable_candidate_ids=" + ",".join(p["viable_candidate_ids"]))
print("incomplete_candidate_ids=" + ",".join(p["incomplete_candidate_ids"]))
for r in p["candidates"]:
    print("{candidate_index}: {candidate_id} status={status} trees={leaf_off_prediction_file_count} points={leaf_off_prediction_point_count} runtime_s={runtime_seconds} peak_gb={peak_rss_gb}".format(**r))
print("next_action=" + p["next_action"])
' "$TLS2TREES_PROBE_SUMMARY_JSON"
else
  echo
  echo "status=probe-in-progress-or-awaiting-summary"
  echo "On failure, inspect logs/tls2trees_for_instance/tls2trees_dt_probe_<array-job>_<task>.err"
fi

echo "accuracy_metrics_accessed=false"
echo "held_out_test_accessed=false"
