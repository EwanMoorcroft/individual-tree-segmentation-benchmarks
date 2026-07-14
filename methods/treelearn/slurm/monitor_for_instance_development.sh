#!/usr/bin/env bash

set -euo pipefail

STATE_FILE="${1:?Usage: monitor_for_instance_development.sh <state-file>}"
source "$STATE_FILE"

date
echo
echo "run_id=$TREELEARN_DEV_RUN_ID"
echo "submission_status=${TREELEARN_DEV_SUBMISSION_STATUS:-unknown}"

JOBS=()
for job in \
  "${TREELEARN_DEV_PREP_JOB:-}" \
  "${TREELEARN_DEV_ARRAY_JOB:-}" \
  "${TREELEARN_DEV_SUMMARY_JOB:-}" \
  "${TREELEARN_DEV_GATE_JOB:-}"; do
  if [[ "$job" =~ ^[0-9]+$ ]]; then
    JOBS+=("$job")
  fi
done

if ((${#JOBS[@]})); then
  JOB_LIST=$(IFS=,; echo "${JOBS[*]}")
  squeue -j "$JOB_LIST" \
    -o "%.18i %.27j %.10T %.10M %.9L %.19e %R" || true
  echo
  sacct -X -j "$JOB_LIST" \
    --format=JobID,JobName%27,State,Elapsed,Start,End,ExitCode || true
  echo
fi

MANIFEST_COUNT=0
if [[ -f "${TREELEARN_DEV_MANIFEST:-}" ]]; then
  MANIFEST_COUNT=$(awk 'END {print NR > 0 ? NR - 1 : 0}' "$TREELEARN_DEV_MANIFEST")
fi
COMPLETED_METRICS=0
DOCUMENTED_FAILURES=0
if [[ -d "${TREELEARN_DEV_TABLE_ROOT:-}/per_plot" ]]; then
  COMPLETED_METRICS=$(find "$TREELEARN_DEV_TABLE_ROOT/per_plot" \
    -mindepth 2 -maxdepth 2 -type f -name metrics.json | wc -l)
  DOCUMENTED_FAILURES=$(find "$TREELEARN_DEV_TABLE_ROOT/per_plot" \
    -mindepth 2 -maxdepth 2 -type f -name status.json | wc -l)
fi
echo "development_manifest=$MANIFEST_COUNT/21"
echo "completed_metrics=$COMPLETED_METRICS/21"
echo "documented_failures=$DOCUMENTED_FAILURES"

if [[ -f "${TREELEARN_DEV_RUN_SUMMARY:-}" ]]; then
  python -c 'import json,sys; p=json.load(open(sys.argv[1])); print("status={}".format(p.get("status", "unknown"))); print("completed_plots={}/{}".format(p.get("completed_plots", "?"), p.get("expected_plots", 21))); print("documented_failures={}".format(p.get("documented_failures", "?"))); print("retention_status={}".format(p.get("retention_status", "unknown"))); print("next_gate={}".format(p.get("next_gate", "unknown")))' "$TREELEARN_DEV_RUN_SUMMARY"
elif ((COMPLETED_METRICS + DOCUMENTED_FAILURES == 21)); then
  echo "status=awaiting-development-summary"
else
  echo "status=development-run-in-progress"
fi

if [[ -f "${TREELEARN_DEV_SITE_SUMMARY:-}" ]]; then
  echo
  echo "Site results:"
  column -s, -t < "$TREELEARN_DEV_SITE_SUMMARY"
fi
if [[ -f "${TREELEARN_DEV_DEVELOPMENT_SUMMARY:-}" ]]; then
  echo
  echo "Development result:"
  column -s, -t < "$TREELEARN_DEV_DEVELOPMENT_SUMMARY"
fi

echo "No held-out test job exists in this state file."
