#!/usr/bin/env bash

set -euo pipefail

STATE_FILE="${1:?Usage: monitor_for_instance_one_plot_smoke.sh <state-file>}"
source "$STATE_FILE"

date
echo
echo "submission_status=${TREELEARN_SUBMISSION_STATUS:-unknown}"

JOBS="$TREELEARN_INFERENCE_JOB"
if [[ "$TREELEARN_EVALUATION_JOB" =~ ^[0-9]+$ ]]; then
  JOBS="$JOBS,$TREELEARN_EVALUATION_JOB"
fi
if [[ "$TREELEARN_INFERENCE_JOB" =~ ^[0-9]+$ ]]; then
  squeue -j "$JOBS" \
    -o "%.18i %.26j %.10T %.10M %.9L %.19e %R" || true
  echo
  sacct -X -j "$JOBS" \
    --format=JobID,JobName%26,State,Elapsed,Start,End,ExitCode || true
  echo
fi

METRICS="$TREELEARN_TABLE_ROOT/metrics.json"
MATCHES="$TREELEARN_TABLE_ROOT/matches.csv"
UNMATCHED_PREDICTIONS="$TREELEARN_TABLE_ROOT/unmatched_predictions.csv"
UNMATCHED_REFERENCES="$TREELEARN_TABLE_ROOT/unmatched_references.csv"
INFERENCE_STATE="unknown"
if [[ "$TREELEARN_INFERENCE_JOB" =~ ^[0-9]+$ ]]; then
  INFERENCE_STATE=$(sacct -X -n -P -j "$TREELEARN_INFERENCE_JOB" \
    --format=State 2>/dev/null | awk -F'|' 'NF && $1 != "" {print $1; exit}')
  INFERENCE_STATE="${INFERENCE_STATE:-unknown}"
fi
EVALUATION_STATE="not_submitted"
if [[ "$TREELEARN_EVALUATION_JOB" =~ ^[0-9]+$ ]]; then
  EVALUATION_STATE=$(sacct -X -n -P -j "$TREELEARN_EVALUATION_JOB" \
    --format=State 2>/dev/null | awk -F'|' 'NF && $1 != "" {print $1; exit}')
  EVALUATION_STATE="${EVALUATION_STATE:-unknown}"
fi

if [[ "$INFERENCE_STATE" == FAILED* \
      || "$INFERENCE_STATE" == CANCELLED* \
      || "$INFERENCE_STATE" == TIMEOUT* \
      || "$INFERENCE_STATE" == OUT_OF_MEMORY* ]]; then
  echo "status=development-smoke-inference-failed"
  echo "inference_state=$INFERENCE_STATE"
elif [[ "$EVALUATION_STATE" == COMPLETED* \
      && -f "$METRICS" \
      && -f "$MATCHES" \
      && -f "$UNMATCHED_PREDICTIONS" \
      && -f "$UNMATCHED_REFERENCES" ]]; then
  python -c 'import json,sys; p=json.load(open(sys.argv[1])); print("status={}".format(p["status"])); print("f1={:.4f} precision={:.4f} recall={:.4f}".format(p["f1"],p["precision"],p["recall"])); print("predictions={} references={}".format(p["prediction_instance_count"],p["reference_instance_count"])); print("next_gate={}".format(p["next_gate"]))' "$METRICS"
elif [[ "$EVALUATION_STATE" == FAILED* \
        || "$EVALUATION_STATE" == CANCELLED* \
        || "$EVALUATION_STATE" == TIMEOUT* \
        || "$EVALUATION_STATE" == OUT_OF_MEMORY* ]]; then
  echo "status=development-smoke-evaluation-failed"
  echo "evaluation_state=$EVALUATION_STATE"
else
  echo "status=development-smoke-in-progress"
  echo "evaluation_state=$EVALUATION_STATE"
  echo "metrics=$METRICS"
fi

echo "No held-out test job exists in this state file."
