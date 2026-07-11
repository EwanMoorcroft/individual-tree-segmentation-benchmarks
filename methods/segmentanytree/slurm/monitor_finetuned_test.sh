#!/bin/bash

set -euo pipefail

STATE_FILE="${1:?Pass the fine-tuned test state file.}"
STATE_FILE=$(realpath "$STATE_FILE")
source "$STATE_FILE"
INTERVAL="${SEGMENTANYTREE_MONITOR_INTERVAL:-30}"
JOB_IDS="$SAT_FINETUNED_TEST_INFERENCE_JOB,$SAT_FINETUNED_TEST_EVALUATION_JOB,$SAT_FINETUNED_TEST_GATE_JOB"

while true; do
  clear
  date
  echo
  squeue -j "$JOB_IDS" -o "%.18i %.27j %.10T %.9M %.9L %.19e %R"
  echo
  sacct -X -j "$JOB_IDS" --format=JobID,JobName%27,State,Elapsed,Start,End,ExitCode
  echo
  metric_count=0
  [[ -d "$SAT_FINETUNED_TEST_METRICS" ]] && \
    metric_count=$(find "$SAT_FINETUNED_TEST_METRICS" -type f -name '*.json' | wc -l)
  echo "aligned_metrics=$metric_count/11"
  echo "summary=$SAT_FINETUNED_TEST_SUMMARY"

  gate_state=$(sacct -X -n -j "$SAT_FINETUNED_TEST_GATE_JOB" -o State 2>/dev/null | awk 'NF {print $1; exit}')
  if [[ "$gate_state" =~ ^(COMPLETED|FAILED|CANCELLED|TIMEOUT|OUT_OF_MEMORY|NODE_FAIL)$ ]]; then
    break
  fi
  sleep "$INTERVAL"
done
