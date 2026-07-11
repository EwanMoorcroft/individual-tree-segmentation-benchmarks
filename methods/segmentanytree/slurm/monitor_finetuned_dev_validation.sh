#!/bin/bash

set -euo pipefail

STATE_FILE="${1:?Pass the fine-tuned development state file.}"
STATE_FILE=$(realpath "$STATE_FILE")
source "$STATE_FILE"
INTERVAL="${SEGMENTANYTREE_MONITOR_INTERVAL:-30}"
JOB_IDS="$SAT_FINETUNE_DEV_SMOKE_JOB,$SAT_FINETUNE_DEV_TRAIN_JOB,$SAT_FINETUNE_DEV_INFERENCE_JOB,$SAT_FINETUNE_DEV_EVALUATION_JOB,$SAT_FINETUNE_DEV_GATE_JOB"

while true; do
  clear
  date
  echo
  squeue -j "$JOB_IDS" -o "%.18i %.27j %.10T %.9M %.9L %.19e %R"
  echo
  sacct -X -j "$JOB_IDS" --format=JobID,JobName%27,State,Elapsed,Start,End,ExitCode
  echo
  metric_count=0
  [[ -d "$SAT_FINETUNE_DEV_METRICS" ]] && \
    metric_count=$(find "$SAT_FINETUNE_DEV_METRICS" -type f -name '*.json' | wc -l)
  checkpoint_count=0
  [[ -d "$SAT_FINETUNE_DEV_CHECKPOINT_ROOT" ]] && \
    checkpoint_count=$(find "$SAT_FINETUNE_DEV_CHECKPOINT_ROOT" -type f -name 'PointGroup-PAPER.pt' | wc -l)
  echo "development_metrics=$metric_count/5"
  echo "trained_checkpoints=$checkpoint_count"
  echo "summary=$SAT_FINETUNE_DEV_SUMMARY"
  echo "held_out_test_jobs=0"

  gate_state=$(sacct -X -n -j "$SAT_FINETUNE_DEV_GATE_JOB" -o State 2>/dev/null | awk 'NF {print $1; exit}')
  if [[ "$gate_state" =~ ^(COMPLETED|FAILED|CANCELLED|TIMEOUT|OUT_OF_MEMORY|NODE_FAIL)$ ]]; then
    break
  fi
  sleep "$INTERVAL"
done
