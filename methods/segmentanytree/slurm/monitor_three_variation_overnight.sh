#!/bin/bash

set -euo pipefail

PROJECT_ROOT="${SEGMENTANYTREE_PROJECT_ROOT:-$HOME/scratch/tree-seg-benchmark}"
FOLLOW=0
STATE_FILE=""
for arg in "$@"; do
  if [[ "$arg" == "--follow" ]]; then
    FOLLOW=1
  elif [[ -z "$STATE_FILE" ]]; then
    STATE_FILE="$arg"
  else
    echo "Usage: monitor_three_variation_overnight.sh [STATE_FILE] [--follow]" >&2
    exit 2
  fi
done
STATE_FILE="${STATE_FILE:-$HOME/fastscratch/segmentanytree_three_variation_latest.env}"
STATE_FILE=$(readlink -f "$STATE_FILE")
test -f "$STATE_FILE"
source "$STATE_FILE"

job_ids=(
  "$SAT_THREE_CHECKPOINT_JOB"
  "$SAT_THREE_PRETRAINED_PILOT_INFER"
  "$SAT_THREE_PRETRAINED_PILOT_EVAL"
  "$SAT_THREE_PRETRAINED_PILOT_GATE"
  "$SAT_THREE_PRETRAINED_REST_INFER"
  "$SAT_THREE_PRETRAINED_REST_EVAL"
  "$SAT_THREE_PRETRAINED_FINAL_GATE"
  "$SAT_THREE_FINETUNE_SMOKE_JOB"
  "$SAT_THREE_FINETUNE_TRAIN_JOB"
  "$SAT_THREE_FINETUNE_VALIDATION_INFER"
  "$SAT_THREE_FINETUNE_VALIDATION_EVAL"
  "$SAT_THREE_FINETUNE_VALIDATION_GATE"
  "$SAT_THREE_FINETUNE_TEST_INFER"
  "$SAT_THREE_FINETUNE_TEST_EVAL"
  "$SAT_THREE_FINETUNE_FINAL_GATE"
  "$SAT_THREE_SUMMARY_JOB"
)
job_csv=$(IFS=,; echo "${job_ids[*]}")

job_state() {
  sacct -X -n -j "$1" -o State 2>/dev/null | awk 'NF {print $1; exit}' || true
}

metric_count() {
  if [[ ! -d "$1" ]]; then
    echo 0
    return 0
  fi
  find "$1" -type f -name '*.json' 2>/dev/null | wc -l | tr -d ' '
}

render() {
  local now queue_counts train_state summary_state epoch_line epoch elapsed_raw eta_epoch eta_text
  now=$(date '+%F %T')
  queue_counts=$(
    { squeue -h -j "$job_csv" -o '%T' 2>/dev/null || true; } |
      sort | uniq -c | awk '{printf "%s=%s ", $2, $1}'
  )
  train_state=$(job_state "$SAT_THREE_FINETUNE_TRAIN_JOB")
  summary_state=$(job_state "$SAT_THREE_SUMMARY_JOB")
  epoch_line=$(grep -ahoE 'EPOCH [0-9]+ / [0-9]+' \
    "$PROJECT_ROOT/logs/segmentanytree_for_instance/for_sat_train_full_${SAT_THREE_FINETUNE_TRAIN_JOB}.err" \
    2>/dev/null | tail -n 1 || true)
  epoch=$(awk '{print $2}' <<<"$epoch_line")
  eta_epoch="$SAT_THREE_EXPECTED_FINISH_EPOCH"
  if [[ -n "$epoch" && "$epoch" =~ ^[1-9][0-9]*$ && "$train_state" == RUNNING* ]]; then
    elapsed_raw=$(sacct -X -n -j "$SAT_THREE_FINETUNE_TRAIN_JOB" -o ElapsedRaw 2>/dev/null | awk 'NF {print $1; exit}')
    if [[ "$elapsed_raw" =~ ^[1-9][0-9]*$ ]]; then
      eta_epoch=$(( $(date +%s) + elapsed_raw * (SAT_THREE_FINETUNE_EPOCHS - epoch) / epoch + 7200 ))
    fi
  fi
  eta_text=$(date -d "@$eta_epoch" '+%F %T')

  echo "$now"
  echo "queue: ${queue_counts:-none}"
  echo "pretrained: metrics=$(metric_count "$SAT_THREE_PRETRAINED_METRICS")/11"
  echo "fine-tune: train=${train_state:-PENDING} epoch=${epoch:-0}/$SAT_THREE_FINETUNE_EPOCHS dev_metrics=$(metric_count "$SAT_THREE_FINETUNE_VALIDATION_METRICS")/5 test_metrics=$(metric_count "$SAT_THREE_FINETUNE_TEST_METRICS")/11"
  echo "ETA: $eta_text (queue delays can move this)"
  echo "final: ${summary_state:-PENDING}"
  if [[ -f "$SAT_THREE_SUMMARY" ]]; then
    echo
    awk -F, 'NR==1 {printf "%-23s %8s %8s %8s\n", "variant", "F1", "prec", "rec"; next} {printf "%-23s %8.4f %8.4f %8.4f\n", $1, $9, $10, $11}' "$SAT_THREE_SUMMARY"
  fi

  if sacct -X -n -j "$job_csv" -o State 2>/dev/null | grep -Eq 'FAILED|TIMEOUT|OUT_OF_MEMORY|CANCELLED|NODE_FAIL'; then
    echo "failure detected; inspect only the failed job named by: sacct -X -j $job_csv -o JobID,State,ExitCode"
    return 2
  fi
  [[ "$summary_state" == COMPLETED* ]] && return 1
  return 0
}

if [[ "$FOLLOW" -eq 0 ]]; then
  render || status=$?
  [[ "${status:-0}" -eq 1 ]] && exit 0
  exit "${status:-0}"
fi

while true; do
  clear
  status=0
  render || status=$?
  if [[ "$status" -eq 1 ]]; then
    exit 0
  fi
  if [[ "$status" -eq 2 ]]; then
    exit 2
  fi
  sleep 60
done
