#!/usr/bin/env bash

set -u

state_file="${1:-}"
if [[ -z "$state_file" || ! -r "$state_file" ]]; then
  echo "Usage: monitor_forainet_finetune_full.sh STATE_FILE [--watch SECONDS]"
  exit 2
fi
shift
watch_seconds=""
if [[ "${1:-}" == "--watch" ]]; then
  watch_seconds="${2:-20}"
  if [[ ! "$watch_seconds" =~ ^[0-9]+$ || "$watch_seconds" -lt 5 ]]; then
    echo "Watch interval must be an integer of at least 5 seconds."
    exit 2
  fi
elif [[ $# -gt 0 ]]; then
  echo "Usage: monitor_forainet_finetune_full.sh STATE_FILE [--watch SECONDS]"
  exit 2
fi

# shellcheck disable=SC1090
source "$state_file"
job_id="$FORAINET_FINETUNE_FULL_JOB_ID"

render() {
  local status state newest_epoch age candidates latest_candidate
  state="$(
    sacct -X -n -P -j "$job_id" --format=State 2>/dev/null |
      awk -F'|' 'NF && $1 != "" {print $1; exit}'
  )"
  candidates="$(
    find "$FORAINET_FINETUNE_ROOT/full/candidates" \
      -maxdepth 1 -type f -name 'PointGroup-PAPER_epoch_*.pt' \
      2>/dev/null | wc -l
  )"
  latest_candidate="$(
    find "$FORAINET_FINETUNE_ROOT/full/candidates" \
      -maxdepth 1 -type f -name 'PointGroup-PAPER_epoch_*.pt' \
      -printf '%f\n' 2>/dev/null | sort | tail -1
  )"
  newest_epoch="$(
    find "$FORAINET_FINETUNE_ROOT/full" -type f -printf '%T@\n' 2>/dev/null |
      sort -nr | head -1 | cut -d. -f1
  )"
  newest_epoch="${newest_epoch:-0}"
  if [[ "$newest_epoch" -gt 0 ]]; then
    age=$(( $(date +%s) - newest_epoch ))
  else
    age=-1
  fi
  if [[ -f "$FORAINET_FINETUNE_ROOT/full/final_gate.json" ]]; then
    status="COMPLETE"
  elif grep -Eq 'FAILED|CANCELLED|TIMEOUT|OUT_OF_MEMORY|NODE_FAIL|PREEMPTED|BOOT_FAIL|DEADLINE' <<<"$state"; then
    status="FAILED"
  elif squeue -h -j "$job_id" -t RUNNING 2>/dev/null | grep -q .; then
    if [[ "$age" -gt 1800 ]]; then status="STALE"; else status="RUNNING"; fi
  elif squeue -h -j "$job_id" -t PENDING 2>/dev/null | grep -q .; then
    status="PENDING"
  elif [[ ! -d "$FORAINET_FINETUNE_ROOT/full" ]]; then
    status="BLOCKED"
  else
    status="UNKNOWN"
  fi

  printf 'FORAINET FULL FINE-TUNE — %s\n' "$(date '+%F %T %Z')"
  printf 'status=%s run_id=%s job=%s\n' \
    "$status" "$FORAINET_FINETUNE_RUN_ID" "$job_id"
  squeue -j "$job_id" \
    -o "%.12i %.18P %.24j %.11T %.11M %.11l %R" 2>/dev/null
  printf 'scheduler_accounting:\n'
  sacct -X -n -P -j "$job_id" \
    --format=JobID,JobName,Partition,State,ExitCode,Elapsed,MaxRSS,NodeList \
    2>/dev/null | sed '/^[[:space:]]*$/d'
  printf 'candidates=%s/5 latest_candidate=%s final_gate=%s newest_age_seconds=%s\n' \
    "$candidates" "${latest_candidate:-none}" \
    "$([[ -f "$FORAINET_FINETUNE_ROOT/full/final_gate.json" ]] && echo yes || echo no)" \
    "$age"
  printf 'held_out_access=forbidden validation_submitted=no\n'
}

if [[ -z "$watch_seconds" ]]; then
  render
else
  while true; do
    clear
    render
    sleep "$watch_seconds"
  done
fi
