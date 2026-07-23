#!/usr/bin/env bash

set -u

state_file="${1:-}"
if [[ -z "$state_file" || ! -r "$state_file" ]]; then
  echo "Usage: monitor_forainet_development.sh STATE_FILE [--watch SECONDS]"
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
  echo "Usage: monitor_forainet_development.sh STATE_FILE [--watch SECONDS]"
  exit 2
fi

# shellcheck disable=SC1090
source "$state_file"
job_ids="$FORAINET_DEVELOPMENT_PREP_JOB_ID,$FORAINET_DEVELOPMENT_ARRAY_JOB_ID,$FORAINET_DEVELOPMENT_SUMMARY_JOB_ID"

render() {
  local gates metrics manifests newest_epoch age status states
  gates="$(find "$FORAINET_DEVELOPMENT_ROOT/plots" -mindepth 2 -maxdepth 2 -name final_gate.json 2>/dev/null | wc -l)"
  metrics="$(find "$FORAINET_DEVELOPMENT_ROOT/plots" -mindepth 3 -maxdepth 3 -path '*/evaluation/metrics.json' 2>/dev/null | wc -l)"
  manifests="$(find "$FORAINET_DEVELOPMENT_ROOT/plots" -mindepth 3 -maxdepth 3 -path '*/retention/manifest.json' 2>/dev/null | wc -l)"
  states="$(
    sacct -X -n -P -j "$job_ids" --format=State 2>/dev/null |
      awk -F'|' 'NF && $1 != "" {print $1}'
  )"
  newest_epoch="$(
    find "$FORAINET_DEVELOPMENT_ROOT" -type f -printf '%T@\n' 2>/dev/null |
      sort -nr | head -1 | cut -d. -f1
  )"
  newest_epoch="${newest_epoch:-0}"
  if [[ "$newest_epoch" -gt 0 ]]; then
    age=$(( $(date +%s) - newest_epoch ))
  else
    age=-1
  fi

  if [[ -f "$FORAINET_DEVELOPMENT_ROOT/final_gate.json" ]]; then
    status="COMPLETE"
  elif grep -Eq 'FAILED|CANCELLED|TIMEOUT|OUT_OF_MEMORY|NODE_FAIL|PREEMPTED|BOOT_FAIL|DEADLINE' <<<"$states"; then
    status="FAILED"
  elif squeue -h -j "$job_ids" -t RUNNING 2>/dev/null | grep -q .; then
    if [[ "$age" -gt 1800 ]]; then status="STALE"; else status="RUNNING"; fi
  elif squeue -h -j "$job_ids" -t PENDING 2>/dev/null | grep -q .; then
    status="PENDING"
  elif [[ "$gates" -eq 21 && ! -f "$FORAINET_DEVELOPMENT_ROOT/final_gate.json" ]]; then
    status="COMPLETED_WAITING_GATE"
  elif [[ ! -d "$FORAINET_DEVELOPMENT_ROOT" ]]; then
    status="BLOCKED"
  else
    status="UNKNOWN"
  fi

  printf 'FORAINET DEVELOPMENT — %s\n' "$(date '+%F %T %Z')"
  printf 'status=%s run_id=%s\n' "$status" "$FORAINET_DEVELOPMENT_RUN_ID"
  printf 'jobs=prep:%s array:%s summary:%s\n' \
    "$FORAINET_DEVELOPMENT_PREP_JOB_ID" \
    "$FORAINET_DEVELOPMENT_ARRAY_JOB_ID" \
    "$FORAINET_DEVELOPMENT_SUMMARY_JOB_ID"
  squeue -j "$job_ids" -o "%.12i %.18P %.24j %.11T %.11M %.11l %R" 2>/dev/null
  printf 'scheduler_accounting:\n'
  sacct -X -n -P -j "$job_ids" \
    --format=JobID,JobName,Partition,State,ExitCode,Elapsed,NodeList \
    2>/dev/null | sed '/^[[:space:]]*$/d'
  printf 'plot_gates=%s/21 metrics=%s/21 manifests=%s/21 newest_age_seconds=%s\n' \
    "$gates" "$metrics" "$manifests" "$age"
  printf 'overall_final_gate=%s held_out_access=forbidden\n' \
    "$([[ -f "$FORAINET_DEVELOPMENT_ROOT/final_gate.json" ]] && echo yes || echo no)"
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
