#!/usr/bin/env bash

set -u

usage() {
  echo "Usage: monitor_forainet_smoke.sh STATE_FILE [--watch SECONDS]"
}

state_file="${1:-}"
if [[ -z "$state_file" || ! -r "$state_file" ]]; then
  usage
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
  usage
  exit 2
fi

# shellcheck disable=SC1090
source "$state_file"
: "${FORAINET_SMOKE_JOB_ID:?state file lacks FORAINET_SMOKE_JOB_ID}"
: "${FORAINET_RUN_ROOT:?state file lacks FORAINET_RUN_ROOT}"

render() {
  local scheduler_state queue_line newest_epoch age final_gate metrics manifest
  local expected_count completed_count status reason
  queue_line="$(
    squeue -h -j "$FORAINET_SMOKE_JOB_ID" \
      -o "%i|%P|%j|%T|%M|%l|%R" 2>/dev/null | head -1
  )"
  scheduler_state="$(
    sacct -X -n -P -j "$FORAINET_SMOKE_JOB_ID" \
      --format=State 2>/dev/null |
      awk -F'|' 'NF && $1 != "" {print $1; exit}'
  )"
  scheduler_state="${scheduler_state:-UNKNOWN}"
  final_gate="$FORAINET_RUN_ROOT/final_gate.json"
  metrics="$FORAINET_RUN_ROOT/evaluation/metrics.json"
  manifest="$FORAINET_RUN_ROOT/retention/manifest.json"
  expected_count=3
  completed_count=0
  [[ -f "$final_gate" ]] && completed_count=$((completed_count + 1))
  [[ -f "$metrics" ]] && completed_count=$((completed_count + 1))
  [[ -f "$manifest" ]] && completed_count=$((completed_count + 1))

  newest_epoch=0
  if [[ -d "$FORAINET_RUN_ROOT" ]]; then
    newest_epoch="$(
      find "$FORAINET_RUN_ROOT" -type f -printf '%T@\n' 2>/dev/null |
        sort -nr | head -1 | cut -d. -f1
    )"
    newest_epoch="${newest_epoch:-0}"
  fi
  if [[ "$newest_epoch" -gt 0 ]]; then
    age=$(( $(date +%s) - newest_epoch ))
  else
    age=-1
  fi

  reason=""
  case "$scheduler_state" in
    PENDING*)
      status="PENDING"
      ;;
    RUNNING*)
      if [[ "$age" -gt 1800 ]]; then
        status="STALE"
        reason="no expected output changed for more than 30 minutes"
      else
        status="RUNNING"
      fi
      ;;
    COMPLETED*)
      if [[ -f "$final_gate" && -f "$metrics" && -f "$manifest" ]]; then
        status="COMPLETE"
      else
        status="COMPLETED_WAITING_GATE"
        reason="Slurm completed but required final artefacts are missing"
      fi
      ;;
    FAILED*|CANCELLED*|TIMEOUT*|OUT_OF_MEMORY*|NODE_FAIL*|PREEMPTED*|BOOT_FAIL*|DEADLINE*)
      status="FAILED"
      reason="$scheduler_state"
      ;;
    UNKNOWN)
      if [[ -f "$final_gate" && -f "$metrics" && -f "$manifest" ]]; then
        status="COMPLETE"
      elif [[ ! -d "$FORAINET_RUN_ROOT" ]]; then
        status="BLOCKED"
        reason="run root has not been created"
      else
        status="UNKNOWN"
      fi
      ;;
    *)
      status="UNKNOWN"
      reason="$scheduler_state"
      ;;
  esac

  printf 'FORAINET SMOKE — %s\n' "$(date '+%F %T %Z')"
  printf 'status=%s\n' "$status"
  printf 'job_id=%s scheduler_state=%s\n' \
    "$FORAINET_SMOKE_JOB_ID" "$scheduler_state"
  if [[ -n "$queue_line" ]]; then
    printf 'squeue=%s\n' "$queue_line"
  fi
  printf 'scheduler_accounting:\n'
  sacct -X -n -P -j "$FORAINET_SMOKE_JOB_ID" \
    --format=JobID,JobName,Partition,State,ExitCode,Elapsed,NodeList \
    2>/dev/null | sed '/^[[:space:]]*$/d'
  printf 'resource_accounting:\n'
  sacct -n -P -j "$FORAINET_SMOKE_JOB_ID" \
    --format=JobID,State,Elapsed,MaxRSS,MaxVMSize \
    2>/dev/null | sed '/^[[:space:]]*$/d'
  printf 'artefacts=%s/%s newest_age_seconds=%s\n' \
    "$completed_count" "$expected_count" "$age"
  [[ -n "$reason" ]] && printf 'reason=%s\n' "$reason"
  printf 'run_root=%s\n' "$FORAINET_RUN_ROOT"
  printf 'final_gate=%s metrics=%s manifest=%s\n' \
    "$([[ -f "$final_gate" ]] && echo yes || echo no)" \
    "$([[ -f "$metrics" ]] && echo yes || echo no)" \
    "$([[ -f "$manifest" ]] && echo yes || echo no)"
  printf 'held_out_access=forbidden\n'
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
