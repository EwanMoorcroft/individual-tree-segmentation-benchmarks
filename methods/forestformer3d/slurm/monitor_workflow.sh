#!/usr/bin/env bash

set -euo pipefail

usage() {
  echo "Usage: monitor_workflow.sh <state.env> [--watch [seconds]]" >&2
}

STATE_FILE="${1:-}"
if [[ -z "$STATE_FILE" || ! -f "$STATE_FILE" ]]; then
  usage
  exit 2
fi
shift

WATCH_SECONDS=0
if [[ "${1:-}" == "--watch" ]]; then
  WATCH_SECONDS="${2:-30}"
  if [[ ! "$WATCH_SECONDS" =~ ^[1-9][0-9]*$ ]]; then
    echo "Watch interval must be a positive integer." >&2
    exit 2
  fi
  shift
  if [[ $# -gt 0 ]]; then
    shift
  fi
fi
if [[ $# -ne 0 ]]; then
  usage
  exit 2
fi

# State files are created only by method-owned submit wrappers using printf %q.
# shellcheck disable=SC1090
source "$STATE_FILE"

: "${FF3D_WORKFLOW:?State file is missing FF3D_WORKFLOW}"
: "${FF3D_JOB_IDS:?State file is missing FF3D_JOB_IDS}"
: "${FF3D_RUN_ROOT:?State file is missing FF3D_RUN_ROOT}"

is_terminal_state() {
  case "$1" in
    COMPLETED*|FAILED*|CANCELLED*|TIMEOUT*|OUT_OF_MEMORY*|NODE_FAIL*|PREEMPTED*)
      return 0
      ;;
    *)
      return 1
      ;;
  esac
}

render() {
  local all_terminal=1
  local job_id state reason expected

  date -Is
  echo "workflow=$FF3D_WORKFLOW"
  echo "run_root=$FF3D_RUN_ROOT"
  echo "job_ids=$FF3D_JOB_IDS"
  echo "cancel_command=scancel ${FF3D_JOB_IDS//,/ }"
  echo
  echo "QUEUE (waiting is not runtime)"
  squeue -j "$FF3D_JOB_IDS" \
    -o "%.18i %.28j %.12P %.2t %.10M %.19S %.30R" 2>/dev/null || true
  echo
  echo "ACCOUNTING"
  sacct -X -j "$FF3D_JOB_IDS" \
    --format=JobID,JobName%28,Partition,State,ExitCode,Submit,Start,Elapsed,NodeList \
    2>/dev/null || true
  echo
  echo "EXPECTED FILES"
  IFS='|' read -r -a expected_files <<< "${FF3D_EXPECTED_FILES:-}"
  if [[ ${#expected_files[@]} -eq 0 || -z "${expected_files[0]:-}" ]]; then
    echo "none_declared"
  else
    for expected in "${expected_files[@]}"; do
      if [[ -f "$expected" ]]; then
        printf 'present size=%s path=%s\n' "$(stat -c %s "$expected")" "$expected"
      else
        printf 'missing path=%s\n' "$expected"
      fi
    done
  fi

  IFS=',' read -r -a job_ids <<< "$FF3D_JOB_IDS"
  for job_id in "${job_ids[@]}"; do
    state="$(
      sacct -X -n -P -j "$job_id" --format=State 2>/dev/null \
        | awk -F'|' 'NF && $1 != "" {print $1; exit}'
    )"
    state="${state:-UNKNOWN}"
    reason="$(
      squeue -h -j "$job_id" -o "%R" 2>/dev/null | head -n 1
    )"
    if [[ "$reason" == *"DependencyNeverSatisfied"* ]]; then
      printf 'blocked_dependency job_id=%s reason=%s\n' "$job_id" "$reason"
      if [[ "${FF3D_CANCEL_INVALID_DEPENDENCIES:-0}" == "1" ]]; then
        if scancel "$job_id"; then
          printf 'cancelled_invalid_dependency job_id=%s\n' "$job_id"
        else
          printf 'dependency_job_already_terminal job_id=%s\n' "$job_id"
        fi
      fi
    elif ! is_terminal_state "$state"; then
      all_terminal=0
    fi
  done
  if [[ -n "${FF3D_EXPECTED_TASKS_ROOT:-}" && -n "${FF3D_EXPECTED_TASK_COUNT:-}" ]]; then
    completed_tasks=0
    failed_tasks=0
    if [[ -d "$FF3D_EXPECTED_TASKS_ROOT" ]]; then
      completed_tasks="$(find "$FF3D_EXPECTED_TASKS_ROOT" -mindepth 2 -maxdepth 2 -type f -name task.complete | wc -l | tr -d ' ')"
      failed_tasks="$(find "$FF3D_EXPECTED_TASKS_ROOT" -mindepth 2 -maxdepth 2 -type f -name task.failed | wc -l | tr -d ' ')"
    fi
    printf 'tasks_complete=%s/%s tasks_failed=%s\n' \
      "$completed_tasks" "$FF3D_EXPECTED_TASK_COUNT" "$failed_tasks"
  fi
  if [[ "$all_terminal" == "1" ]]; then
    echo
    echo "monitor_status=terminal"
    return 0
  fi
  echo
  echo "monitor_status=active"
  return 1
}

if [[ "$WATCH_SECONDS" == "0" ]]; then
  render || true
  exit 0
fi

while true; do
  clear 2>/dev/null || true
  if render; then
    break
  fi
  echo
  echo "Refreshing in ${WATCH_SECONDS}s; Ctrl-C stops monitoring but does not cancel jobs."
  sleep "$WATCH_SECONDS"
done
