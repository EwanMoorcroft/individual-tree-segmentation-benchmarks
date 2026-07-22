#!/usr/bin/env bash

set -euo pipefail

continuous=0
interval=60
while (($#)); do
  case "$1" in
    --continuous) continuous=1 ;;
    --interval) shift; interval="$1" ;;
    *) printf 'unknown argument: %s\n' "$1" >&2; exit 2 ;;
  esac
  shift
done

: "${FORAINET_STATE_FILE:?set FORAINET_STATE_FILE}"
test -f "$FORAINET_STATE_FILE"
# shellcheck disable=SC1090
source "$FORAINET_STATE_FILE"
: "${FORAINET_QUALIFICATION_JOB_ID:?state file lacks job id}"
: "${FORAINET_QUALIFICATION_ROOT:?state file lacks output root}"

report() {
  queue="$(squeue -h -j "$FORAINET_QUALIFICATION_JOB_ID" -o '%T|%M|%l|%R' || true)"
  accounting="$(
    sacct -X -n -j "$FORAINET_QUALIFICATION_JOB_ID" \
      -o JobID,State,ExitCode,Elapsed -P 2>/dev/null | sed -n '1,8p'
  )"
  if test -f "$FORAINET_QUALIFICATION_ROOT/retention.json"; then
    status=COMPLETE
  elif test -n "$queue"; then
    case "${queue%%|*}" in
      PENDING) status=PENDING ;;
      RUNNING|COMPLETING) status=RUNNING ;;
      *) status=UNKNOWN ;;
    esac
  elif printf '%s\n' "$accounting" | grep -Eq 'FAILED|OUT_OF_MEMORY|TIMEOUT|NODE_FAIL|PREEMPTED|CANCELLED|DEPENDENCY'; then
    status=FAILED
  elif printf '%s\n' "$accounting" | grep -q 'COMPLETED'; then
    status=COMPLETED_WAITING_GATE
  else
    status=UNKNOWN
  fi
  printf 'status=%s job_id=%s\n' "$status" "$FORAINET_QUALIFICATION_JOB_ID"
  test -n "$queue" && printf 'queue=%s\n' "$queue"
  test -n "$accounting" && printf 'accounting:\n%s\n' "$accounting"
  printf 'artefacts=%s/4\n' "$(find "$FORAINET_QUALIFICATION_ROOT" -maxdepth 1 -type f 2>/dev/null | wc -l | tr -d ' ')"
}

report
while ((continuous)); do
  sleep "$interval"
  report
done
