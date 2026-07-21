#!/usr/bin/env bash

set -euo pipefail

PROJECT_ROOT="${TLS2TREES_PROJECT_ROOT:-$HOME/scratch/tree-seg-benchmark}"
STATE_FILE="${1:-}"
if [[ -z "$STATE_FILE" ]]; then
  POINTER="$PROJECT_ROOT/logs/tls2trees_for_instance/latest_finalisation_state_file.txt"
  test -s "$POINTER"
  STATE_FILE=$(tr -d '\r\n' < "$POINTER")
fi
test -f "$STATE_FILE"
# shellcheck disable=SC1090
source "$STATE_FILE"
cd "$PROJECT_ROOT"

echo "time=$(date --iso-8601=seconds)"
echo "run_id=$TLS2TREES_FINALIZE_RUN_ID"
echo "job_id=$TLS2TREES_FINALIZE_JOB"
squeue -j "$TLS2TREES_FINALIZE_JOB" \
  -o "%.18i %.30j %.10T %.10M %.10L %.30R" 2>/dev/null || true

STATE=$(sacct -X -n -P -j "$TLS2TREES_FINALIZE_JOB" \
  --format=JobIDRaw,State 2>/dev/null | awk -F'|' \
  -v id="$TLS2TREES_FINALIZE_JOB" '$1==id {sub(/[+ ].*$/, "", $2); print $2; exit}')
echo "accounting_state=${STATE:-NOT_STARTED}"

OUT="logs/tls2trees_for_instance/tls2trees_dt_finalise_${TLS2TREES_FINALIZE_JOB}.out"
ERR="logs/tls2trees_for_instance/tls2trees_dt_finalise_${TLS2TREES_FINALIZE_JOB}.err"
if [[ -s "$OUT" ]]; then
  echo "=== STDOUT ==="
  tail -n 40 "$OUT"
fi
if [[ -s "$ERR" ]]; then
  echo "=== STDERR ==="
  tail -n 80 "$ERR"
fi
if [[ -s "$TLS2TREES_FINALIZE_RECEIPT_JSON" ]]; then
  echo "status=TLS2TREES_RESULTS_READY_TO_COMMIT"
  echo "receipt=$TLS2TREES_FINALIZE_RECEIPT_JSON"
  git status --short
else
  echo "status=tls2trees_result_finalisation_in_progress_or_failed"
fi
