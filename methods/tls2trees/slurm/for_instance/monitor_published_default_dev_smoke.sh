#!/usr/bin/env bash

set -euo pipefail

STATE_FILE="${1:?Usage: monitor_published_default_dev_smoke.sh <state-file>}"
if [[ ! -f "$STATE_FILE" ]]; then
  echo "State file does not exist: $STATE_FILE" >&2
  exit 2
fi
# shellcheck disable=SC1090
source "$STATE_FILE"

echo "run_id=$TLS2TREES_SMOKE_RUN_ID"
echo "submission_status=${TLS2TREES_SMOKE_SUBMISSION_STATUS:-unknown}"

JOB_VARS=(
  TLS2TREES_SMOKE_INVENTORY_JOB
  TLS2TREES_SMOKE_CONVERT_JOB
  TLS2TREES_SMOKE_SEMANTIC_JOB
  TLS2TREES_SMOKE_INSTANCE_JOB
  TLS2TREES_SMOKE_ADAPTER_JOB
  TLS2TREES_SMOKE_LEAF_OFF_EVALUATE_JOB
  TLS2TREES_SMOKE_LEAF_ON_EVALUATE_JOB
  TLS2TREES_SMOKE_GATE_JOB
  TLS2TREES_SMOKE_SUMMARY_JOB
)
JOBS=()
for variable in "${JOB_VARS[@]}"; do
  value="${!variable:-}"
  if [[ "$value" =~ ^[0-9]+$ ]]; then
    JOBS+=("$value")
  fi
done

if ((${#JOBS[@]})); then
  JOB_LIST=$(IFS=,; echo "${JOBS[*]}")
  squeue -j "$JOB_LIST" \
    -o "%.18i %.28j %.10T %.10M %.9L %.19e %R" 2>/dev/null || true
  echo
  sacct -X -j "$JOB_LIST" \
    --format=JobID,JobName%28,State,Submit,Start,Elapsed,TotalCPU,AllocCPUS,MaxRSS,ExitCode 2>/dev/null || true
fi

SUMMARY="$TLS2TREES_SMOKE_WORKFLOW_TABLE_ROOT/run_summary.json"
GATE="$TLS2TREES_SMOKE_WORKFLOW_METADATA_ROOT/gate.json"
if [[ -f "$SUMMARY" ]]; then
  python -c 'import json,sys; p=json.load(open(sys.argv[1])); print("status="+str(p.get("status","unknown"))); print("safe_for_scoring="+str(p.get("safe_for_scoring",False)).lower()); [print("{}: f1={} precision={} recall={}".format(t,m.get("f1"),m.get("precision"),m.get("recall"))) for t,m in sorted(p.get("targets",{}).items())]' "$SUMMARY"
elif [[ -f "$GATE" ]]; then
  python -c 'import json,sys; p=json.load(open(sys.argv[1])); print("status="+str(p.get("status","unknown"))); print("manual_alignment_review_required="+str(p.get("manual_alignment_review_required",False)).lower()); print("full_development_authorised="+str(p.get("full_development_authorised",False)).lower()); print("held_out_test_authorised="+str(p.get("held_out_test_authorised",False)).lower())' "$GATE"
else
  echo "status=development-smoke-in-progress-or-failed"
  echo "If a job failed, inspect only its matching logs/tls2trees_for_instance/*_<jobid>.err file."
fi

echo "No tuning, full-development array or held-out-test job exists in this state file."
