#!/usr/bin/env bash
set -euo pipefail
source "${1:?state file required}"
JOBS="$TREELEARN_FINETUNE_VALIDATION_ARRAY_JOB,$TREELEARN_FINETUNE_VALIDATION_SUMMARY_JOB"
date
squeue -j "$JOBS" -o "%.18i %.24j %.10T %.10M %.9L %.19e %R" 2>/dev/null || true
sacct -X -j "$JOBS" --format=JobID,JobName%24,State,Elapsed,Start,End,ExitCode
if [[ -f "$TREELEARN_FINETUNE_VALIDATION_SUMMARY" ]]; then
  python -c 'import json,sys; p=json.load(open(sys.argv[1])); print("status={} plots={} mean_F1={:.6f} micro_F1={:.6f} retention={}".format(p["status"],p["plots"],p["mean_plot_f1"],p["micro_f1"],p["retention_status"]))' "$TREELEARN_FINETUNE_VALIDATION_SUMMARY"
else
  echo validation_summary=pending
fi
echo "No held-out test job exists in this state file."
