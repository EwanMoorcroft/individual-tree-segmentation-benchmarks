#!/usr/bin/env bash
set -euo pipefail
source "${1:?state file required}"
JOBS="$TREELEARN_FINETUNE_PREP_JOB,$TREELEARN_FINETUNE_SMOKE_JOB,$TREELEARN_FINETUNE_FULL_JOB"
date
squeue -j "$JOBS" -o "%.18i %.24j %.10T %.10M %.9L %.19e %R" 2>/dev/null || true
sacct -X -j "$JOBS" --format=JobID,JobName%24,State,Elapsed,Start,End,ExitCode
echo "checkpoint=$TREELEARN_FINETUNE_CHECKPOINT"
[[ -f "$TREELEARN_FINETUNE_CHECKPOINT" ]] && echo checkpoint_ready=true || echo checkpoint_ready=false
echo "No held-out test job exists in this state file."
