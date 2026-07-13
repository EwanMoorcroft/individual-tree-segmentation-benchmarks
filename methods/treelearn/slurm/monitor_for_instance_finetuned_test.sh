#!/usr/bin/env bash
set -euo pipefail
STATE_FILE="${1:?Pass the TreeLearn fine-tuned test state file.}"
source "$(realpath "$STATE_FILE")"
JOBS="$TREELEARN_TEST_PREP_JOB,$TREELEARN_TEST_ARRAY_JOB,$TREELEARN_TEST_SUMMARY_JOB,$TREELEARN_TEST_GATE_JOB"
date
echo
echo "submission_status=$TREELEARN_TEST_SUBMISSION_STATUS"
squeue -j "$JOBS" -o "%.18i %.25j %.10T %.9M %.9L %.19e %R" 2>/dev/null || true
echo
sacct -X -j "$JOBS" --format=JobID,JobName%25,State,Elapsed,Start,End,ExitCode
echo
COUNT=0
if [[ -d "$TREELEARN_TEST_TABLE_ROOT/per_plot" ]]; then
  COUNT=$(find "$TREELEARN_TEST_TABLE_ROOT/per_plot" -type f -name metrics.json | wc -l)
fi
echo "completed_metrics=$COUNT/11"
echo "final_summary=$TREELEARN_TEST_FINAL_SUMMARY"
if [[ -f "$TREELEARN_TEST_FINAL_SUMMARY" ]]; then
  "$HOME/fastscratch/venvs/treebench/bin/python" - "$TREELEARN_TEST_FINAL_SUMMARY" <<'PY'
import csv, sys
row = next(csv.DictReader(open(sys.argv[1])))
print(
    "result_status={} mean_F1={:.6f} micro_F1={:.6f} precision={:.6f} recall={:.6f}".format(
        row["result_status"], float(row["mean_plot_f1"]), float(row["micro_f1"]),
        float(row["micro_precision"]), float(row["micro_recall"])
    )
)
PY
fi
if [[ -f "$TREELEARN_TEST_COMPLETION_GATE" ]]; then
  echo "completion_status=verified"
fi
