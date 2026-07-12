#!/usr/bin/env bash
set -euo pipefail
source "${1:?state file required}"
JOBS="${TREELEARN_LONG_PREP_JOB},${TREELEARN_LONG_CROPS_JOB},${TREELEARN_LONG_CONSOLIDATE_JOB},${TREELEARN_LONG_TRAIN_JOB},${TREELEARN_LONG_VALIDATION_JOB},${TREELEARN_LONG_SELECTION_JOB},${TREELEARN_LONG_GATE_JOB}"
date
echo
echo "submission_status=${TREELEARN_LONG_SUBMISSION_STATUS:-unknown}"
echo
squeue -j "$JOBS" \
  -o "%.18i %.24j %.10T %.10M %.9L %.19e %R" 2>/dev/null || true
echo
for ENTRY in \
  "prep:$TREELEARN_LONG_PREP_JOB" \
  "crops:$TREELEARN_LONG_CROPS_JOB" \
  "merge:$TREELEARN_LONG_CONSOLIDATE_JOB" \
  "train:$TREELEARN_LONG_TRAIN_JOB" \
  "validate:$TREELEARN_LONG_VALIDATION_JOB" \
  "select:$TREELEARN_LONG_SELECTION_JOB" \
  "gate:$TREELEARN_LONG_GATE_JOB"
do
  LABEL=${ENTRY%%:*}
  JOB=${ENTRY#*:}
  STATES=$(sacct -X -n -j "$JOB" --format=State -P 2>/dev/null \
    | cut -d '|' -f 1 | sed 's/[[:space:]]//g' | sed '/^$/d' \
    | sort | uniq -c | tr '\n' ',' | sed 's/,$//')
  printf '%-9s job=%-10s %s\n' "$LABEL" "$JOB" "${STATES:-unknown}"
done
echo
if [[ -f "$TREELEARN_LONG_SELECTION_FREEZE" ]]; then
  python - "$TREELEARN_LONG_SELECTION_FREEZE" <<'PY'
import json, sys
p = json.load(open(sys.argv[1]))
s = p["selected"]
print("selection_status={} config={} epoch={} average_mean_F1={:.6f} average_micro_F1={:.6f}".format(
    p["status"], s["config_id"], s["epoch"],
    s["average_mean_plot_f1"], s["average_micro_f1"]
))
PY
else
  echo selection_status=pending
fi
if [[ -f "$TREELEARN_LONG_SELECTED_FREEZE" ]]; then
  python - "$TREELEARN_LONG_SELECTED_FREEZE" <<'PY'
import json, sys
p = json.load(open(sys.argv[1]))
print("selected_status={} checkpoint_ready=true checkpoint_sha256={}".format(
    p["status"], p["checkpoint_sha256"]
))
PY
else
  echo selected_status=pending
fi
echo "No held-out test job exists in this state file."
