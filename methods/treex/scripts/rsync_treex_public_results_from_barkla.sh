#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/../../.." && pwd)
: "${TREEX_BARKLA_USER:?Set TREEX_BARKLA_USER to your Barkla username.}"
: "${TREEX_BARKLA_HOST:?Set TREEX_BARKLA_HOST to the Barkla login host.}"
: "${TREEX_BARKLA_ROOT:?Set TREEX_BARKLA_ROOT to the remote tree-seg-benchmark checkout.}"
REMOTE_USER="$TREEX_BARKLA_USER"
REMOTE_HOST="$TREEX_BARKLA_HOST"
REMOTE_ROOT="$TREEX_BARKLA_ROOT"
LOCAL_METHOD_ROOT="${TREEX_LOCAL_METHOD_ROOT:-$REPO_ROOT/methods/treex}"
CONTROL_PATH="/tmp/treex-barkla-%C"
SSH_OPTS=(
  -o ControlMaster=auto
  -o ControlPersist=10m
  -o ControlPath="${CONTROL_PATH}"
)
SSH_CMD=(ssh "${SSH_OPTS[@]}")
RSYNC_RSH="ssh ${SSH_OPTS[*]}"

mkdir -p "${LOCAL_METHOD_ROOT}/examples"
mkdir -p "${LOCAL_METHOD_ROOT}/plots"

cleanup() {
  "${SSH_CMD[@]}" -O exit "${REMOTE_USER}@${REMOTE_HOST}" >/dev/null 2>&1 || true
}

trap cleanup EXIT

"${SSH_CMD[@]}" -MNf "${REMOTE_USER}@${REMOTE_HOST}"

rsync -avh --progress -e "${RSYNC_RSH}" \
  "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_ROOT}/results/treex_for_instance/treex_dev_full_summary.csv" \
  "${LOCAL_METHOD_ROOT}/examples/"

rsync -avh --progress -e "${RSYNC_RSH}" \
  "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_ROOT}/results/treex_for_instance/treex_test_full_summary.csv" \
  "${LOCAL_METHOD_ROOT}/examples/"

rsync -avh --progress -e "${RSYNC_RSH}" \
  "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_ROOT}/results/treex_for_instance/treex_combined_dev_test_summary.csv" \
  "${LOCAL_METHOD_ROOT}/examples/"

rsync -avh --progress -e "${RSYNC_RSH}" \
  "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_ROOT}/results/treex_for_instance/treex_split_summary.csv" \
  "${LOCAL_METHOD_ROOT}/examples/"

rsync -avh --progress -e "${RSYNC_RSH}" \
  "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_ROOT}/results/treex_for_instance/treex_site_summary.csv" \
  "${LOCAL_METHOD_ROOT}/examples/"

rsync -avh --progress -e "${RSYNC_RSH}" \
  "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_ROOT}/results/treex_for_instance/treex_best_plots_by_strict_f1.csv" \
  "${LOCAL_METHOD_ROOT}/examples/"

rsync -avh --progress -e "${RSYNC_RSH}" \
  "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_ROOT}/results/treex_for_instance/treex_worst_plots_by_strict_f1.csv" \
  "${LOCAL_METHOD_ROOT}/examples/"

rsync -avh --progress -e "${RSYNC_RSH}" \
  "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_ROOT}/results/treex_for_instance/plots/treex_labelled_mask_f1_by_plot.png" \
  "${LOCAL_METHOD_ROOT}/plots/"

rsync -avh --progress -e "${RSYNC_RSH}" \
  "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_ROOT}/results/treex_for_instance/plots/treex_predicted_vs_reference_counts.png" \
  "${LOCAL_METHOD_ROOT}/plots/"

rsync -avh --progress -e "${RSYNC_RSH}" \
  "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_ROOT}/results/treex_for_instance/plots/treex_runtime_vs_strict_f1.png" \
  "${LOCAL_METHOD_ROOT}/plots/"

rsync -avh --progress -e "${RSYNC_RSH}" \
  "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_ROOT}/results/treex_for_instance/plots/treex_strict_f1_by_plot.png" \
  "${LOCAL_METHOD_ROOT}/plots/"

printf '\nTreeX public-safe results copied into:\n%s\n' "${LOCAL_METHOD_ROOT}"
