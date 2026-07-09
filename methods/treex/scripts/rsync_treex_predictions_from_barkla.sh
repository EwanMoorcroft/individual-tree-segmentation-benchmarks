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
LOCAL_ROOT="${TREEX_LOCAL_PREDICTION_ROOT:-$REPO_ROOT/local_outputs/treex_predictions}"
CONTROL_PATH="/tmp/treex-barkla-%C"
SSH_OPTS=(
  -o ControlMaster=auto
  -o ControlPersist=10m
  -o ControlPath="${CONTROL_PATH}"
)
SSH_CMD=(ssh "${SSH_OPTS[@]}")
RSYNC_RSH="ssh ${SSH_OPTS[*]}"

mkdir -p "${LOCAL_ROOT}"

cleanup() {
  "${SSH_CMD[@]}" -O exit "${REMOTE_USER}@${REMOTE_HOST}" >/dev/null 2>&1 || true
}

trap cleanup EXIT

"${SSH_CMD[@]}" -MNf "${REMOTE_USER}@${REMOTE_HOST}"

rsync -avh --progress -e "${RSYNC_RSH}" \
  "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_ROOT}/data/predictions/treex/" \
  "${LOCAL_ROOT}/"

printf '\nTreeX predictions copied to:\n%s\n' "${LOCAL_ROOT}"
printf 'These files should remain gitignored and must not be committed.\n'
