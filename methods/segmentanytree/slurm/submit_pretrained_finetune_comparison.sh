#!/bin/bash

set -euo pipefail

if [[ "${SEGMENTANYTREE_PRETRAINED_FINETUNE_CONFIRMED:-0}" != "1" ]]; then
  echo "Set SEGMENTANYTREE_PRETRAINED_FINETUNE_CONFIRMED=1 after reviewing the workflow." >&2
  exit 2
fi

export SEGMENTANYTREE_THREE_VARIATION_CONFIRMED=1
exec "$(dirname "$0")/submit_three_variation_overnight.sh" "$@"
