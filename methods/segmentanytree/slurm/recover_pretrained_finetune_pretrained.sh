#!/bin/bash

set -euo pipefail

exec "$(dirname "$0")/recover_three_variation_pretrained.sh" "$@"
