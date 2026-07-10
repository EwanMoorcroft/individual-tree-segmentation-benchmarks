#!/bin/bash

set -euo pipefail

exec "$(dirname "$0")/monitor_three_variation_overnight.sh" "$@"
