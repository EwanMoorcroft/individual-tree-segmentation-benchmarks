#!/usr/bin/env bash

set -euo pipefail

if [[ "${TLS2TREES_LEAF_SCREEN_PUBLICATION_CONFIRMED:-0}" != "1" ]]; then
  echo "Refusing TLS2trees development leaf-screen publication." >&2
  echo "Set TLS2TREES_LEAF_SCREEN_PUBLICATION_CONFIRMED=1 after reviewing the completed 45/45 development-only screen." >&2
  exit 2
fi

PROJECT_ROOT="${TLS2TREES_PROJECT_ROOT:-$HOME/scratch/tree-seg-benchmark}"
STATE_FILE="${1:-}"
if [[ -z "$STATE_FILE" ]]; then
  POINTER="$PROJECT_ROOT/logs/tls2trees_for_instance/latest_leaf_screen_state_file.txt"
  test -s "$POINTER"
  STATE_FILE=$(tr -d '\r\n' < "$POINTER")
fi
test -f "$STATE_FILE"
SOURCE_STATE_SHA256=$(sha256sum "$STATE_FILE" | awk '{print $1}')

# shellcheck disable=SC1090
source "$STATE_FILE"

RUN_ID="${TLS2TREES_LEAF_SCREEN_RUN_ID:?Leaf-screen state has no run ID}"
SUBMISSION_STATUS="${TLS2TREES_LEAF_SCREEN_SUBMISSION_STATUS:?Leaf-screen state has no submission status}"
SUMMARY_JOB="${TLS2TREES_LEAF_SCREEN_SUMMARY_JOB:?Leaf-screen state has no summary job}"
SOURCE_RUN_ID="${TLS2TREES_LEAF_SCREEN_SOURCE_RUN_ID:?Leaf-screen state has no source run ID}"
SEMANTIC_CACHE_RUN_ID="${TLS2TREES_LEAF_SCREEN_SOURCE_SEMANTIC_CACHE_RUN_ID:?Leaf-screen state has no semantic-cache run ID}"
SUMMARY_JSON="${TLS2TREES_LEAF_SCREEN_SUMMARY_JSON:?Leaf-screen state has no summary JSON}"
PLOT_CSV="${TLS2TREES_LEAF_SCREEN_PLOT_CSV:?Leaf-screen state has no plot CSV}"
CANDIDATE_CSV="${TLS2TREES_LEAF_SCREEN_AGGREGATE_CSV:?Leaf-screen state has no candidate CSV}"
MANIFEST_SHA256="${TLS2TREES_LEAF_SCREEN_MANIFEST_SHA256:?Leaf-screen state has no manifest hash}"
SOURCE_CONFIG_SHA256="${TLS2TREES_LEAF_SCREEN_CONFIG_SHA256:?Leaf-screen state has no config hash}"
DEVELOPMENT_EVIDENCE_SHA256="${TLS2TREES_LEAF_SCREEN_DEVELOPMENT_EVIDENCE_SHA256:?Leaf-screen state has no development-evidence hash}"
TREEBENCH_ENV="${TLS2TREES_LEAF_SCREEN_TREEBENCH_ENV:-$HOME/fastscratch/venvs/treebench}"

test "$SUBMISSION_STATUS" = "development_leaf_screen_chain_submitted"
test -x "$TREEBENCH_ENV/bin/python"
test -s "$SUMMARY_JSON"
test -s "$PLOT_CSV"
test -s "$CANDIDATE_CSV"

SUMMARY_STATE=$(sacct -X -n -P -j "$SUMMARY_JOB" \
  --format=JobIDRaw,State | awk -F'|' -v id="$SUMMARY_JOB" '
    $1 == id {sub(/[+ ].*$/, "", $2); print $2; exit}
  ')
if [[ "$SUMMARY_STATE" != "COMPLETED" ]]; then
  echo "Leaf-screen summary job $SUMMARY_JOB is ${SUMMARY_STATE:-UNKNOWN}, not COMPLETED." >&2
  exit 2
fi

cd "$PROJECT_ROOT"
test -d .git
test -z "$(git status --porcelain)"

FINALISER="methods/tls2trees/scripts/evaluation/finalise_tls2trees_development_leaf_screen.py"
CANDIDATE_CONFIG="methods/tls2trees/configs/for_instance_development_tuned_leaf_screen.yml"
OUTPUT_DIR="methods/tls2trees/examples"
test -f "$FINALISER"
test -f "$CANDIDATE_CONFIG"

for path in \
  "$OUTPUT_DIR/tls2trees_development_leaf_screen_plot_results.csv" \
  "$OUTPUT_DIR/tls2trees_development_leaf_screen_candidate_results.csv" \
  "$OUTPUT_DIR/tls2trees_development_leaf_screen_provenance.json"; do
  test ! -e "$path"
done

"$TREEBENCH_ENV/bin/python" "$FINALISER" \
  --summary-json "$SUMMARY_JSON" \
  --source-plot-csv "$PLOT_CSV" \
  --source-candidate-csv "$CANDIDATE_CSV" \
  --candidate-config "$CANDIDATE_CONFIG" \
  --output-dir "$OUTPUT_DIR" \
  --source-state-sha256 "$SOURCE_STATE_SHA256" \
  --expected-run-id "$RUN_ID" \
  --expected-source-run-id "$SOURCE_RUN_ID" \
  --expected-semantic-cache-run-id "$SEMANTIC_CACHE_RUN_ID" \
  --expected-manifest-sha256 "$MANIFEST_SHA256" \
  --expected-source-config-sha256 "$SOURCE_CONFIG_SHA256" \
  --expected-development-evidence-sha256 "$DEVELOPMENT_EVIDENCE_SHA256"

git diff --check
echo "source_state=$STATE_FILE"
echo "source_state_sha256=$SOURCE_STATE_SHA256"
echo "summary_job=$SUMMARY_JOB"
git status --short -- "$OUTPUT_DIR/tls2trees_development_leaf_screen_"'*'
