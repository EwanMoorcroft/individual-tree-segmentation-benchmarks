#!/usr/bin/env bash

set -euo pipefail

if [[ "${TLS2TREES_LEAF_SCREEN_PUBLICATION_CONFIRMED:-0}" != "1" ]]; then
  echo "Refusing TLS2trees development leaf-screen publication." >&2
  echo "Set TLS2TREES_LEAF_SCREEN_PUBLICATION_CONFIRMED=1 after reviewing the completed 45/45 development-only screen." >&2
  exit 2
fi

RECOVERY_CONFIRMED="${TLS2TREES_LEAF_SCREEN_PUBLICATION_RECOVERY_CONFIRMED:-0}"
if [[ "$RECOVERY_CONFIRMED" != "0" && "$RECOVERY_CONFIRMED" != "1" ]]; then
  echo "TLS2TREES_LEAF_SCREEN_PUBLICATION_RECOVERY_CONFIRMED must be 0 or 1." >&2
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
STATE_SNAPSHOT=$(
  mktemp "${TMPDIR:-/tmp}/tls2trees_leaf_screen_state.XXXXXX"
)
cleanup_state_snapshot() {
  rm -f -- "$STATE_SNAPSHOT"
}
trap cleanup_state_snapshot EXIT
cp -- "$STATE_FILE" "$STATE_SNAPSHOT"
SOURCE_STATE_SHA256=$(sha256sum "$STATE_SNAPSHOT" | awk '{print $1}')

# shellcheck disable=SC1090
source "$STATE_SNAPSHOT"
POST_SOURCE_STATE_SHA256=$(sha256sum "$STATE_FILE" | awk '{print $1}')
if [[ "$POST_SOURCE_STATE_SHA256" != "$SOURCE_STATE_SHA256" ]]; then
  echo "Leaf-screen state changed while finalisation was starting." >&2
  exit 2
fi

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
SOURCE_BENCHMARK_COMMIT="${TLS2TREES_LEAF_SCREEN_BENCHMARK_COMMIT:?Leaf-screen state has no benchmark commit}"
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
git cat-file -e "${SOURCE_BENCHMARK_COMMIT}^{commit}"
PUBLICATION_BENCHMARK_COMMIT=$(git rev-parse HEAD)
if git merge-base --is-ancestor \
  "$SOURCE_BENCHMARK_COMMIT" "$PUBLICATION_BENCHMARK_COMMIT"; then
  SOURCE_HISTORY_RELATION="ancestor"
else
  DIVERGED_SOURCE_COMMIT="${TLS2TREES_LEAF_SCREEN_DIVERGED_SOURCE_COMMIT:-}"
  if [[ "$DIVERGED_SOURCE_COMMIT" != "$SOURCE_BENCHMARK_COMMIT" ]]; then
    echo "Leaf-screen source commit is not an ancestor of the publication commit." >&2
    echo "source_benchmark_commit=$SOURCE_BENCHMARK_COMMIT" >&2
    echo "publication_benchmark_commit=$PUBLICATION_BENCHMARK_COMMIT" >&2
    echo "If this is the reviewed closure branch, set" >&2
    echo "TLS2TREES_LEAF_SCREEN_DIVERGED_SOURCE_COMMIT=$SOURCE_BENCHMARK_COMMIT" >&2
    echo "to approve only this exact frozen source commit." >&2
    exit 2
  fi
  SOURCE_HISTORY_RELATION="reviewed_divergence"
fi

FINALISER="methods/tls2trees/scripts/evaluation/finalise_tls2trees_development_leaf_screen.py"
CANDIDATE_CONFIG="methods/tls2trees/configs/for_instance_development_tuned_leaf_screen.yml"
OUTPUT_DIR="methods/tls2trees/examples"
STAGE_DIR="$OUTPUT_DIR/.tls2trees_development_leaf_screen_publication.staging"
PUBLIC_OUTPUTS=(
  "$OUTPUT_DIR/tls2trees_development_leaf_screen_plot_results.csv"
  "$OUTPUT_DIR/tls2trees_development_leaf_screen_candidate_results.csv"
  "$OUTPUT_DIR/tls2trees_development_leaf_screen_provenance.json"
)
test -f "$FINALISER"
test -f "$CANDIDATE_CONFIG"

RECOVERY_PATHS=("${PUBLIC_OUTPUTS[@]}")
for path in "${PUBLIC_OUTPUTS[@]}"; do
  RECOVERY_PATHS+=("$STAGE_DIR/${path##*/}")
done

if [[ "$RECOVERY_CONFIRMED" == "0" ]]; then
  if [[ -n "$(git status --porcelain=v1 --untracked-files=all)" ]]; then
    echo "Refusing leaf-screen publication from a dirty worktree." >&2
    echo "Use recovery confirmation only for an interrupted leaf-screen publication." >&2
    exit 2
  fi
else
  WORKTREE_VIOLATIONS=()
  while IFS= read -r -d '' ENTRY; do
    STATUS=${ENTRY:0:2}
    PATHNAME=${ENTRY:3}
    PATH_ALLOWED=0
    for ALLOWED_PATH in "${RECOVERY_PATHS[@]}"; do
      if [[ "$PATHNAME" == "$ALLOWED_PATH" ]]; then
        PATH_ALLOWED=1
        break
      fi
    done
    if [[ "$STATUS" != " M" && "$STATUS" != "??" ]] || \
      ((PATH_ALLOWED == 0)) || [[ -L "$PATHNAME" ]]; then
      WORKTREE_VIOLATIONS+=("$STATUS $PATHNAME")
    fi
  done < <(git status --porcelain=v1 -z --untracked-files=all)

  if ((${#WORKTREE_VIOLATIONS[@]})); then
    echo "Refusing recovery with staged, deleted, renamed, symlink, or unrelated changes:" >&2
    printf '%s\n' "${WORKTREE_VIOLATIONS[@]}" >&2
    exit 2
  fi
fi

"$TREEBENCH_ENV/bin/python" "$FINALISER" \
  --project-root "$PWD" \
  --summary-json "$SUMMARY_JSON" \
  --source-plot-csv "$PLOT_CSV" \
  --source-candidate-csv "$CANDIDATE_CSV" \
  --candidate-config "$CANDIDATE_CONFIG" \
  --output-dir "$OUTPUT_DIR" \
  --source-state-sha256 "$SOURCE_STATE_SHA256" \
  --source-benchmark-commit "$SOURCE_BENCHMARK_COMMIT" \
  --publication-benchmark-commit "$PUBLICATION_BENCHMARK_COMMIT" \
  --recovery-confirmed "$RECOVERY_CONFIRMED" \
  --expected-run-id "$RUN_ID" \
  --expected-source-run-id "$SOURCE_RUN_ID" \
  --expected-semantic-cache-run-id "$SEMANTIC_CACHE_RUN_ID" \
  --expected-manifest-sha256 "$MANIFEST_SHA256" \
  --expected-source-config-sha256 "$SOURCE_CONFIG_SHA256" \
  --expected-development-evidence-sha256 "$DEVELOPMENT_EVIDENCE_SHA256"

git diff --check
echo "source_state=$STATE_FILE"
echo "source_state_sha256=$SOURCE_STATE_SHA256"
echo "source_benchmark_commit=$SOURCE_BENCHMARK_COMMIT"
echo "publication_benchmark_commit=$PUBLICATION_BENCHMARK_COMMIT"
echo "source_history_relation=$SOURCE_HISTORY_RELATION"
echo "recovery_confirmed=$RECOVERY_CONFIRMED"
echo "summary_job=$SUMMARY_JOB"
git status --short -- "$OUTPUT_DIR/tls2trees_development_leaf_screen_"'*'
