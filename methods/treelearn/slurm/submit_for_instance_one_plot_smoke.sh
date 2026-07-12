#!/usr/bin/env bash

set -euo pipefail

cd "$HOME/scratch/tree-seg-benchmark"

if [[ "${TREELEARN_SMOKE_CONFIRMED:-0}" != "1" ]]; then
  echo "Refusing TreeLearn development smoke submission."
  echo "Set TREELEARN_SMOKE_CONFIRMED=1 after reviewing the pinned route."
  exit 2
fi

TREELEARN_ENV="${TREELEARN_ENV:-$HOME/fastscratch/venvs/treelearn}"
TREELEARN_REPO="${TREELEARN_REPO:-$HOME/fastscratch/external/TreeLearn}"
TREELEARN_CHECKPOINT="${TREELEARN_CHECKPOINT:-$HOME/fastscratch/treelearn_checkpoints/model_weights_20241213.pth}"
TREELEARN_DATASET_ROOT="${TREELEARN_DATASET_ROOT:-$HOME/data/datasets/for_instance/FORinstance_dataset}"
EXPECTED_COMMIT="fd240ce7caa4c444fe3418aca454dc578bc557d4"
EXPECTED_CHECKPOINT_MD5="56a3d78f689ae7f1190906b975700311"
BENCHMARK_COMMIT=$(git rev-parse HEAD)
if [[ -n "$(git status --porcelain)" ]]; then
  echo "Refusing a dirty benchmark checkout; commit or remove tracked/untracked changes first."
  exit 1
fi

test -x "$TREELEARN_ENV/bin/python"
test -f "$TREELEARN_ENV/.treelearn_setup_complete"
test -d "$TREELEARN_REPO/.git"
test -f "$TREELEARN_CHECKPOINT"
test -f "$TREELEARN_DATASET_ROOT/CULS/plot_1_annotated.las"
test -f "$TREELEARN_DATASET_ROOT/data_split_metadata.csv"
test -x "$HOME/fastscratch/venvs/treebench/bin/python"

ACTUAL_COMMIT=$(git -C "$TREELEARN_REPO" rev-parse HEAD)
test "$ACTUAL_COMMIT" = "$EXPECTED_COMMIT"
test -z "$(git -C "$TREELEARN_REPO" status --porcelain)"
CHECKPOINT_MD5=$(md5sum "$TREELEARN_CHECKPOINT" | cut -d ' ' -f 1)
if [[ "$CHECKPOINT_MD5" != "$EXPECTED_CHECKPOINT_MD5" ]]; then
  echo "TreeLearn checkpoint MD5 $CHECKPOINT_MD5 is not the official default $EXPECTED_CHECKPOINT_MD5."
  exit 1
fi
CHECKPOINT_SHA256=$(sha256sum "$TREELEARN_CHECKPOINT" | cut -d ' ' -f 1)

"$TREELEARN_ENV/bin/python" - "$TREELEARN_REPO" <<'PY'
from pathlib import Path
import sys
import tree_learn

repo = Path(sys.argv[1]).resolve()
package = Path(tree_learn.__file__).resolve()
if not package.is_relative_to(repo):
    raise SystemExit(f"tree_learn imported from {package}, not pinned repo {repo}")
print(f"treelearn_package={package}")
PY

STAMP=$(date -u +%Y%m%d_%H%M%S)
RUN_ID="${TREELEARN_RUN_ID:-treelearn_for-instance_published_pretrained_dev_smoke_$STAMP}"
if [[ ! "$RUN_ID" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]]; then
  echo "Unsafe TreeLearn run ID: $RUN_ID"
  exit 2
fi

RUNTIME_ROOT="$HOME/scratch/tree-seg-benchmark/data/interim/treelearn/for_instance/one_plot_smokes/$RUN_ID"
PREDICTION_ROOT="$HOME/scratch/tree-seg-benchmark/data/predictions/treelearn/for_instance_smokes/$RUN_ID"
METADATA_ROOT="$HOME/scratch/tree-seg-benchmark/results/metadata/treelearn_for_instance/one_plot_smokes/$RUN_ID"
TABLE_ROOT="$HOME/scratch/tree-seg-benchmark/results/tables/treelearn_for_instance/one_plot_smokes/$RUN_ID"
STATE_FILE="$HOME/fastscratch/treelearn_dev_smoke_${RUN_ID}.env"
for path in "$RUNTIME_ROOT" "$PREDICTION_ROOT" "$METADATA_ROOT" "$TABLE_ROOT" "$STATE_FILE"; do
  if [[ -e "$path" ]]; then
    echo "Refusing existing TreeLearn run path: $path"
    exit 1
  fi
done

mkdir -p logs/treelearn_for_instance

INFERENCE_JOB="not_submitted"
EVALUATION_JOB="not_submitted"
SUBMISSION_STATUS="preflight_completed"

write_state() {
  {
    printf 'TREELEARN_RUN_ID=%q\n' "$RUN_ID"
    printf 'TREELEARN_INFERENCE_JOB=%q\n' "$INFERENCE_JOB"
    printf 'TREELEARN_EVALUATION_JOB=%q\n' "$EVALUATION_JOB"
    printf 'TREELEARN_SUBMISSION_STATUS=%q\n' "$SUBMISSION_STATUS"
    printf 'TREELEARN_CHECKPOINT_MD5=%q\n' "$CHECKPOINT_MD5"
    printf 'TREELEARN_CHECKPOINT_SHA256=%q\n' "$CHECKPOINT_SHA256"
    printf 'TREELEARN_BENCHMARK_COMMIT=%q\n' "$BENCHMARK_COMMIT"
    printf 'TREELEARN_RUNTIME_ROOT=%q\n' "$RUNTIME_ROOT"
    printf 'TREELEARN_PREDICTION_ROOT=%q\n' "$PREDICTION_ROOT"
    printf 'TREELEARN_METADATA_ROOT=%q\n' "$METADATA_ROOT"
    printf 'TREELEARN_TABLE_ROOT=%q\n' "$TABLE_ROOT"
  } > "$STATE_FILE"
}

INFERENCE_JOB=$(sbatch --parsable \
  --export="ALL,TREELEARN_SMOKE_CONFIRMED=1,TREELEARN_ENV=$TREELEARN_ENV,TREELEARN_REPO=$TREELEARN_REPO,TREELEARN_CHECKPOINT=$TREELEARN_CHECKPOINT,TREELEARN_DATASET_ROOT=$TREELEARN_DATASET_ROOT,TREELEARN_RUN_ID=$RUN_ID,TREELEARN_EXPECTED_BENCHMARK_COMMIT=$BENCHMARK_COMMIT,TREELEARN_EXPECTED_CHECKPOINT_MD5=$EXPECTED_CHECKPOINT_MD5,TREELEARN_EXPECTED_CHECKPOINT_SHA256=$CHECKPOINT_SHA256" \
  methods/treelearn/slurm/run_for_instance_one_plot_smoke.sbatch)
SUBMISSION_STATUS="inference_submitted"
write_state

if ! EVALUATION_JOB=$(sbatch --parsable \
  --dependency="afterok:$INFERENCE_JOB" \
  --kill-on-invalid-dep=yes \
  --export="ALL,TREELEARN_RUN_ID=$RUN_ID,TREELEARN_EXPECTED_BENCHMARK_COMMIT=$BENCHMARK_COMMIT" \
  methods/treelearn/slurm/evaluate_for_instance_one_plot_smoke.sbatch); then
  scancel "$INFERENCE_JOB" 2>/dev/null || true
  EVALUATION_JOB="submission_failed"
  SUBMISSION_STATUS="evaluation_submission_failed_inference_cancelled"
  write_state
  echo "Evaluation submission failed; inference job $INFERENCE_JOB was cancelled."
  echo "state_file=$STATE_FILE"
  exit 1
fi
SUBMISSION_STATUS="chain_submitted"
write_state

echo "run_id=$RUN_ID"
echo "inference_job=$INFERENCE_JOB evaluation_job=$EVALUATION_JOB"
echo "state_file=$STATE_FILE"
echo "benchmark_commit=$BENCHMARK_COMMIT"
echo "checkpoint_md5=$CHECKPOINT_MD5"
echo "checkpoint_sha256=$CHECKPOINT_SHA256"
echo "No training, full development array or held-out test job was submitted."
