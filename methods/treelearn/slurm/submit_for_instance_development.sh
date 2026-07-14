#!/usr/bin/env bash

set -euo pipefail

if [[ "${TREELEARN_FULL_DEV_CONFIRMED:-0}" != "1" ]]; then
  echo "Set TREELEARN_FULL_DEV_CONFIRMED=1 after reviewing the development-only route." >&2
  exit 2
fi
if [[ "${TREELEARN_DEV_ALIGNMENT_REVIEW_CONFIRMED:-0}" != "1" ]]; then
  echo "Set TREELEARN_DEV_ALIGNMENT_REVIEW_CONFIRMED=1 after accepting the smoke alignment evidence." >&2
  exit 2
fi

PROJECT_ROOT="${TREELEARN_PROJECT_ROOT:-$HOME/scratch/tree-seg-benchmark}"
TREELEARN_ENV="${TREELEARN_ENV:-$HOME/fastscratch/venvs/treelearn}"
TREELEARN_REPO="${TREELEARN_REPO:-$HOME/fastscratch/external/TreeLearn}"
TREELEARN_CHECKPOINT="${TREELEARN_CHECKPOINT:-$HOME/fastscratch/treelearn_checkpoints/model_weights_20241213.pth}"
TREELEARN_DATASET_ROOT="${TREELEARN_DATASET_ROOT:-$HOME/data/datasets/for_instance/FORinstance_dataset}"
ACCEPTED_SMOKE_RUN_ID="${TREELEARN_ACCEPTED_SMOKE_RUN_ID:-}"
EXPECTED_SMOKE_RUN_ID="treelearn_for-instance_published_pretrained_dev_smoke_20260712_135205"
EXPECTED_UPSTREAM_COMMIT="fd240ce7caa4c444fe3418aca454dc578bc557d4"
EXPECTED_CHECKPOINT_MD5="56a3d78f689ae7f1190906b975700311"
EXPECTED_CHECKPOINT_SHA256="5df2f92828f92755bc12e114eaebe83f7ecea94a74c25a6170b68844cc5e19bb"
MIN_FREE_BYTES="${TREELEARN_DEV_MIN_FREE_BYTES:-85899345920}"

if [[ "$ACCEPTED_SMOKE_RUN_ID" != "$EXPECTED_SMOKE_RUN_ID" ]]; then
  echo "TREELEARN_ACCEPTED_SMOKE_RUN_ID must equal $EXPECTED_SMOKE_RUN_ID." >&2
  exit 2
fi
if [[ ! "$MIN_FREE_BYTES" =~ ^[1-9][0-9]*$ ]]; then
  echo "TREELEARN_DEV_MIN_FREE_BYTES must be a positive integer." >&2
  exit 2
fi

cd "$PROJECT_ROOT"
mkdir -p logs/treelearn_for_instance

BENCHMARK_COMMIT=$(git rev-parse HEAD)
if [[ -n "$(git status --porcelain)" ]]; then
  echo "Refusing a dirty benchmark checkout." >&2
  exit 2
fi
test -x "$TREELEARN_ENV/bin/python"
test -f "$TREELEARN_ENV/.treelearn_setup_complete"
test -x "$HOME/fastscratch/venvs/treebench/bin/python"
test -d "$TREELEARN_REPO/.git"
test -f "$TREELEARN_CHECKPOINT"
test -d "$TREELEARN_DATASET_ROOT"
test -f "$TREELEARN_DATASET_ROOT/data_split_metadata.csv"
test "$(git -C "$TREELEARN_REPO" rev-parse HEAD)" = "$EXPECTED_UPSTREAM_COMMIT"
test -z "$(git -C "$TREELEARN_REPO" status --porcelain)"

CHECKPOINT_MD5=$(md5sum "$TREELEARN_CHECKPOINT" | cut -d ' ' -f 1)
CHECKPOINT_SHA256=$(sha256sum "$TREELEARN_CHECKPOINT" | cut -d ' ' -f 1)
if [[ "$CHECKPOINT_MD5" != "$EXPECTED_CHECKPOINT_MD5" \
      || "$CHECKPOINT_SHA256" != "$EXPECTED_CHECKPOINT_SHA256" ]]; then
  echo "TreeLearn checkpoint identity differs from the accepted smoke." >&2
  exit 2
fi

"$TREELEARN_ENV/bin/python" \
  methods/treelearn/scripts/validate_treelearn_environment.py \
  --treelearn-repo "$TREELEARN_REPO"

SCRATCH_FREE=$(df -PB1 "$PROJECT_ROOT" | awk 'NR == 2 {print $4}')
if [[ ! "$SCRATCH_FREE" =~ ^[0-9]+$ ]] || ((SCRATCH_FREE < MIN_FREE_BYTES)); then
  echo "Need at least $MIN_FREE_BYTES free bytes on scratch; found ${SCRATCH_FREE:-unknown}." >&2
  exit 2
fi

STAMP="${TREELEARN_DEV_STAMP:-$(date -u +%Y%m%d_%H%M%S)}"
if [[ ! "$STAMP" =~ ^[0-9]{8}_[0-9]{6}$ ]]; then
  echo "TREELEARN_DEV_STAMP must use YYYYMMDD_HHMMSS." >&2
  exit 2
fi
RUN_ID="treelearn_for-instance_published_pretrained_development_$STAMP"
RUNTIME_ROOT="$PROJECT_ROOT/data/interim/treelearn/for_instance/development_runs/$RUN_ID"
PREDICTION_ROOT="$PROJECT_ROOT/data/predictions/treelearn/for_instance_development/$RUN_ID"
METADATA_ROOT="$PROJECT_ROOT/results/metadata/treelearn_for_instance/development_runs/$RUN_ID"
TABLE_ROOT="$PROJECT_ROOT/results/tables/treelearn_for_instance/development_runs/$RUN_ID"
ACCEPTANCE="$METADATA_ROOT/accepted_smoke.json"
MANIFEST_CSV="$METADATA_ROOT/development_manifest.csv"
MANIFEST_JSON="$METADATA_ROOT/development_manifest.json"
RUN_SUMMARY="$TABLE_ROOT/run_summary.json"
SITE_SUMMARY="$TABLE_ROOT/site_summary.csv"
DEVELOPMENT_SUMMARY="$TABLE_ROOT/development_summary.csv"
STATE_FILE="$HOME/fastscratch/treelearn_development_${RUN_ID}.env"

for target in "$RUNTIME_ROOT" "$PREDICTION_ROOT" "$METADATA_ROOT" "$TABLE_ROOT" "$STATE_FILE"; do
  if [[ -e "$target" ]]; then
    echo "Refusing existing TreeLearn development target: $target" >&2
    exit 2
  fi
done

PREP_JOB="not_submitted"
ARRAY_JOB="not_submitted"
SUMMARY_JOB="not_submitted"
GATE_JOB="not_submitted"
SUBMISSION_STATUS="preflight_completed"
SUBMITTED_JOBS=()

write_state() {
  {
    printf 'TREELEARN_DEV_RUN_ID=%q\n' "$RUN_ID"
    printf 'TREELEARN_DEV_SUBMISSION_STATUS=%q\n' "$SUBMISSION_STATUS"
    printf 'TREELEARN_DEV_PREP_JOB=%q\n' "$PREP_JOB"
    printf 'TREELEARN_DEV_ARRAY_JOB=%q\n' "$ARRAY_JOB"
    printf 'TREELEARN_DEV_SUMMARY_JOB=%q\n' "$SUMMARY_JOB"
    printf 'TREELEARN_DEV_GATE_JOB=%q\n' "$GATE_JOB"
    printf 'TREELEARN_DEV_BENCHMARK_COMMIT=%q\n' "$BENCHMARK_COMMIT"
    printf 'TREELEARN_DEV_CHECKPOINT_MD5=%q\n' "$CHECKPOINT_MD5"
    printf 'TREELEARN_DEV_CHECKPOINT_SHA256=%q\n' "$CHECKPOINT_SHA256"
    printf 'TREELEARN_DEV_ACCEPTED_SMOKE_RUN_ID=%q\n' "$ACCEPTED_SMOKE_RUN_ID"
    printf 'TREELEARN_DEV_RUNTIME_ROOT=%q\n' "$RUNTIME_ROOT"
    printf 'TREELEARN_DEV_PREDICTION_ROOT=%q\n' "$PREDICTION_ROOT"
    printf 'TREELEARN_DEV_METADATA_ROOT=%q\n' "$METADATA_ROOT"
    printf 'TREELEARN_DEV_TABLE_ROOT=%q\n' "$TABLE_ROOT"
    printf 'TREELEARN_DEV_ACCEPTANCE=%q\n' "$ACCEPTANCE"
    printf 'TREELEARN_DEV_MANIFEST=%q\n' "$MANIFEST_CSV"
    printf 'TREELEARN_DEV_MANIFEST_JSON=%q\n' "$MANIFEST_JSON"
    printf 'TREELEARN_DEV_RUN_SUMMARY=%q\n' "$RUN_SUMMARY"
    printf 'TREELEARN_DEV_SITE_SUMMARY=%q\n' "$SITE_SUMMARY"
    printf 'TREELEARN_DEV_DEVELOPMENT_SUMMARY=%q\n' "$DEVELOPMENT_SUMMARY"
  } > "$STATE_FILE"
}

cancel_partial_submission() {
  local status=$?
  if ((${#SUBMITTED_JOBS[@]})); then
    scancel "${SUBMITTED_JOBS[@]}" 2>/dev/null || true
  fi
  SUBMISSION_STATUS="submission_failed_jobs_cancelled"
  write_state
  echo "Submission failed; cancelled jobs created by this attempt." >&2
  echo "state_file=$STATE_FILE" >&2
  exit "$status"
}
trap cancel_partial_submission ERR

PREP_JOB=$(sbatch --parsable \
  --export="ALL,TREELEARN_FULL_DEV_CONFIRMED=1,TREELEARN_DEV_ALIGNMENT_REVIEW_CONFIRMED=1,TREELEARN_DEV_RUN_ID=$RUN_ID,TREELEARN_DATASET_ROOT=$TREELEARN_DATASET_ROOT,TREELEARN_ACCEPTED_SMOKE_RUN_ID=$ACCEPTED_SMOKE_RUN_ID,TREELEARN_EXPECTED_BENCHMARK_COMMIT=$BENCHMARK_COMMIT,TREELEARN_DEV_ACCEPTANCE=$ACCEPTANCE,TREELEARN_DEV_MANIFEST=$MANIFEST_CSV,TREELEARN_DEV_MANIFEST_JSON=$MANIFEST_JSON" \
  methods/treelearn/slurm/prepare_for_instance_development.sbatch)
SUBMITTED_JOBS+=("$PREP_JOB")
SUBMISSION_STATUS="preparation_submitted"
write_state

ARRAY_JOB=$(sbatch --parsable \
  --array=0-20%2 \
  --dependency="afterok:$PREP_JOB" \
  --kill-on-invalid-dep=yes \
  --export="ALL,TREELEARN_FULL_DEV_CONFIRMED=1,TREELEARN_DEV_ALIGNMENT_REVIEW_CONFIRMED=1,TREELEARN_ENV=$TREELEARN_ENV,TREELEARN_REPO=$TREELEARN_REPO,TREELEARN_CHECKPOINT=$TREELEARN_CHECKPOINT,TREELEARN_DATASET_ROOT=$TREELEARN_DATASET_ROOT,TREELEARN_DEV_RUN_ID=$RUN_ID,TREELEARN_DEV_MANIFEST=$MANIFEST_CSV,TREELEARN_EXPECTED_BENCHMARK_COMMIT=$BENCHMARK_COMMIT,TREELEARN_EXPECTED_CHECKPOINT_MD5=$EXPECTED_CHECKPOINT_MD5,TREELEARN_EXPECTED_CHECKPOINT_SHA256=$EXPECTED_CHECKPOINT_SHA256" \
  methods/treelearn/slurm/run_for_instance_development.sbatch)
SUBMITTED_JOBS+=("$ARRAY_JOB")
SUBMISSION_STATUS="development_array_submitted"
write_state

SUMMARY_JOB=$(sbatch --parsable \
  --dependency="afterany:$ARRAY_JOB" \
  --export="ALL,TREELEARN_DEV_RUN_ID=$RUN_ID,TREELEARN_DEV_MANIFEST=$MANIFEST_CSV,TREELEARN_EXPECTED_BENCHMARK_COMMIT=$BENCHMARK_COMMIT" \
  methods/treelearn/slurm/summarise_for_instance_development.sbatch)
SUBMITTED_JOBS+=("$SUMMARY_JOB")
SUBMISSION_STATUS="summary_submitted"
write_state

GATE_JOB=$(sbatch --parsable \
  --dependency="afterok:$SUMMARY_JOB" \
  --kill-on-invalid-dep=yes \
  --export="ALL,TREELEARN_DEV_RUN_ID=$RUN_ID,TREELEARN_DEV_RUN_SUMMARY=$RUN_SUMMARY,TREELEARN_EXPECTED_BENCHMARK_COMMIT=$BENCHMARK_COMMIT" \
  methods/treelearn/slurm/gate_for_instance_development.sbatch)
SUBMITTED_JOBS+=("$GATE_JOB")
SUBMISSION_STATUS="chain_submitted"
write_state
trap - ERR

echo "run_id=$RUN_ID"
echo "preparation_job=$PREP_JOB development_array_job=$ARRAY_JOB"
echo "summary_job=$SUMMARY_JOB gate_job=$GATE_JOB"
echo "state_file=$STATE_FILE"
echo "manifest=$MANIFEST_CSV"
echo "development_summary=$DEVELOPMENT_SUMMARY"
echo "No training, tuning or held-out test job was submitted."
