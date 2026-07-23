#!/usr/bin/env bash

set -euo pipefail

: "${FORAINET_BENCHMARK_ROOT:?set FORAINET_BENCHMARK_ROOT}"
: "${FORAINET_EXPECTED_BENCHMARK_COMMIT:?set FORAINET_EXPECTED_BENCHMARK_COMMIT}"
: "${FORAINET_UPSTREAM_ROOT:?set FORAINET_UPSTREAM_ROOT}"
: "${FORAINET_CHECKPOINT:?set FORAINET_CHECKPOINT}"
: "${FORAINET_IMAGE:?set FORAINET_IMAGE}"
: "${FORAINET_QUALIFICATION_ROOT:?set FORAINET_QUALIFICATION_ROOT}"
: "${FORAINET_DATASET_ROOT:?set FORAINET_DATASET_ROOT}"
: "${FORAINET_DEVELOPMENT_ROOT:?set FORAINET_DEVELOPMENT_ROOT}"
: "${FORAINET_DEVELOPMENT_RUN_ID:?set FORAINET_DEVELOPMENT_RUN_ID}"
: "${FORAINET_DEVELOPMENT_STATE_FILE:?set FORAINET_DEVELOPMENT_STATE_FILE}"

if [[ "${FORAINET_DEVELOPMENT_CONFIRMED:-0}" != "1" ]]; then
  echo "Refusing development submission without confirmation."
  exit 2
fi
if [[ ! "$FORAINET_DEVELOPMENT_RUN_ID" =~ ^forainet__for-instance__published-pretrained__none__development__[0-9]{8}T[0-9]{6}$ ]]; then
  echo "FORAINET_DEVELOPMENT_RUN_ID does not match the frozen pattern."
  exit 2
fi

test "$(git -C "$FORAINET_BENCHMARK_ROOT" rev-parse HEAD)" = \
  "$FORAINET_EXPECTED_BENCHMARK_COMMIT"
test -z "$(git -C "$FORAINET_BENCHMARK_ROOT" status --porcelain)"
test -f "$FORAINET_CHECKPOINT"
test -f "$FORAINET_IMAGE"
test -d "$FORAINET_QUALIFICATION_ROOT"
test -r "$FORAINET_DATASET_ROOT/data_split_metadata.csv"
test ! -e "$FORAINET_DEVELOPMENT_ROOT"
test ! -e "$FORAINET_DEVELOPMENT_STATE_FILE"

mkdir -p "$FORAINET_BENCHMARK_ROOT/logs" \
  "$(dirname "$FORAINET_DEVELOPMENT_STATE_FILE")"
mkdir "$FORAINET_DEVELOPMENT_ROOT"
export_names="ALL,FORAINET_BENCHMARK_ROOT,FORAINET_EXPECTED_BENCHMARK_COMMIT,FORAINET_UPSTREAM_ROOT,FORAINET_CHECKPOINT,FORAINET_IMAGE,FORAINET_QUALIFICATION_ROOT,FORAINET_DATASET_ROOT,FORAINET_DEVELOPMENT_ROOT,FORAINET_DEVELOPMENT_RUN_ID,FORAINET_DEVELOPMENT_CONFIRMED"
prepare_job="$(
  cd "$FORAINET_BENCHMARK_ROOT"
  sbatch --parsable --export="$export_names" \
    methods/forainet/slurm/prepare_forainet_development.sbatch
)"
array_job="$(
  cd "$FORAINET_BENCHMARK_ROOT"
  sbatch --parsable \
    --dependency="afterok:$prepare_job" \
    --array="0-20%2" \
    --export="$export_names" \
    methods/forainet/slurm/run_forainet_development.sbatch
)"
summary_job="$(
  cd "$FORAINET_BENCHMARK_ROOT"
  sbatch --parsable \
    --dependency="afterany:$array_job" \
    --export="$export_names" \
    methods/forainet/slurm/summarise_forainet_development.sbatch
)"
{
  printf 'FORAINET_DEVELOPMENT_PREP_JOB_ID=%q\n' "$prepare_job"
  printf 'FORAINET_DEVELOPMENT_ARRAY_JOB_ID=%q\n' "$array_job"
  printf 'FORAINET_DEVELOPMENT_SUMMARY_JOB_ID=%q\n' "$summary_job"
  printf 'FORAINET_DEVELOPMENT_RUN_ID=%q\n' "$FORAINET_DEVELOPMENT_RUN_ID"
  printf 'FORAINET_DEVELOPMENT_ROOT=%q\n' "$FORAINET_DEVELOPMENT_ROOT"
  printf 'FORAINET_EXPECTED_BENCHMARK_COMMIT=%q\n' \
    "$FORAINET_EXPECTED_BENCHMARK_COMMIT"
  printf 'FORAINET_SUBMITTED_AT_EPOCH=%q\n' "$(date +%s)"
} > "$FORAINET_DEVELOPMENT_STATE_FILE"
printf 'preparation_job=%s array_job=%s summary_job=%s\n' \
  "$prepare_job" "$array_job" "$summary_job"
