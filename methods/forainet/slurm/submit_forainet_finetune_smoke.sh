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
: "${FORAINET_FINETUNE_ROOT:?set FORAINET_FINETUNE_ROOT}"
: "${FORAINET_FINETUNE_RUN_ID:?set FORAINET_FINETUNE_RUN_ID}"
: "${FORAINET_FINETUNE_STATE_FILE:?set FORAINET_FINETUNE_STATE_FILE}"

if [[ "${FORAINET_FINETUNE_CONFIRMED:-0}" != "1" ]]; then
  echo "Refusing fine-tune submission without confirmation."
  exit 2
fi
if [[ ! "$FORAINET_FINETUNE_RUN_ID" =~ ^forainet__for-instance__fine-tuned-dev__checkpoint-initialised__finetune-smoke__[0-9]{8}T[0-9]{6}$ ]]; then
  echo "FORAINET_FINETUNE_RUN_ID does not match the frozen pattern."
  exit 2
fi
test "$(git -C "$FORAINET_BENCHMARK_ROOT" rev-parse HEAD)" = \
  "$FORAINET_EXPECTED_BENCHMARK_COMMIT"
test -z "$(git -C "$FORAINET_BENCHMARK_ROOT" status --porcelain)"
test -f "$FORAINET_DEVELOPMENT_ROOT/final_gate.json"
test ! -e "$FORAINET_FINETUNE_ROOT"
test ! -e "$FORAINET_FINETUNE_STATE_FILE"

mkdir -p "$FORAINET_BENCHMARK_ROOT/logs" \
  "$(dirname "$FORAINET_FINETUNE_ROOT")" \
  "$(dirname "$FORAINET_FINETUNE_STATE_FILE")"
export_names="ALL,FORAINET_BENCHMARK_ROOT,FORAINET_EXPECTED_BENCHMARK_COMMIT,FORAINET_UPSTREAM_ROOT,FORAINET_CHECKPOINT,FORAINET_IMAGE,FORAINET_QUALIFICATION_ROOT,FORAINET_DATASET_ROOT,FORAINET_DEVELOPMENT_ROOT,FORAINET_FINETUNE_ROOT,FORAINET_FINETUNE_RUN_ID,FORAINET_FINETUNE_CONFIRMED"
prepare_job="$(
  cd "$FORAINET_BENCHMARK_ROOT"
  sbatch --parsable --export="$export_names" \
    methods/forainet/slurm/prepare_forainet_finetune.sbatch
)"
smoke_job="$(
  cd "$FORAINET_BENCHMARK_ROOT"
  sbatch --parsable \
    --dependency="afterok:$prepare_job" \
    --export="$export_names" \
    methods/forainet/slurm/run_forainet_finetune_smoke.sbatch
)"
{
  printf 'FORAINET_FINETUNE_PREP_JOB_ID=%q\n' "$prepare_job"
  printf 'FORAINET_FINETUNE_SMOKE_JOB_ID=%q\n' "$smoke_job"
  printf 'FORAINET_FINETUNE_RUN_ID=%q\n' "$FORAINET_FINETUNE_RUN_ID"
  printf 'FORAINET_FINETUNE_ROOT=%q\n' "$FORAINET_FINETUNE_ROOT"
  printf 'FORAINET_EXPECTED_BENCHMARK_COMMIT=%q\n' \
    "$FORAINET_EXPECTED_BENCHMARK_COMMIT"
  printf 'FORAINET_SUBMITTED_AT_EPOCH=%q\n' "$(date +%s)"
} > "$FORAINET_FINETUNE_STATE_FILE"
printf 'preparation_job=%s smoke_job=%s\n' "$prepare_job" "$smoke_job"
