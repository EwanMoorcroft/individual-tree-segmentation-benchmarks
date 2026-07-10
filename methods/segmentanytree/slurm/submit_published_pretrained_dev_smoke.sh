#!/bin/bash

set -euo pipefail

if [[ "${SEGMENTANYTREE_PRETRAINED_DEV_SMOKE_CONFIRMED:-0}" != "1" ]]; then
  echo "Set SEGMENTANYTREE_PRETRAINED_DEV_SMOKE_CONFIRMED=1 after reviewing this development-only route." >&2
  exit 2
fi

PROJECT_ROOT="${SEGMENTANYTREE_PROJECT_ROOT:-$HOME/scratch/tree-seg-benchmark}"
STAMP="${SEGMENTANYTREE_PRETRAINED_DEV_SMOKE_STAMP:-$(date +%Y%m%d_%H%M%S)}"
RUN_ID="segmentanytree_for-instance_published_pretrained_${STAMP}"
CHECKPOINT_SHA256="0b4d74b4644e37a16f59008ad0f5c62894fc4d2d906f3abd803bbfc5b5dd803a"
CHECKPOINT_BUNDLE="$HOME/fastscratch/segmentanytree_pretrained/released_model_bundle"
PREDICTIONS="$PROJECT_ROOT/data/predictions/segmentanytree/for_instance_variants/$RUN_ID/development_smoke"
RUN_METADATA_ROOT="$PROJECT_ROOT/results/metadata/segmentanytree_for_instance/variant_runs/$RUN_ID/development_smoke"
METRICS_ROOT="$PROJECT_ROOT/results/metadata/segmentanytree_for_instance/variants/$RUN_ID/development_smoke"
TABLE_ROOT="$PROJECT_ROOT/results/tables/segmentanytree_for_instance/variants/$RUN_ID/development_smoke"
RUN_METADATA="$RUN_METADATA_ROOT/CULS/plot_1_annotated_run.json"
METRICS="$METRICS_ROOT/CULS/plot_1_annotated.json"
EVIDENCE="$PROJECT_ROOT/results/metadata/segmentanytree_for_instance/stage1_smokes/$RUN_ID.json"
STATE_FILE="$HOME/fastscratch/segmentanytree_published_pretrained_dev_smoke_${STAMP}.env"
ARCHIVE_ROOT="$HOME/fastscratch/segmentanytree_recovery_archive/$RUN_ID/$(date +%Y%m%d_%H%M%S)"
submitted_jobs=()

cancel_partial_submission() {
  local status=$?
  if ((${#submitted_jobs[@]})); then
    scancel "${submitted_jobs[@]}" 2>/dev/null || true
  fi
  echo "Submission failed; cancelled jobs created by this attempt." >&2
  exit "$status"
}
trap cancel_partial_submission ERR

if [[ ! "$STAMP" =~ ^[0-9]{8}_[0-9]{6}$ ]]; then
  echo "SEGMENTANYTREE_PRETRAINED_DEV_SMOKE_STAMP must use YYYYMMDD_HHMMSS." >&2
  exit 2
fi

cd "$PROJECT_ROOT"
mkdir -p logs/segmentanytree_for_instance
test -f "$HOME/scratch/containers/segment-any-tree_latest.sif"
test -d "$HOME/fastscratch/venvs/treebench"
test "$(git -C external/SegmentAnyTree rev-parse HEAD)" = "a3561ed8447bbb7938f059ba65a3e9c97d6e2ee9"
if [[ -n "$(git status --porcelain)" ]]; then
  echo "Barkla repository is not clean; refusing the development smoke." >&2
  exit 2
fi
module purge
module load miniforge3/25.3.0-python3.12.10
source "$HOME/fastscratch/venvs/treebench/bin/activate"

mapfile -t plot_info < <(
  python methods/segmentanytree/scripts/data/select_for_instance_plot.py \
    --config methods/segmentanytree/configs/for_instance_benchmark.yml \
    --split dev \
    --plot-path CULS/plot_1_annotated.las \
    --format lines
)
if [[ "${plot_info[1]}" != "CULS/plot_1_annotated.las" || "${plot_info[4]}" != "dev" ]]; then
  echo "Configured smoke plot is not the expected development plot." >&2
  exit 2
fi

if [[ -e "$CHECKPOINT_BUNDLE" ]]; then
  checkpoint_ready=0
  if [[ -f "$CHECKPOINT_BUNDLE/PointGroup-PAPER.pt" && -s "$CHECKPOINT_BUNDLE/.hydra/overrides.yaml" ]]; then
    actual_checkpoint_sha256=$(sha256sum "$CHECKPOINT_BUNDLE/PointGroup-PAPER.pt" | awk '{print $1}')
    [[ "$actual_checkpoint_sha256" == "$CHECKPOINT_SHA256" ]] && checkpoint_ready=1
  fi
  if [[ "$checkpoint_ready" -ne 1 ]]; then
    mkdir -p "$ARCHIVE_ROOT"
    mv "$CHECKPOINT_BUNDLE" "$ARCHIVE_ROOT/released_model_bundle"
  fi
fi

stale_targets=()
for target in "$PREDICTIONS" "$RUN_METADATA_ROOT" "$METRICS_ROOT" "$TABLE_ROOT" "$EVIDENCE"; do
  [[ -e "$target" ]] && stale_targets+=("$target")
done
if ((${#stale_targets[@]})); then
  mkdir -p "$ARCHIVE_ROOT"
  for target in "${stale_targets[@]}"; do
    relative=${target#"$PROJECT_ROOT"/}
    archived="$ARCHIVE_ROOT/$relative"
    mkdir -p "$(dirname "$archived")"
    mv "$target" "$archived"
  done
  printf 'run_id=%s\narchived_at=%s\n' "$RUN_ID" "$(date -u +%FT%TZ)" \
    > "$ARCHIVE_ROOT/recovery_manifest.txt"
fi

checkpoint_job=$(sbatch --parsable \
  --export="ALL,SEGMENTANYTREE_RELEASED_CHECKPOINT_DIR=$CHECKPOINT_BUNDLE,SEGMENTANYTREE_RELEASED_CHECKPOINT_SHA256=$CHECKPOINT_SHA256" \
  methods/segmentanytree/slurm/training/prepare_segmentanytree_released_checkpoint.sbatch)
submitted_jobs+=("$checkpoint_job")

inference_job=$(sbatch --parsable \
  --dependency="afterok:$checkpoint_job" \
  --export="ALL,SEGMENTANYTREE_PRETRAINED_DEV_SMOKE_CONFIRMED=1,SEGMENTANYTREE_POINTWISE_OUTPUT_ROOT=$PREDICTIONS,SEGMENTANYTREE_POINTWISE_RUN_METADATA_ROOT=$RUN_METADATA_ROOT,SEGMENTANYTREE_CHECKPOINT_DIR=$CHECKPOINT_BUNDLE" \
  methods/segmentanytree/slurm/inference/run_published_pretrained_dev_smoke.sbatch)
submitted_jobs+=("$inference_job")

evaluation_job=$(sbatch --parsable \
  --dependency="afterok:$inference_job" \
  --export="ALL,SEGMENTANYTREE_POINTWISE_OUTPUT_ROOT=$PREDICTIONS,SEGMENTANYTREE_POINTWISE_EVALUATION_ROOT=$METRICS_ROOT,SEGMENTANYTREE_POINTWISE_TABLE_ROOT=$TABLE_ROOT" \
  methods/segmentanytree/slurm/evaluation/evaluate_published_pretrained_dev_smoke.sbatch)
submitted_jobs+=("$evaluation_job")

gate_job=$(sbatch --parsable \
  --dependency="afterok:$evaluation_job" \
  --export="ALL,SEGMENTANYTREE_DEV_SMOKE_RUN_METADATA=$RUN_METADATA,SEGMENTANYTREE_DEV_SMOKE_METRICS=$METRICS,SEGMENTANYTREE_CHECKPOINT_DIR=$CHECKPOINT_BUNDLE,SEGMENTANYTREE_DEV_SMOKE_EVIDENCE=$EVIDENCE" \
  methods/segmentanytree/slurm/evaluation/validate_published_pretrained_dev_smoke.sbatch)
submitted_jobs+=("$gate_job")

{
  printf 'SAT_PRETRAINED_DEV_SMOKE_RUN_ID=%q\n' "$RUN_ID"
  printf 'SAT_PRETRAINED_DEV_SMOKE_PLOT=%q\n' "CULS/plot_1_annotated.las"
  printf 'SAT_PRETRAINED_DEV_SMOKE_CHECKPOINT_JOB=%q\n' "$checkpoint_job"
  printf 'SAT_PRETRAINED_DEV_SMOKE_INFERENCE_JOB=%q\n' "$inference_job"
  printf 'SAT_PRETRAINED_DEV_SMOKE_EVALUATION_JOB=%q\n' "$evaluation_job"
  printf 'SAT_PRETRAINED_DEV_SMOKE_GATE_JOB=%q\n' "$gate_job"
  printf 'SAT_PRETRAINED_DEV_SMOKE_EVIDENCE=%q\n' "$EVIDENCE"
} > "$STATE_FILE"
trap - ERR

echo "run_id=$RUN_ID"
echo "checkpoint_job=$checkpoint_job inference_job=$inference_job evaluation_job=$evaluation_job gate_job=$gate_job"
echo "state_file=$STATE_FILE"
echo "evidence=$EVIDENCE"
echo "No held-out test or fine-tuning job was submitted."
