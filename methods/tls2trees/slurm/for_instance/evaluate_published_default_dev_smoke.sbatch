#!/bin/bash -l
#SBATCH --job-name=tls2trees_smoke_evaluate
#SBATCH --partition=nodes
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --output=logs/tls2trees_for_instance/%x_%j.out
#SBATCH --error=logs/tls2trees_for_instance/%x_%j.err

set -euo pipefail

cd "${TLS2TREES_PROJECT_ROOT:-$HOME/scratch/tree-seg-benchmark}"
# shellcheck disable=SC1091
source methods/tls2trees/slurm/for_instance/published_default_dev_smoke_common.sh
tls2trees_load_contract
tls2trees_validate_contract
tls2trees_load_treebench_env
tls2trees_resolve_stage0_context
tls2trees_stage_banner evaluation

TARGET="${TLS2TREES_TARGET:?TLS2TREES_TARGET is required}"
if [[ "$TARGET" != "leaf_off" && "$TARGET" != "leaf_on" ]]; then
  tls2trees_die "target must be leaf_off or leaf_on"
fi
tls2trees_require_file "$TLS2TREES_EVALUATE_CLI" "target-explicit evaluator"
tls2trees_require_file "$PLOT_ROOT/metadata/adapter_run.json" "prediction-adapter metadata"
ALIGNED_ROOT="$PLOT_ROOT/predictions/aligned/$TARGET"
tls2trees_require_file "$ALIGNED_ROOT/source_row_predictions.npz" "source-row predictions"
tls2trees_require_file "$ALIGNED_ROOT/alignment_metadata.json" "target alignment metadata"

EVALUATION_ROOT="$PLOT_ROOT/evaluation/$TARGET"
if [[ -e "$EVALUATION_ROOT" ]]; then
  tls2trees_die "refusing existing target evaluation: $EVALUATION_ROOT"
fi
mkdir -p "$EVALUATION_ROOT"

python "$TLS2TREES_EVALUATE_CLI" \
  --target "$TARGET" \
  --aligned-predictions-npz "$ALIGNED_ROOT/source_row_predictions.npz" \
  --reference-labelled-point-cloud "$INPUT_LAS" \
  --alignment-metadata-json "$ALIGNED_ROOT/alignment_metadata.json" \
  --plot-id "$SAFE_PLOT_ID" \
  --relative-path "$RELATIVE_PATH" \
  --split dev \
  --iou-threshold 0.5 \
  --output-json "$EVALUATION_ROOT/plot_metrics.json" \
  --output-matches-csv "$EVALUATION_ROOT/matches.csv" \
  --output-unmatched-predictions-csv "$EVALUATION_ROOT/unmatched_predictions.csv" \
  --output-unmatched-references-csv "$EVALUATION_ROOT/unmatched_references.csv"

test -f "$EVALUATION_ROOT/plot_metrics.json"
test -f "$EVALUATION_ROOT/matches.csv"
test -f "$EVALUATION_ROOT/unmatched_predictions.csv"
test -f "$EVALUATION_ROOT/unmatched_references.csv"
echo "target=$TARGET"
echo "evaluation_root=$EVALUATION_ROOT"
echo "success_criteria=complete source-row correspondence and target-explicit one-to-one IoU evidence"
echo "failure_indicators=frame or LAS-scale mismatch, coordinate contention, wrong target glob, unsafe_for_scoring"
echo "held_out_test_accessed=false"
echo "finished_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
