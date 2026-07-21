#!/bin/bash -l
#SBATCH --job-name=tls2trees_dt_leaf_screen
#SBATCH --partition=nodes
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --no-requeue
#SBATCH --output=logs/tls2trees_for_instance/%x_%A_%a.out
#SBATCH --error=logs/tls2trees_for_instance/%x_%A_%a.err

set -euo pipefail

cd "${TLS2TREES_PROJECT_ROOT:-$HOME/scratch/tree-seg-benchmark}"
test "${TLS2TREES_LEAF_SCREEN_CONFIRMED:-0}" = "1"
test "${TLS2TREES_REQUESTED_VARIANT:-}" = "development_tuned"
test "${TLS2TREES_REQUESTED_SPLIT:-}" = "development"
test "$(git rev-parse HEAD)" = "$TLS2TREES_EXPECTED_BENCHMARK_COMMIT"
test -z "$(git status --porcelain)"
test "$(sha256sum "$TLS2TREES_MANIFEST_JSON" | awk '{print $1}')" = \
  "$TLS2TREES_MANIFEST_SHA256"
test "$(sha256sum "$TLS2TREES_LEAF_SCREEN_CONFIG" | awk '{print $1}')" = \
  "$TLS2TREES_LEAF_SCREEN_CONFIG_SHA256"
test "$(sha256sum "$TLS2TREES_DEVELOPMENT_EVIDENCE_JSON" | awk '{print $1}')" = \
  "$TLS2TREES_DEVELOPMENT_EVIDENCE_SHA256"

CANDIDATE_INDEX=$((SLURM_ARRAY_TASK_ID / 5))
STAGE0_INDEX=$((SLURM_ARRAY_TASK_ID % 5))
TASK_INDEX=$("$TLS2TREES_TREEBENCH_ENV/bin/python" "$TLS2TREES_MANIFEST_CLI" resolve-stage0 \
  --manifest-json "$TLS2TREES_MANIFEST_JSON" --stage0-index "$STAGE0_INDEX" --field task_index)
SAFE_PLOT_ID=$("$TLS2TREES_TREEBENCH_ENV/bin/python" "$TLS2TREES_MANIFEST_CLI" resolve-stage0 \
  --manifest-json "$TLS2TREES_MANIFEST_JSON" --stage0-index "$STAGE0_INDEX" --field safe_plot_id)
RELATIVE_PATH=$("$TLS2TREES_TREEBENCH_ENV/bin/python" "$TLS2TREES_MANIFEST_CLI" resolve-stage0 \
  --manifest-json "$TLS2TREES_MANIFEST_JSON" --stage0-index "$STAGE0_INDEX" --field relative_path)
INPUT_LAS=$("$TLS2TREES_TREEBENCH_ENV/bin/python" "$TLS2TREES_MANIFEST_CLI" resolve-stage0 \
  --manifest-json "$TLS2TREES_MANIFEST_JSON" --stage0-index "$STAGE0_INDEX" --field input_las)
CANDIDATE_ID=$("$TLS2TREES_TREEBENCH_ENV/bin/python" -c \
  'import sys,yaml; print(yaml.safe_load(open(sys.argv[1]))["candidates"][int(sys.argv[2])]["candidate_id"])' \
  "$TLS2TREES_LEAF_SCREEN_CONFIG" "$CANDIDATE_INDEX")
CANDIDATE_RUN_ID="${TLS2TREES_LEAF_SCREEN_RUN_ID}__${CANDIDATE_ID}"
SOURCE_PLOT_ROOT="$TLS2TREES_OUTPUT_ROOT/tls2trees/for_instance/development_tuned/development/$TLS2TREES_SOURCE_SEMANTIC_CACHE_RUN_ID/$SAFE_PLOT_ID"
PLOT_ROOT="$TLS2TREES_OUTPUT_ROOT/tls2trees/for_instance/development_tuned/development/$CANDIDATE_RUN_ID/$SAFE_PLOT_ID"
test -f "$SOURCE_PLOT_ROOT/metadata/semantic_run.json"
test ! -e "$PLOT_ROOT"

module purge
module load miniforge3/25.3.0-python3.12.10
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$TLS2TREES_METHOD_ENV"
export PYTHONNOUSERSITE=1
export LD_LIBRARY_PATH="$TLS2TREES_METHOD_ENV/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
python "$TLS2TREES_ENV_VALIDATOR" \
  --tls2trees-repo "$TLS2TREES_UPSTREAM_REPO" \
  --setup-marker-json "$TLS2TREES_METHOD_ENV_MARKER" \
  --skip-model-load
python "$TLS2TREES_CANDIDATE_CLI" \
  --manifest-json "$TLS2TREES_MANIFEST_JSON" \
  --task-index "$TASK_INDEX" \
  --source-plot-root "$SOURCE_PLOT_ROOT" \
  --output-root "$TLS2TREES_OUTPUT_ROOT" \
  --workflow-run-id "$TLS2TREES_LEAF_SCREEN_RUN_ID" \
  --candidate-run-id "$CANDIDATE_RUN_ID" \
  --candidate-index "$CANDIDATE_INDEX" \
  --candidate-config "$TLS2TREES_LEAF_SCREEN_CONFIG" \
  --development-evidence-json "$TLS2TREES_DEVELOPMENT_EVIDENCE_JSON" \
  --development-evidence-sha256 "$TLS2TREES_DEVELOPMENT_EVIDENCE_SHA256" \
  --tls2trees-repo "$TLS2TREES_UPSTREAM_REPO" \
  --target leaf_on

conda deactivate
# shellcheck disable=SC1090
source "$TLS2TREES_TREEBENCH_ENV/bin/activate"
export PYTHONNOUSERSITE=1
python "$TLS2TREES_ADAPTER_CLI" \
  --manifest-json "$TLS2TREES_MANIFEST_JSON" \
  --task-index "$TASK_INDEX" \
  --output-root "$TLS2TREES_OUTPUT_ROOT" \
  --run-id "$CANDIDATE_RUN_ID" \
  --variant development_tuned \
  --split development \
  --coordinate-tolerance-m 0.001

ALIGNED_ROOT="$PLOT_ROOT/predictions/aligned/leaf_on"
EVALUATION_ROOT="$PLOT_ROOT/evaluation/leaf_on"
test ! -e "$EVALUATION_ROOT"
mkdir -p "$EVALUATION_ROOT"
python "$TLS2TREES_EVALUATE_CLI" \
  --target leaf_on \
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

echo "status=development_leaf_screen_candidate_plot_evaluated"
echo "candidate_id=$CANDIDATE_ID"
echo "stage0_index=$STAGE0_INDEX"
echo "target=leaf_on"
echo "semantic_cache_reused=true"
echo "development_accuracy_metrics_computed=true"
echo "held_out_test_accessed=false"
