#!/bin/bash -l
#SBATCH --job-name=tls2trees_dt_test
#SBATCH --partition=nodes
#SBATCH --time=08:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --no-requeue
#SBATCH --output=logs/tls2trees_for_instance/%x_%A_%a.out
#SBATCH --error=logs/tls2trees_for_instance/%x_%A_%a.err

set -euo pipefail
cd "${TLS2TREES_PROJECT_ROOT:-$HOME/scratch/tree-seg-benchmark}"
test "${TLS2TREES_HELD_OUT_TEST_CONFIRMED:-0}" = "1"
test "${TLS2TREES_REQUESTED_SPLIT:-}" = "test"
test "$(git rev-parse HEAD)" = "$TLS2TREES_EXPECTED_BENCHMARK_COMMIT"
test -z "$(git status --porcelain)"
test "$(sha256sum "$TLS2TREES_MANIFEST_JSON" | awk '{print $1}')" = \
  "$(awk '{print $1}' "$TLS2TREES_MANIFEST_SHA256_FILE")"
test "$(sha256sum "$TLS2TREES_FINAL_SELECTION_JSON" | awk '{print $1}')" = "$TLS2TREES_FINAL_SELECTION_SHA256"
test "$(sha256sum "$TLS2TREES_STAGE1_CONFIG" | awk '{print $1}')" = "$TLS2TREES_STAGE1_CONFIG_SHA256"
test "$(sha256sum "$TLS2TREES_PROBE_SUMMARY_JSON" | awk '{print $1}')" = "$TLS2TREES_PROBE_SUMMARY_SHA256"

TARGET_INDEX=$((SLURM_ARRAY_TASK_ID / 11))
TASK_INDEX=$((SLURM_ARRAY_TASK_ID % 11))
readarray -t FIELDS < <("$TLS2TREES_TREEBENCH_ENV/bin/python" -c '
import json,sys
p=json.load(open(sys.argv[1])); target=("leaf_off","leaf_on")[int(sys.argv[2])]
s=p["selected_by_target"][target]
print(target); print(s["stage1_candidate_index"]); print(s["candidate_id"])
' "$TLS2TREES_FINAL_SELECTION_JSON" "$TARGET_INDEX")
TARGET="${FIELDS[0]}"
CANDIDATE_INDEX="${FIELDS[1]}"
CANDIDATE_ID="${FIELDS[2]}"
SAFE_PLOT_ID=$("$TLS2TREES_TREEBENCH_ENV/bin/python" "$TLS2TREES_MANIFEST_CLI" resolve \
  --manifest-json "$TLS2TREES_MANIFEST_JSON" --expected-split test --allow-held-out-test \
  --task-index "$TASK_INDEX" --field safe_plot_id)
RELATIVE_PATH=$("$TLS2TREES_TREEBENCH_ENV/bin/python" "$TLS2TREES_MANIFEST_CLI" resolve \
  --manifest-json "$TLS2TREES_MANIFEST_JSON" --expected-split test --allow-held-out-test \
  --task-index "$TASK_INDEX" --field relative_path)
INPUT_LAS=$("$TLS2TREES_TREEBENCH_ENV/bin/python" "$TLS2TREES_MANIFEST_CLI" resolve \
  --manifest-json "$TLS2TREES_MANIFEST_JSON" --expected-split test --allow-held-out-test \
  --task-index "$TASK_INDEX" --field input_las)
CANDIDATE_RUN_ID="${TLS2TREES_TEST_RUN_ID}__${TARGET}__${CANDIDATE_ID}"
SOURCE_PLOT_ROOT="$TLS2TREES_OUTPUT_ROOT/tls2trees/for_instance/development_tuned/test/$TLS2TREES_TEST_SEMANTIC_CACHE_RUN_ID/$SAFE_PLOT_ID"
PLOT_ROOT="$TLS2TREES_OUTPUT_ROOT/tls2trees/for_instance/development_tuned/test/$CANDIDATE_RUN_ID/$SAFE_PLOT_ID"
test -f "$SOURCE_PLOT_ROOT/metadata/semantic_run.json"
test ! -e "$PLOT_ROOT"

module purge
module load miniforge3/25.3.0-python3.12.10
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$TLS2TREES_METHOD_ENV"
export PYTHONNOUSERSITE=1
export LD_LIBRARY_PATH="$TLS2TREES_METHOD_ENV/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
python "$TLS2TREES_ENV_VALIDATOR" --tls2trees-repo "$TLS2TREES_UPSTREAM_REPO" \
  --setup-marker-json "$TLS2TREES_METHOD_ENV_MARKER" --skip-model-load
python "$TLS2TREES_CANDIDATE_CLI" \
  --manifest-json "$TLS2TREES_MANIFEST_JSON" --task-index "$TASK_INDEX" \
  --source-plot-root "$SOURCE_PLOT_ROOT" --output-root "$TLS2TREES_OUTPUT_ROOT" \
  --workflow-run-id "$TLS2TREES_TEST_RUN_ID" --candidate-run-id "$CANDIDATE_RUN_ID" \
  --candidate-index "$CANDIDATE_INDEX" --stage1-config "$TLS2TREES_STAGE1_CONFIG" \
  --probe-summary-json "$TLS2TREES_PROBE_SUMMARY_JSON" \
  --probe-summary-sha256 "$TLS2TREES_PROBE_SUMMARY_SHA256" \
  --tls2trees-repo "$TLS2TREES_UPSTREAM_REPO" --split test --target "$TARGET" \
  --final-selection-json "$TLS2TREES_FINAL_SELECTION_JSON" \
  --final-selection-sha256 "$TLS2TREES_FINAL_SELECTION_SHA256"

conda deactivate
# shellcheck disable=SC1090
source "$TLS2TREES_TREEBENCH_ENV/bin/activate"
export PYTHONNOUSERSITE=1
python "$TLS2TREES_ADAPTER_CLI" \
  --manifest-json "$TLS2TREES_MANIFEST_JSON" --task-index "$TASK_INDEX" \
  --output-root "$TLS2TREES_OUTPUT_ROOT" --run-id "$CANDIDATE_RUN_ID" \
  --variant development_tuned --split test --coordinate-tolerance-m 0.001 \
  --allow-held-out-test

ALIGNED_ROOT="$PLOT_ROOT/predictions/aligned/$TARGET"
EVALUATION_ROOT="$PLOT_ROOT/evaluation/$TARGET"
test ! -e "$EVALUATION_ROOT"
mkdir -p "$EVALUATION_ROOT"
python "$TLS2TREES_EVALUATE_CLI" \
  --target "$TARGET" --aligned-predictions-npz "$ALIGNED_ROOT/source_row_predictions.npz" \
  --reference-labelled-point-cloud "$INPUT_LAS" \
  --alignment-metadata-json "$ALIGNED_ROOT/alignment_metadata.json" \
  --plot-id "$SAFE_PLOT_ID" --relative-path "$RELATIVE_PATH" --split test \
  --iou-threshold 0.5 --output-json "$EVALUATION_ROOT/plot_metrics.json" \
  --output-matches-csv "$EVALUATION_ROOT/matches.csv" \
  --output-unmatched-predictions-csv "$EVALUATION_ROOT/unmatched_predictions.csv" \
  --output-unmatched-references-csv "$EVALUATION_ROOT/unmatched_references.csv"

echo "status=held_out_test_plot_evaluated"
echo "target=$TARGET"
echo "candidate_id=$CANDIDATE_ID"
echo "task_index=$TASK_INDEX"
echo "held_out_test_accessed=true"
