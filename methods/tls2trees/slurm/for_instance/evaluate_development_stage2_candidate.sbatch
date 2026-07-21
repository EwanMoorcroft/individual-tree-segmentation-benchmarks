#!/bin/bash -l
#SBATCH --job-name=tls2trees_dt_stage2
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
test "${TLS2TREES_STAGE2_CONFIRMED:-0}" = "1"
test "${TLS2TREES_REQUESTED_VARIANT:-}" = "development_tuned"
test "${TLS2TREES_REQUESTED_SPLIT:-}" = "development"
test "$(git rev-parse HEAD)" = "$TLS2TREES_EXPECTED_BENCHMARK_COMMIT"
test -z "$(git status --porcelain)"
test "$(sha256sum "$TLS2TREES_MANIFEST_JSON" | awk '{print $1}')" = \
  "$TLS2TREES_MANIFEST_SHA256"
test "$(sha256sum "$TLS2TREES_STAGE2_SELECTION_JSON" | awk '{print $1}')" = \
  "$TLS2TREES_STAGE2_SELECTION_SHA256"
test "$(sha256sum "$TLS2TREES_STAGE1_CONFIG" | awk '{print $1}')" = \
  "$TLS2TREES_STAGE1_CONFIG_SHA256"
test "$(sha256sum "$TLS2TREES_PROBE_SUMMARY_JSON" | awk '{print $1}')" = \
  "$TLS2TREES_PROBE_SUMMARY_SHA256"

SELECTION_INDEX=$((SLURM_ARRAY_TASK_ID / 21))
TASK_INDEX=$((SLURM_ARRAY_TASK_ID % 21))
readarray -t CANDIDATE_FIELDS < <(
  "$TLS2TREES_TREEBENCH_ENV/bin/python" -c '
import json,sys
p=json.load(open(sys.argv[1]))
c=p["selected_candidates"][int(sys.argv[2])]
print(c["stage1_candidate_index"])
print(c["candidate_id"])
' "$TLS2TREES_STAGE2_SELECTION_JSON" "$SELECTION_INDEX"
)
CANDIDATE_INDEX="${CANDIDATE_FIELDS[0]}"
CANDIDATE_ID="${CANDIDATE_FIELDS[1]}"
SAFE_PLOT_ID=$("$TLS2TREES_TREEBENCH_ENV/bin/python" "$TLS2TREES_MANIFEST_CLI" resolve \
  --manifest-json "$TLS2TREES_MANIFEST_JSON" --expected-split development \
  --task-index "$TASK_INDEX" --field safe_plot_id)
RELATIVE_PATH=$("$TLS2TREES_TREEBENCH_ENV/bin/python" "$TLS2TREES_MANIFEST_CLI" resolve \
  --manifest-json "$TLS2TREES_MANIFEST_JSON" --expected-split development \
  --task-index "$TASK_INDEX" --field relative_path)
INPUT_LAS=$("$TLS2TREES_TREEBENCH_ENV/bin/python" "$TLS2TREES_MANIFEST_CLI" resolve \
  --manifest-json "$TLS2TREES_MANIFEST_JSON" --expected-split development \
  --task-index "$TASK_INDEX" --field input_las)
CANDIDATE_RUN_ID="${TLS2TREES_STAGE2_RUN_ID}__${CANDIDATE_ID}"
SOURCE_PLOT_ROOT="$TLS2TREES_OUTPUT_ROOT/tls2trees/for_instance/development_tuned/development/$TLS2TREES_STAGE2_SEMANTIC_CACHE_RUN_ID/$SAFE_PLOT_ID"
PLOT_ROOT="$TLS2TREES_OUTPUT_ROOT/tls2trees/for_instance/development_tuned/development/$CANDIDATE_RUN_ID/$SAFE_PLOT_ID"
test -f "$SOURCE_PLOT_ROOT/metadata/semantic_run.json"
RECOVERY_ARGUMENTS=()
RECOVERY_MODE="none"
if [[ "${TLS2TREES_STAGE2_RECOVERY:-0}" == "1" ]]; then
  test -f "$PLOT_ROOT/metadata/instance_run.json"
  test ! -e "$PLOT_ROOT/predictions/aligned"
  test ! -e "$PLOT_ROOT/evaluation"
  INSTANCE_STATUS=$("$TLS2TREES_TREEBENCH_ENV/bin/python" -c \
    'import json,sys; print(json.load(open(sys.argv[1]))["status"])' \
    "$PLOT_ROOT/metadata/instance_run.json")
  test "$INSTANCE_STATUS" = "failed"
  RECOVERY_MODE="audited_empty_prediction_instance"
  RECOVERY_ARGUMENTS+=(--resume-failed-audited-instance)
else
  test ! -e "$PLOT_ROOT"
fi

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
  --workflow-run-id "$TLS2TREES_STAGE2_RUN_ID" \
  --candidate-run-id "$CANDIDATE_RUN_ID" \
  --candidate-index "$CANDIDATE_INDEX" \
  --stage1-config "$TLS2TREES_STAGE1_CONFIG" \
  --probe-summary-json "$TLS2TREES_PROBE_SUMMARY_JSON" \
  --probe-summary-sha256 "$TLS2TREES_PROBE_SUMMARY_SHA256" \
  --tls2trees-repo "$TLS2TREES_UPSTREAM_REPO" \
  "${RECOVERY_ARGUMENTS[@]}"

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

for TARGET in leaf_off leaf_on; do
  ALIGNED_ROOT="$PLOT_ROOT/predictions/aligned/$TARGET"
  EVALUATION_ROOT="$PLOT_ROOT/evaluation/$TARGET"
  test ! -e "$EVALUATION_ROOT"
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
done

echo "status=stage2_candidate_plot_evaluated"
echo "selection_index=$SELECTION_INDEX"
echo "candidate_id=$CANDIDATE_ID"
echo "task_index=$TASK_INDEX"
echo "recovery_mode=$RECOVERY_MODE"
echo "development_accuracy_metrics_computed=true"
echo "held_out_test_accessed=false"
