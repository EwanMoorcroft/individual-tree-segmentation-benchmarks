#!/bin/bash

set -euo pipefail

PROJECT_ROOT="$HOME/scratch/tree-seg-benchmark"
CONFIG="$PROJECT_ROOT/methods/segmentanytree/configs/for_instance_benchmark.yml"
IMAGE="$HOME/scratch/containers/segment-any-tree_latest.sif"
USERBASE="$HOME/fastscratch/segmentanytree_pyuser_v1"
EXTERNAL_REPO="$PROJECT_ROOT/external/SegmentAnyTree"
EXPECTED_EXTERNAL_COMMIT="a3561ed8447bbb7938f059ba65a3e9c97d6e2ee9"
RUNTIME_ROOT="$HOME/fastscratch/segmentanytree_pointwise_runtime"
PATCH_ROOT="$PROJECT_ROOT/methods/segmentanytree/scripts/runtime/patches"
REQUIRED_SPLIT="${SEGMENTANYTREE_REQUIRED_SPLIT:-test}"
OUTPUT_BASE="${SEGMENTANYTREE_POINTWISE_OUTPUT_ROOT:-$PROJECT_ROOT/data/predictions/segmentanytree/for_instance_pointwise}"
METADATA_BASE="${SEGMENTANYTREE_POINTWISE_RUN_METADATA_ROOT:-$PROJECT_ROOT/results/metadata/segmentanytree_for_instance/pointwise_runs}"
CUSTOM_CHECKPOINT_DIR="${SEGMENTANYTREE_CHECKPOINT_DIR:-}"
RUN_TYPE="${SEGMENTANYTREE_RUN_TYPE:-published_pretrained_inference}"
TRAINING_RUN_ID="${SEGMENTANYTREE_TRAINING_RUN_ID:-}"

cd "$PROJECT_ROOT"
mkdir -p logs/segmentanytree_for_instance

test -f "$IMAGE"
test -d "$USERBASE/lib/python3.8/site-packages"
test -d "$EXTERNAL_REPO/.git"

ACTUAL_EXTERNAL_COMMIT=$(git -C "$EXTERNAL_REPO" rev-parse HEAD)
if [[ "$ACTUAL_EXTERNAL_COMMIT" != "$EXPECTED_EXTERNAL_COMMIT" ]]; then
  echo "Unexpected SegmentAnyTree commit: $ACTUAL_EXTERNAL_COMMIT" >&2
  echo "Expected: $EXPECTED_EXTERNAL_COMMIT" >&2
  exit 2
fi

mapfile -t PLOT_INFO < <(
  python methods/segmentanytree/scripts/data/select_for_instance_plot.py \
    --config "$CONFIG" \
    "$@" \
    --format lines
)
REFERENCE_PATH="${PLOT_INFO[0]}"
RELATIVE_PATH="${PLOT_INFO[1]}"
COLLECTION="${PLOT_INFO[2]}"
PLOT_NAME="${PLOT_INFO[3]}"
SPLIT="${PLOT_INFO[4]}"

if [[ "$SPLIT" != "$REQUIRED_SPLIT" ]]; then
  echo "Inference requires split $REQUIRED_SPLIT, not $SPLIT: $RELATIVE_PATH" >&2
  exit 2
fi

STAGED_INPUT_DIR="$PROJECT_ROOT/data/interim/segmentanytree/for_instance/pointwise_staged_inputs/${COLLECTION}/${PLOT_NAME}"
OUTPUT_DIR="$OUTPUT_BASE/${COLLECTION}/${PLOT_NAME}"
INSTANCE_EVALUATION="$OUTPUT_DIR/Instance_results_forEval_0.ply"
RUN_METADATA="$METADATA_BASE/${COLLECTION}/${PLOT_NAME}_run.json"
TASK_ID="${SLURM_ARRAY_TASK_ID:-pilot}"
RUNTIME_DIR="$RUNTIME_ROOT/${SLURM_JOB_ID:-manual}_${TASK_ID}_${COLLECTION}_${PLOT_NAME}"
PROCESSED_DIR="$RUNTIME_DIR/processed_data"
PATCHED_TRACKER="$RUNTIME_DIR/panoptic_tracker_pointgroup_treeins.py"
PATCH_METADATA="$RUNTIME_DIR/pointgroup_tracker_patch.json"
PACKAGE_VERSIONS="$RUNTIME_DIR/package_versions.json"
TIME_LOG="$RUNTIME_DIR/time.txt"

if [[ -f "$INSTANCE_EVALUATION" ]]; then
  echo "Aligned evaluation output already exists: $INSTANCE_EVALUATION"
  exit 0
fi
if [[ -d "$OUTPUT_DIR" ]] && find "$OUTPUT_DIR" -mindepth 1 -print -quit | grep -q .; then
  echo "Point-wise output directory is not empty: $OUTPUT_DIR" >&2
  echo "Move the incomplete directory before retrying." >&2
  exit 2
fi

mkdir -p "$STAGED_INPUT_DIR" "$OUTPUT_DIR" "$PROCESSED_DIR"
cp "$REFERENCE_PATH" "$STAGED_INPUT_DIR/${PLOT_NAME}.las"

python "$PATCH_ROOT/prepare_pointgroup_tracker_patch.py" \
  --source "$EXTERNAL_REPO/torch_points3d/metrics/panoptic_tracker_pointgroup_treeins.py" \
  --output "$PATCHED_TRACKER" \
  --metadata-output "$PATCH_METADATA"

CHECKPOINT_BIND=()
if [[ -n "$CUSTOM_CHECKPOINT_DIR" ]]; then
  CUSTOM_CHECKPOINT_DIR=$(realpath "$CUSTOM_CHECKPOINT_DIR")
  test -f "$CUSTOM_CHECKPOINT_DIR/PointGroup-PAPER.pt"
  CUSTOM_DATA_ROOT="$RUNTIME_DIR/checkpoint_data"
  mkdir -p "$CUSTOM_DATA_ROOT/treeinsfused"
  CHECKPOINT_BIND=(
    --bind "$CUSTOM_CHECKPOINT_DIR:/home/nibio/mutable-outside-world/model_file:ro"
    --bind "$CUSTOM_DATA_ROOT:/sat_data"
    --env "SEGMENTANYTREE_CHECKPOINT_DATA_CACHE=/sat_data/treeinsfused/processed_0.2_test"
  )
fi

APPTAINER_ARGS=(
  exec
  --nv
  --bind "$STAGED_INPUT_DIR:/sat_input:ro"
  --bind "$OUTPUT_DIR:/sat_output"
  --bind "$USERBASE:/sat_pyuser:ro"
  --bind "$PATCH_ROOT:/sat_patch:ro"
  --bind "$RUNTIME_DIR:/sat_runtime"
  --bind "$PATCHED_TRACKER:/home/nibio/mutable-outside-world/torch_points3d/metrics/panoptic_tracker_pointgroup_treeins.py:ro"
  --bind "$PROCESSED_DIR:/home/nibio/mutable-outside-world/processed_data_ready_for_training_sparse_1000_500_100_10"
  --env "PYTHONUSERBASE=/sat_pyuser"
  --env "PYTHONPATH=/sat_patch:/sat_pyuser/lib/python3.8/site-packages:/home/nibio/mutable-outside-world"
  --env "SEGMENTANYTREE_SERIAL_POOL=1"
  --env "SEGMENTANYTREE_ALIGNED_OUTPUT_DIR=/sat_output"
  --env "OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}"
  "${CHECKPOINT_BIND[@]}"
  "$IMAGE"
)

apptainer "${APPTAINER_ARGS[@]}" python3 -c '
import json
import numpy
import pandas
import scipy
import sklearn
import torch
versions = {
    "numpy": numpy.__version__,
    "pandas": pandas.__version__,
    "scipy": scipy.__version__,
    "scikit_learn": sklearn.__version__,
    "torch": torch.__version__,
}
with open("/sat_runtime/package_versions.json", "w", encoding="utf-8") as handle:
    json.dump(versions, handle, sort_keys=True)
    handle.write("\n")
'

START_SECONDS=$(date +%s)
set +e
/usr/bin/time -v -o "$TIME_LOG" \
  apptainer "${APPTAINER_ARGS[@]}" \
  bash /sat_patch/run_inference_for_pointwise_evaluation.sh \
    /sat_input \
    /sat_output
RETURN_CODE=$?
set -e
RUNTIME_SECONDS=$(( $(date +%s) - START_SECONDS ))

mapfile -t SEMANTIC_FILES < <(
  find "$OUTPUT_DIR" -maxdepth 1 -type f \
    -name 'semantic_segmentation_*.ply' -print | sort
)
SEMANTIC_EVALUATION="${SEMANTIC_FILES[0]:-}"
CHECKPOINT_COPY="$OUTPUT_DIR/PointGroup-PAPER.pt"
CHECKPOINT_FOR_METADATA="$CHECKPOINT_COPY"
if [[ -n "$CUSTOM_CHECKPOINT_DIR" ]]; then
  CHECKPOINT_FOR_METADATA="$CUSTOM_CHECKPOINT_DIR/PointGroup-PAPER.pt"
fi

STATUS="failed"
if [[ "$RETURN_CODE" -eq 0 && -f "$INSTANCE_EVALUATION" && -f "$SEMANTIC_EVALUATION" ]]; then
  STATUS="completed"
elif [[ "$RETURN_CODE" -eq 0 ]]; then
  RETURN_CODE=3
fi

RECORD_ARGS=(
  --output "$RUN_METADATA"
  --input-file "$REFERENCE_PATH"
  --relative-path "$RELATIVE_PATH"
  --collection "$COLLECTION"
  --plot-name "$PLOT_NAME"
  --split "$SPLIT"
  --prediction-directory "$OUTPUT_DIR"
  --image "$IMAGE"
  --external-repo "$EXTERNAL_REPO"
  --checkpoint "$CHECKPOINT_FOR_METADATA"
  --python-userbase "$USERBASE"
  --run-type "$RUN_TYPE"
  --status "$STATUS"
  --return-code "$RETURN_CODE"
  --runtime-seconds "$RUNTIME_SECONDS"
  --time-log "$TIME_LOG"
  --package-versions-json "$PACKAGE_VERSIONS"
  --aligned-instance-evaluation "$INSTANCE_EVALUATION"
)
if [[ -n "$TRAINING_RUN_ID" ]]; then
  RECORD_ARGS+=(--training-run-id "$TRAINING_RUN_ID")
fi
if [[ -n "$SEMANTIC_EVALUATION" ]]; then
  RECORD_ARGS+=(--aligned-semantic-evaluation "$SEMANTIC_EVALUATION")
fi
python methods/segmentanytree/scripts/runtime/record_segmentanytree_run.py "${RECORD_ARGS[@]}"

if [[ "$RETURN_CODE" -ne 0 ]]; then
  exit "$RETURN_CODE"
fi

echo "Aligned instance evaluation: $INSTANCE_EVALUATION"
echo "Aligned semantic evaluation: $SEMANTIC_EVALUATION"
echo "Run metadata: $RUN_METADATA"
