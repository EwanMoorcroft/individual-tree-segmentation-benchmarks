#!/bin/bash

set -euo pipefail

PROJECT_ROOT="$HOME/scratch/tree-seg-benchmark"
CONFIG="$PROJECT_ROOT/configs/for_instance_segmentanytree_benchmark.yml"
IMAGE="$HOME/scratch/containers/segment-any-tree_latest.sif"
USERBASE="$HOME/fastscratch/segmentanytree_pyuser_v1"
EXTERNAL_REPO="$PROJECT_ROOT/external/SegmentAnyTree"
EXPECTED_EXTERNAL_COMMIT="a3561ed8447bbb7938f059ba65a3e9c97d6e2ee9"
RUNTIME_ROOT="$HOME/fastscratch/segmentanytree_runtime"
PATCH_ROOT="$PROJECT_ROOT/scripts/methods/segmentanytree_runtime_patches"

cd "$PROJECT_ROOT"
mkdir -p logs/segmentanytree_for_instance

if [[ ! -f "$IMAGE" ]]; then
  echo "Apptainer image is missing: $IMAGE" >&2
  exit 2
fi
if [[ ! -d "$USERBASE/lib/python3.8/site-packages" ]]; then
  echo "Repaired Python userbase is missing: $USERBASE" >&2
  exit 2
fi
if [[ ! -d "$EXTERNAL_REPO/.git" ]]; then
  echo "SegmentAnyTree checkout is missing: $EXTERNAL_REPO" >&2
  exit 2
fi
ACTUAL_EXTERNAL_COMMIT=$(git -C "$EXTERNAL_REPO" rev-parse HEAD)
if [[ "$ACTUAL_EXTERNAL_COMMIT" != "$EXPECTED_EXTERNAL_COMMIT" ]]; then
  echo "Unexpected SegmentAnyTree commit: $ACTUAL_EXTERNAL_COMMIT" >&2
  echo "Expected: $EXPECTED_EXTERNAL_COMMIT" >&2
  exit 2
fi

mapfile -t PLOT_INFO < <(
  python scripts/data/select_for_instance_plot.py \
    --config "$CONFIG" \
    "$@" \
    --format lines
)
REFERENCE_PATH="${PLOT_INFO[0]}"
RELATIVE_PATH="${PLOT_INFO[1]}"
COLLECTION="${PLOT_INFO[2]}"
PLOT_NAME="${PLOT_INFO[3]}"
SPLIT="${PLOT_INFO[4]}"

STAGED_INPUT_DIR="$PROJECT_ROOT/data/interim/segmentanytree/for_instance/staged_inputs/${COLLECTION}/${PLOT_NAME}"
OUTPUT_DIR="$PROJECT_ROOT/data/predictions/segmentanytree/for_instance/${COLLECTION}/${PLOT_NAME}"
FINAL_PREDICTION="$OUTPUT_DIR/final_results/${PLOT_NAME}_out.laz"
CHECKPOINT_COPY="$OUTPUT_DIR/PointGroup-PAPER.pt"
RUN_METADATA="$PROJECT_ROOT/results/metadata/segmentanytree_for_instance/runs/${COLLECTION}/${PLOT_NAME}_run.json"
TASK_ID="${SLURM_ARRAY_TASK_ID:-pilot}"
RUNTIME_DIR="$RUNTIME_ROOT/${SLURM_JOB_ID:-manual}_${TASK_ID}_${COLLECTION}_${PLOT_NAME}"
PROCESSED_DIR="$RUNTIME_DIR/processed_data"
PATCHED_EXPORT="$RUNTIME_DIR/pandas_to_las.py"
PATCH_METADATA="$RUNTIME_DIR/pandas_to_las_patch.json"
PACKAGE_VERSIONS="$RUNTIME_DIR/package_versions.json"
TIME_LOG="$RUNTIME_DIR/time.txt"

if [[ -f "$FINAL_PREDICTION" ]]; then
  echo "Prediction already exists; leaving it unchanged: $FINAL_PREDICTION"
  exit 0
fi
if [[ -d "$OUTPUT_DIR" ]] && find "$OUTPUT_DIR" -mindepth 1 -print -quit | grep -q .; then
  echo "Prediction directory is not empty: $OUTPUT_DIR" >&2
  echo "Move or remove the incomplete output before retrying." >&2
  exit 2
fi

mkdir -p "$STAGED_INPUT_DIR" "$OUTPUT_DIR" "$PROCESSED_DIR"
cp "$REFERENCE_PATH" "$STAGED_INPUT_DIR/${PLOT_NAME}.las"

python "$PATCH_ROOT/prepare_pandas_to_las_patch.py" \
  --source "$EXTERNAL_REPO/nibio_inference/pandas_to_las.py" \
  --output "$PATCHED_EXPORT" \
  --metadata-output "$PATCH_METADATA"

APPTAINER_ARGS=(
  exec
  --nv
  --bind "$STAGED_INPUT_DIR:/sat_input:ro"
  --bind "$OUTPUT_DIR:/sat_output"
  --bind "$USERBASE:/sat_pyuser:ro"
  --bind "$PATCH_ROOT:/sat_patch:ro"
  --bind "$RUNTIME_DIR:/sat_runtime"
  --bind "$PATCHED_EXPORT:/home/nibio/mutable-outside-world/nibio_inference/pandas_to_las.py:ro"
  --bind "$PROCESSED_DIR:/home/nibio/mutable-outside-world/processed_data_ready_for_training_sparse_1000_500_100_10"
  --env "PYTHONUSERBASE=/sat_pyuser"
  --env "PYTHONPATH=/sat_patch:/sat_pyuser/lib/python3.8/site-packages:/home/nibio/mutable-outside-world"
  --env "SEGMENTANYTREE_SERIAL_POOL=1"
  --env "OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}"
  "$IMAGE"
)

apptainer "${APPTAINER_ARGS[@]}" python3 -c '
import json
import numpy
import pandas
import scipy
import sklearn
import torch
import torch_geometric
from sklearn.neighbors import KDTree
import MinkowskiEngine
import torch_points3d
del KDTree, MinkowskiEngine, torch_points3d
versions = {
    "numpy": numpy.__version__,
    "pandas": pandas.__version__,
    "scipy": scipy.__version__,
    "scikit_learn": sklearn.__version__,
    "torch": torch.__version__,
    "torch_geometric": torch_geometric.__version__,
}
with open("/sat_runtime/package_versions.json", "w", encoding="utf-8") as handle:
    json.dump(versions, handle, sort_keys=True)
    handle.write("\n")
'

START_SECONDS=$(date +%s)
set +e
/usr/bin/time -v -o "$TIME_LOG" \
  apptainer "${APPTAINER_ARGS[@]}" \
  bash -lc 'cd /home/nibio/mutable-outside-world && bash run_inference.sh /sat_input /sat_output true'
RETURN_CODE=$?
set -e
RUNTIME_SECONDS=$(( $(date +%s) - START_SECONDS ))

STATUS="failed"
if [[ "$RETURN_CODE" -eq 0 && -f "$FINAL_PREDICTION" ]]; then
  STATUS="completed"
elif [[ "$RETURN_CODE" -eq 0 ]]; then
  RETURN_CODE=3
fi

python scripts/methods/record_segmentanytree_run.py \
  --output "$RUN_METADATA" \
  --input-file "$REFERENCE_PATH" \
  --relative-path "$RELATIVE_PATH" \
  --collection "$COLLECTION" \
  --plot-name "$PLOT_NAME" \
  --split "$SPLIT" \
  --prediction-directory "$OUTPUT_DIR" \
  --final-prediction "$FINAL_PREDICTION" \
  --image "$IMAGE" \
  --external-repo "$EXTERNAL_REPO" \
  --checkpoint "$CHECKPOINT_COPY" \
  --python-userbase "$USERBASE" \
  --status "$STATUS" \
  --return-code "$RETURN_CODE" \
  --runtime-seconds "$RUNTIME_SECONDS" \
  --time-log "$TIME_LOG" \
  --package-versions-json "$PACKAGE_VERSIONS"

if [[ "$RETURN_CODE" -ne 0 ]]; then
  exit "$RETURN_CODE"
fi

echo "Prediction: $FINAL_PREDICTION"
echo "Run metadata: $RUN_METADATA"
