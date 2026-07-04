#!/bin/bash

set -euo pipefail

SOURCE_DIR="${1:?Usage: run_inference_for_pointwise_evaluation.sh SOURCE_DIR DEST_DIR}"
DEST_DIR="${2:?Usage: run_inference_for_pointwise_evaluation.sh SOURCE_DIR DEST_DIR}"
SCRIPT_DIR="/home/nibio/mutable-outside-world"

mkdir -p "$DEST_DIR"
if find "$DEST_DIR" -mindepth 1 -print -quit | grep -q .; then
  echo "Evaluation output directory is not empty: $DEST_DIR" >&2
  exit 2
fi

mkdir -p "$DEST_DIR/input_data"
cp -r "$SOURCE_DIR/"* "$DEST_DIR/input_data/"

python3 "$SCRIPT_DIR/nibio_inference/fix_naming_of_input_files.py" \
  "$DEST_DIR/input_data"
python3 "$SCRIPT_DIR/nibio_inference/pipeline_utm2local_parallel.py" \
  -i "$DEST_DIR/input_data" \
  -o "$DEST_DIR/utm2local"

cp "$SCRIPT_DIR/conf/eval.yaml" "$DEST_DIR/eval.yaml"
python3 "$SCRIPT_DIR/nibio_inference/modify_eval.py" \
  "$DEST_DIR/eval.yaml" \
  "$DEST_DIR/utm2local" \
  "$DEST_DIR"
python3 "$SCRIPT_DIR/nibio_inference/clear_cache.py" \
  --eval_yaml "$DEST_DIR/eval.yaml"

if [[ -n "${SEGMENTANYTREE_CHECKPOINT_DATA_CACHE:-}" ]]; then
  mkdir -p "$SEGMENTANYTREE_CHECKPOINT_DATA_CACHE"
fi

cd "$SCRIPT_DIR"
python3 eval.py --config-name "$DEST_DIR/eval.yaml"

python3 "$SCRIPT_DIR/nibio_inference/rename_result_files_instance.py" \
  "$DEST_DIR/eval.yaml" \
  "$DEST_DIR"
python3 "$SCRIPT_DIR/nibio_inference/rename_result_files_segmentation.py" \
  "$DEST_DIR/eval.yaml" \
  "$DEST_DIR"

mapfile -t INSTANCE_FILES < <(
  find "$DEST_DIR" -maxdepth 1 -type f \
    -name 'Instance_results_forEval_*.ply' -print | sort
)
mapfile -t SEMANTIC_FILES < <(
  find "$DEST_DIR" -maxdepth 1 -type f \
    -name 'semantic_segmentation_*.ply' -print | sort
)

if [[ "${#INSTANCE_FILES[@]}" -ne 1 ]]; then
  echo "Expected one aligned instance evaluation PLY; found ${#INSTANCE_FILES[@]}" >&2
  exit 3
fi
if [[ "${#SEMANTIC_FILES[@]}" -ne 1 ]]; then
  echo "Expected one aligned semantic evaluation PLY; found ${#SEMANTIC_FILES[@]}" >&2
  exit 3
fi

echo "Aligned instance evaluation: ${INSTANCE_FILES[0]}"
echo "Aligned semantic evaluation: ${SEMANTIC_FILES[0]}"
