#!/bin/bash

set -euo pipefail

PROJECT_ROOT="$HOME/scratch/tree-seg-benchmark"
PROFILE="${SEGMENTANYTREE_TRAIN_PROFILE:-pilot}"
RUN_ID="${SEGMENTANYTREE_TRAINING_RUN_ID:-${PROFILE}_${SLURM_JOB_ID:-manual}}"
IMAGE="$HOME/scratch/containers/segment-any-tree_latest.sif"
USERBASE="$HOME/fastscratch/segmentanytree_pyuser_v1"
EXTERNAL_REPO="$PROJECT_ROOT/external/SegmentAnyTree"
EXPECTED_EXTERNAL_COMMIT="a3561ed8447bbb7938f059ba65a3e9c97d6e2ee9"
TRAINING_DATA_ROOT="$HOME/fastscratch/segmentanytree_for_instance_training/$PROFILE"
MANIFEST="$PROJECT_ROOT/results/metadata/segmentanytree_for_instance/training_splits/${PROFILE}_split_manifest.json"
TRAINING_OUTPUT_ROOT="$HOME/fastscratch/segmentanytree_for_instance_checkpoints/$RUN_ID"
RUN_METADATA="$PROJECT_ROOT/results/metadata/segmentanytree_for_instance/training_runs/${RUN_ID}.json"
COMMAND_FILE="$TRAINING_OUTPUT_ROOT/training_command.txt"
PACKAGE_VERSIONS="$TRAINING_OUTPUT_ROOT/package_versions.json"
TIME_LOG="$TRAINING_OUTPUT_ROOT/time.txt"
PATCHED_TRAINER="$TRAINING_OUTPUT_ROOT/trainer_dev_only.py"
PATCH_METADATA="$TRAINING_OUTPUT_ROOT/trainer_dev_only_patch.json"

if [[ "${SEGMENTANYTREE_EXECUTE:-0}" != "1" ]]; then
  echo "Refusing training. Set SEGMENTANYTREE_EXECUTE=1 after reviewing the split manifest." >&2
  exit 2
fi
if [[ "$PROFILE" == "pilot" ]]; then
  REQUESTED_EPOCHS="${SEGMENTANYTREE_TRAIN_EPOCHS:-2}"
  BATCH_SIZE="${SEGMENTANYTREE_TRAIN_BATCH_SIZE:-2}"
  RUN_TYPE="pilot_training"
elif [[ "$PROFILE" == "full" ]]; then
  REQUESTED_EPOCHS="${SEGMENTANYTREE_TRAIN_EPOCHS:-150}"
  BATCH_SIZE="${SEGMENTANYTREE_TRAIN_BATCH_SIZE:-4}"
  RUN_TYPE="full_training"
else
  echo "SEGMENTANYTREE_TRAIN_PROFILE must be pilot or full." >&2
  exit 2
fi
if [[ ! "$REQUESTED_EPOCHS" =~ ^[1-9][0-9]*$ ]]; then
  echo "SEGMENTANYTREE_TRAIN_EPOCHS must be a positive integer." >&2
  exit 2
fi
if [[ ! "$BATCH_SIZE" =~ ^[1-9][0-9]*$ ]]; then
  echo "SEGMENTANYTREE_TRAIN_BATCH_SIZE must be a positive integer." >&2
  exit 2
fi

# The pinned trainer uses range(start_epoch, training.epochs), so the stop
# value must be one greater than the requested number of epochs.
HYDRA_STOP_EPOCH=$((REQUESTED_EPOCHS + 1))

cd "$PROJECT_ROOT"
mkdir -p logs/segmentanytree_for_instance "$(dirname "$RUN_METADATA")"

test -f "$IMAGE"
test -d "$USERBASE/lib/python3.8/site-packages"
test -f "$MANIFEST"
test -d "$TRAINING_DATA_ROOT/treeinsfused/raw"
test -d "$EXTERNAL_REPO/.git"

ACTUAL_EXTERNAL_COMMIT=$(git -C "$EXTERNAL_REPO" rev-parse HEAD)
if [[ "$ACTUAL_EXTERNAL_COMMIT" != "$EXPECTED_EXTERNAL_COMMIT" ]]; then
  echo "Unexpected SegmentAnyTree commit: $ACTUAL_EXTERNAL_COMMIT" >&2
  echo "Expected: $EXPECTED_EXTERNAL_COMMIT" >&2
  exit 2
fi
if [[ -d "$TRAINING_OUTPUT_ROOT" ]] && find "$TRAINING_OUTPUT_ROOT" -mindepth 1 -print -quit | grep -q .; then
  echo "Training output already exists: $TRAINING_OUTPUT_ROOT" >&2
  echo "Choose a new SEGMENTANYTREE_TRAINING_RUN_ID." >&2
  exit 2
fi
mkdir -p "$TRAINING_OUTPUT_ROOT"

python methods/segmentanytree/scripts/runtime/patches/prepare_dev_only_trainer_patch.py \
  --source "$EXTERNAL_REPO/torch_points3d/trainer.py" \
  --output "$PATCHED_TRAINER" \
  --metadata-output "$PATCH_METADATA"
python -m py_compile "$PATCHED_TRAINER"

APPTAINER_ARGS=(
  exec
  --nv
  --bind "$EXTERNAL_REPO:/sat_repo:ro"
  --bind "$TRAINING_DATA_ROOT:/sat_data"
  --bind "$TRAINING_OUTPUT_ROOT:/sat_output"
  --bind "$USERBASE:/sat_pyuser:ro"
  --bind "$PATCHED_TRAINER:/sat_repo/torch_points3d/trainer.py:ro"
  --env "PYTHONUSERBASE=/sat_pyuser"
  --env "PYTHONPATH=/sat_pyuser/lib/python3.8/site-packages:/sat_repo"
  --env "HYDRA_FULL_ERROR=1"
  --env "OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}"
  "$IMAGE"
)

TRAIN_COMMAND=(
  python3
  /sat_repo/train.py
  task=panoptic
  data=panoptic/treeins_rad8
  models=panoptic/area4_ablation_3heads_5
  model_name=PointGroup-PAPER
  training=treeins
  "job_name=$RUN_ID"
  data.dataroot=/sat_data
  data.fold=[]
  data.forest_regions=[]
  "epochs=$HYDRA_STOP_EPOCH"
  "batch_size=$BATCH_SIZE"
  num_workers=0
  wandb.log=false
  tensorboard.log=false
  selection_stage=val
  eval_frequency=1
  pretty_print=true
  hydra.run.dir=/sat_output/run
)

{
  printf 'apptainer'
  printf ' %q' "${APPTAINER_ARGS[@]}"
  printf ' bash -lc '
  printf '%q' 'cd /sat_output && "$@"'
  printf ' training'
  printf ' %q' "${TRAIN_COMMAND[@]}"
  printf '\n'
} > "$COMMAND_FILE"

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
with open("/sat_output/package_versions.json", "w", encoding="utf-8") as handle:
    json.dump(versions, handle, sort_keys=True)
    handle.write("\n")
'

set +e
/usr/bin/time -v -o "$TIME_LOG" \
  apptainer "${APPTAINER_ARGS[@]}" \
  bash -lc 'cd /sat_output && "$@"' training "${TRAIN_COMMAND[@]}"
RETURN_CODE=$?
set -e

CHECKPOINT=$(find "$TRAINING_OUTPUT_ROOT" -type f -name 'PointGroup-PAPER.pt' -print -quit)
STATUS="failed"
if [[ "$RETURN_CODE" -eq 0 && -n "$CHECKPOINT" && -f "$CHECKPOINT" ]]; then
  STATUS="completed"
elif [[ "$RETURN_CODE" -eq 0 ]]; then
  RETURN_CODE=3
fi

RECORD_ARGS=(
  --output "$RUN_METADATA"
  --run-id "$RUN_ID"
  --run-type "$RUN_TYPE"
  --training-mode retrained_from_dev
  --profile "$PROFILE"
  --split-manifest "$MANIFEST"
  --training-data-root "$TRAINING_DATA_ROOT"
  --training-output-root "$TRAINING_OUTPUT_ROOT"
  --external-repo "$EXTERNAL_REPO"
  --image "$IMAGE"
  --python-userbase "$USERBASE"
  --command-file "$COMMAND_FILE"
  --package-versions-json "$PACKAGE_VERSIONS"
  --time-log "$TIME_LOG"
  --requested-epochs "$REQUESTED_EPOCHS"
  --hydra-stop-epoch "$HYDRA_STOP_EPOCH"
  --batch-size "$BATCH_SIZE"
  --status "$STATUS"
  --return-code "$RETURN_CODE"
)
if [[ -n "$CHECKPOINT" ]]; then
  RECORD_ARGS+=(--checkpoint "$CHECKPOINT")
fi
python methods/segmentanytree/scripts/runtime/record_segmentanytree_training_run.py "${RECORD_ARGS[@]}"

if [[ "$RETURN_CODE" -ne 0 ]]; then
  exit "$RETURN_CODE"
fi

echo "Training checkpoint: $CHECKPOINT"
echo "Training metadata: $RUN_METADATA"
