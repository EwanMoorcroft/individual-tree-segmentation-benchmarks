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
PATCHED_MEANSHIFT="$TRAINING_OUTPUT_ROOT/meanshift_cluster_spawn.py"
MEANSHIFT_PATCH_METADATA="$TRAINING_OUTPUT_ROOT/meanshift_cluster_spawn_patch.json"
STALL_MARKER="$TRAINING_OUTPUT_ROOT/stall_watchdog.txt"
STALL_TIMEOUT_SECONDS="${SEGMENTANYTREE_STALL_TIMEOUT_SECONDS:-1200}"
DIAGNOSTIC_STACK_SECONDS="${SEGMENTANYTREE_DIAGNOSTIC_STACK_SECONDS:-0}"
MEANSHIFT_JOBS="${SEGMENTANYTREE_MEANSHIFT_JOBS:-2}"
OMP_THREADS="${SEGMENTANYTREE_OMP_NUM_THREADS:-1}"
RESUME_CHECKPOINT="${SEGMENTANYTREE_RESUME_CHECKPOINT:-}"
RESUME_CHECKPOINT_SHA256="${SEGMENTANYTREE_RESUME_CHECKPOINT_SHA256:-}"
PRETRAINED_CHECKPOINT="${SEGMENTANYTREE_PRETRAINED_CHECKPOINT:-}"
PRETRAINED_CHECKPOINT_SHA256="${SEGMENTANYTREE_PRETRAINED_CHECKPOINT_SHA256:-}"
PRETRAINED_WEIGHT_NAME="${SEGMENTANYTREE_PRETRAINED_WEIGHT_NAME:-latest}"
PRETRAINED_MIN_FRACTION="${SEGMENTANYTREE_PRETRAINED_MIN_COMPATIBLE_FRACTION:-0.95}"
PRETRAINED_VALIDATION="$TRAINING_OUTPUT_ROOT/pretrained_load_validation.json"
BASE_LR="${SEGMENTANYTREE_TRAIN_BASE_LR:-}"
RESUME_START_EPOCH=""
TRAINING_MODE="retrained_from_dev"

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
if [[ ! "$STALL_TIMEOUT_SECONDS" =~ ^[1-9][0-9]*$ ]]; then
  echo "SEGMENTANYTREE_STALL_TIMEOUT_SECONDS must be a positive integer." >&2
  exit 2
fi
if [[ ! "$DIAGNOSTIC_STACK_SECONDS" =~ ^[0-9]+$ ]]; then
  echo "SEGMENTANYTREE_DIAGNOSTIC_STACK_SECONDS must be a non-negative integer." >&2
  exit 2
fi
if [[ ! "$MEANSHIFT_JOBS" =~ ^[1-9][0-9]*$ ]]; then
  echo "SEGMENTANYTREE_MEANSHIFT_JOBS must be a positive integer." >&2
  exit 2
fi
if [[ ! "$OMP_THREADS" =~ ^[1-9][0-9]*$ ]]; then
  echo "SEGMENTANYTREE_OMP_NUM_THREADS must be a positive integer." >&2
  exit 2
fi
if [[ -n "$BASE_LR" && ! "$BASE_LR" =~ ^[0-9]+([.][0-9]+)?([eE]-?[0-9]+)?$ ]]; then
  echo "SEGMENTANYTREE_TRAIN_BASE_LR must be a non-negative number." >&2
  exit 2
fi
if [[ -n "$RESUME_CHECKPOINT" && -n "$PRETRAINED_CHECKPOINT" ]]; then
  echo "Resume and pretrained weight-only initialization are mutually exclusive." >&2
  exit 2
fi
if [[ -n "$RESUME_CHECKPOINT" ]]; then
  if [[ ! -f "$RESUME_CHECKPOINT" ]]; then
    echo "Resume checkpoint does not exist: $RESUME_CHECKPOINT" >&2
    exit 2
  fi
  RESUME_CHECKPOINT=$(realpath "$RESUME_CHECKPOINT")
  if [[ "$(basename "$RESUME_CHECKPOINT")" != "PointGroup-PAPER.pt" ]]; then
    echo "Resume checkpoint must be named PointGroup-PAPER.pt." >&2
    exit 2
  fi
  if [[ ! "$RESUME_CHECKPOINT_SHA256" =~ ^[0-9a-f]{64}$ ]]; then
    echo "Set SEGMENTANYTREE_RESUME_CHECKPOINT_SHA256 to the reviewed checkpoint hash." >&2
    exit 2
  fi
  ACTUAL_RESUME_SHA256=$(sha256sum "$RESUME_CHECKPOINT" | awk '{print $1}')
  if [[ "$ACTUAL_RESUME_SHA256" != "$RESUME_CHECKPOINT_SHA256" ]]; then
    echo "Resume checkpoint SHA-256 mismatch: $ACTUAL_RESUME_SHA256" >&2
    echo "Expected: $RESUME_CHECKPOINT_SHA256" >&2
    exit 2
  fi
  RESUME_CHECKPOINT_DIR=$(dirname "$RESUME_CHECKPOINT")
  TRAINING_MODE="resumed_from_dev_checkpoint"
elif [[ -n "$RESUME_CHECKPOINT_SHA256" ]]; then
  echo "SEGMENTANYTREE_RESUME_CHECKPOINT_SHA256 requires SEGMENTANYTREE_RESUME_CHECKPOINT." >&2
  exit 2
fi
if [[ -n "$PRETRAINED_CHECKPOINT" ]]; then
  if [[ ! -f "$PRETRAINED_CHECKPOINT" ]]; then
    echo "Pretrained checkpoint does not exist: $PRETRAINED_CHECKPOINT" >&2
    exit 2
  fi
  PRETRAINED_CHECKPOINT=$(realpath "$PRETRAINED_CHECKPOINT")
  if [[ "$(basename "$PRETRAINED_CHECKPOINT")" != "PointGroup-PAPER.pt" ]]; then
    echo "Pretrained checkpoint must be named PointGroup-PAPER.pt." >&2
    exit 2
  fi
  if [[ ! "$PRETRAINED_CHECKPOINT_SHA256" =~ ^[0-9a-f]{64}$ ]]; then
    echo "Set SEGMENTANYTREE_PRETRAINED_CHECKPOINT_SHA256 to the reviewed checkpoint hash." >&2
    exit 2
  fi
  ACTUAL_PRETRAINED_SHA256=$(sha256sum "$PRETRAINED_CHECKPOINT" | awk '{print $1}')
  if [[ "$ACTUAL_PRETRAINED_SHA256" != "$PRETRAINED_CHECKPOINT_SHA256" ]]; then
    echo "Pretrained checkpoint SHA-256 mismatch: $ACTUAL_PRETRAINED_SHA256" >&2
    echo "Expected: $PRETRAINED_CHECKPOINT_SHA256" >&2
    exit 2
  fi
  if [[ ! "$PRETRAINED_WEIGHT_NAME" =~ ^[A-Za-z0-9_.-]+$ ]]; then
    echo "SEGMENTANYTREE_PRETRAINED_WEIGHT_NAME contains unsupported characters." >&2
    exit 2
  fi
  if [[ ! "$PRETRAINED_MIN_FRACTION" =~ ^0([.][0-9]+)?$|^1([.]0+)?$ ]]; then
    echo "SEGMENTANYTREE_PRETRAINED_MIN_COMPATIBLE_FRACTION must be between 0 and 1." >&2
    exit 2
  fi
  PRETRAINED_CHECKPOINT_DIR=$(dirname "$PRETRAINED_CHECKPOINT")
  TRAINING_MODE="fine_tuned_on_dev"
elif [[ -n "$PRETRAINED_CHECKPOINT_SHA256" ]]; then
  echo "SEGMENTANYTREE_PRETRAINED_CHECKPOINT_SHA256 requires SEGMENTANYTREE_PRETRAINED_CHECKPOINT." >&2
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
python methods/segmentanytree/scripts/runtime/patches/prepare_spawn_meanshift_patch.py \
  --source "$EXTERNAL_REPO/torch_points3d/utils/meanshift_cluster.py" \
  --output "$PATCHED_MEANSHIFT" \
  --metadata-output "$MEANSHIFT_PATCH_METADATA"
python -m py_compile "$PATCHED_MEANSHIFT"

APPTAINER_ARGS=(
  exec
  --nv
  --bind "$EXTERNAL_REPO:/sat_repo:ro"
  --bind "$TRAINING_DATA_ROOT:/sat_data"
  --bind "$TRAINING_OUTPUT_ROOT:/sat_output"
  --bind "$USERBASE:/sat_pyuser:ro"
  --bind "$PATCHED_TRAINER:/sat_repo/torch_points3d/trainer.py:ro"
  --bind "$PATCHED_MEANSHIFT:/sat_repo/torch_points3d/utils/meanshift_cluster.py:ro"
  --env "PYTHONUSERBASE=/sat_pyuser"
  --env "PYTHONPATH=/sat_pyuser/lib/python3.8/site-packages:/sat_repo"
  --env "HYDRA_FULL_ERROR=1"
  --env "OMP_NUM_THREADS=$OMP_THREADS"
  --env "MKL_NUM_THREADS=$OMP_THREADS"
  --env "OPENBLAS_NUM_THREADS=$OMP_THREADS"
  --env "SEGMENTANYTREE_MEANSHIFT_JOBS=$MEANSHIFT_JOBS"
  --env "SEGMENTANYTREE_DIAGNOSTIC_STACK_SECONDS=$DIAGNOSTIC_STACK_SECONDS"
)
if [[ -n "$RESUME_CHECKPOINT" ]]; then
  APPTAINER_ARGS+=(--bind "$RESUME_CHECKPOINT_DIR:/sat_resume:ro")
fi
if [[ -n "$PRETRAINED_CHECKPOINT" ]]; then
  APPTAINER_ARGS+=(
    --bind "$PRETRAINED_CHECKPOINT_DIR:/sat_pretrained:ro"
    --env "SEGMENTANYTREE_REQUIRE_PRETRAINED_LOAD=1"
    --env "SEGMENTANYTREE_PRETRAINED_PATH=/sat_pretrained/PointGroup-PAPER.pt"
    --env "SEGMENTANYTREE_PRETRAINED_WEIGHT_NAME=$PRETRAINED_WEIGHT_NAME"
    --env "SEGMENTANYTREE_PRETRAINED_MIN_COMPATIBLE_FRACTION=$PRETRAINED_MIN_FRACTION"
    --env "SEGMENTANYTREE_PRETRAINED_VALIDATION_OUTPUT=/sat_output/pretrained_load_validation.json"
  )
fi
APPTAINER_ARGS+=("$IMAGE")

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
if [[ -n "$RESUME_CHECKPOINT" ]]; then
  TRAIN_COMMAND+=(
    checkpoint_dir=/sat_resume
    weight_name=latest
  )
  read -r COMPLETED_TRAIN_COUNT COMPLETED_TRAIN_EPOCH \
    COMPLETED_VAL_COUNT COMPLETED_VAL_EPOCH < <(
    apptainer "${APPTAINER_ARGS[@]}" python3 -c '
import torch
checkpoint = torch.load("/sat_resume/PointGroup-PAPER.pt", map_location="cpu")
stats = checkpoint["stats"]
train = stats.get("train", [])
val = stats.get("val", [])
print(
    len(train),
    train[-1]["epoch"] if train else 0,
    len(val),
    val[-1]["epoch"] if val else 0,
)
'
  )
  if [[ "$COMPLETED_TRAIN_COUNT" -le 0 ||
        "$COMPLETED_TRAIN_COUNT" != "$COMPLETED_TRAIN_EPOCH" ||
        "$COMPLETED_VAL_COUNT" != "$COMPLETED_VAL_EPOCH" ||
        "$COMPLETED_TRAIN_EPOCH" != "$COMPLETED_VAL_EPOCH" ]]; then
    echo "Resume checkpoint has inconsistent epoch history." >&2
    exit 2
  fi
  RESUME_START_EPOCH=$((COMPLETED_TRAIN_COUNT + 1))
  if (( RESUME_START_EPOCH > REQUESTED_EPOCHS )); then
    echo "Resume checkpoint already reached requested epoch $REQUESTED_EPOCHS." >&2
    exit 2
  fi
fi
if [[ -n "$BASE_LR" ]]; then
  # training/treeins.yaml has no package directive in the pinned Hydra stack,
  # so its optimiser settings are composed into the global config.
  TRAIN_COMMAND+=("optim.base_lr=$BASE_LR")
fi

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

run_stall_watchdog() {
  local process_group_id="$1"
  local progress_log="$2"
  local last_update now

  while kill -0 "$process_group_id" 2>/dev/null; do
    sleep 60
    kill -0 "$process_group_id" 2>/dev/null || return 0
    [[ -f "$progress_log" ]] || continue
    last_update=$(stat -c %Y "$progress_log")
    now=$(date +%s)
    if (( now - last_update >= STALL_TIMEOUT_SECONDS )); then
      printf 'No training log progress for %s seconds at %s.\n' \
        "$STALL_TIMEOUT_SECONDS" "$(date --iso-8601=seconds)" > "$STALL_MARKER"
      kill -TERM -- "-$process_group_id" 2>/dev/null || true
      sleep 60
      kill -KILL -- "-$process_group_id" 2>/dev/null || true
      return 0
    fi
  done
}

WATCHDOG_PID=""
if [[ -n "${SLURM_JOB_ID:-}" && -n "${SLURM_JOB_NAME:-}" ]]; then
  PROGRESS_LOG="$PROJECT_ROOT/logs/segmentanytree_for_instance/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.err"
fi

set +e
setsid /usr/bin/time -v -o "$TIME_LOG" \
  apptainer "${APPTAINER_ARGS[@]}" \
  bash -lc 'cd /sat_output && "$@"' training "${TRAIN_COMMAND[@]}" &
TRAIN_PID=$!
if [[ -n "${PROGRESS_LOG:-}" ]]; then
  run_stall_watchdog "$TRAIN_PID" "$PROGRESS_LOG" &
  WATCHDOG_PID=$!
fi
wait "$TRAIN_PID"
RETURN_CODE=$?
if [[ -n "$WATCHDOG_PID" ]]; then
  if [[ -f "$STALL_MARKER" ]]; then
    wait "$WATCHDOG_PID" 2>/dev/null || true
  else
    kill "$WATCHDOG_PID" 2>/dev/null || true
    wait "$WATCHDOG_PID" 2>/dev/null || true
  fi
fi
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
  --training-mode "$TRAINING_MODE"
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
  --stall-timeout-seconds "$STALL_TIMEOUT_SECONDS"
)
if [[ -n "$CHECKPOINT" ]]; then
  RECORD_ARGS+=(--checkpoint "$CHECKPOINT")
fi
if [[ -n "$RESUME_CHECKPOINT" ]]; then
  RECORD_ARGS+=(
    --resume-checkpoint "$RESUME_CHECKPOINT"
    --resume-checkpoint-sha256 "$RESUME_CHECKPOINT_SHA256"
    --resume-start-epoch "$RESUME_START_EPOCH"
  )
fi
if [[ -n "$PRETRAINED_CHECKPOINT" ]]; then
  RECORD_ARGS+=(
    --pretrained-checkpoint "$PRETRAINED_CHECKPOINT"
    --pretrained-checkpoint-sha256 "$PRETRAINED_CHECKPOINT_SHA256"
    --pretrained-weight-name "$PRETRAINED_WEIGHT_NAME"
    --pretrained-validation-json "$PRETRAINED_VALIDATION"
  )
fi
if [[ -n "$BASE_LR" ]]; then
  RECORD_ARGS+=(--base-lr "$BASE_LR")
fi
if [[ -f "$STALL_MARKER" ]]; then
  RECORD_ARGS+=(--stall-marker "$STALL_MARKER")
fi
python methods/segmentanytree/scripts/runtime/record_segmentanytree_training_run.py "${RECORD_ARGS[@]}"

if [[ "$RETURN_CODE" -ne 0 ]]; then
  exit "$RETURN_CODE"
fi

echo "Training checkpoint: $CHECKPOINT"
echo "Training metadata: $RUN_METADATA"
