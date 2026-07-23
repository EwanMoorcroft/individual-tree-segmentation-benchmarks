#!/usr/bin/env bash
#SBATCH --job-name=forai_ftfull
#SBATCH --partition=gpu-a-lowsmall
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=128G
#SBATCH --time=3-00:00:00
#SBATCH --output=logs/forai_ftfull_%j.out
#SBATCH --error=logs/forai_ftfull_%j.err

set -euo pipefail
trap 'status=$?; printf "ForAINet full fine-tune failed at line %s: %s\n" "$LINENO" "$BASH_COMMAND" >&2; exit "$status"' ERR

: "${FORAINET_BENCHMARK_ROOT:?set FORAINET_BENCHMARK_ROOT}"
: "${FORAINET_EXPECTED_BENCHMARK_COMMIT:?set FORAINET_EXPECTED_BENCHMARK_COMMIT}"
: "${FORAINET_UPSTREAM_ROOT:?set FORAINET_UPSTREAM_ROOT}"
: "${FORAINET_CHECKPOINT:?set FORAINET_CHECKPOINT}"
: "${FORAINET_IMAGE:?set FORAINET_IMAGE}"
: "${FORAINET_QUALIFICATION_ROOT:?set FORAINET_QUALIFICATION_ROOT}"
: "${FORAINET_FINETUNE_ROOT:?set FORAINET_FINETUNE_ROOT}"
: "${FORAINET_FINETUNE_RUN_ID:?set FORAINET_FINETUNE_RUN_ID}"

if [[ "${FORAINET_FINETUNE_FULL_CONFIRMED:-0}" != "1" ]]; then
  echo "Refusing full fine-tune without explicit development-only confirmation."
  exit 2
fi
test "$(git -C "$FORAINET_BENCHMARK_ROOT" rev-parse HEAD)" = \
  "$FORAINET_EXPECTED_BENCHMARK_COMMIT"
test -z "$(git -C "$FORAINET_BENCHMARK_ROOT" status --porcelain)"
test -f "$FORAINET_FINETUNE_ROOT/finetune_data_manifest.json"
test -f "$FORAINET_FINETUNE_ROOT/smoke/final_gate.json"
test -f "$FORAINET_QUALIFICATION_ROOT/checkpoint_load.json"
test "$(sha256sum "$FORAINET_CHECKPOINT" | awk '{print $1}')" = \
  "97c03ce81621dc4193e55d2ca2294861b1f4421c94d192799e5fe031f9d35861"
test "$(sha256sum "$FORAINET_IMAGE" | awk '{print $1}')" = \
  "ad0df684209014c52421dc213cd0e15ddbb47214c00fac264e829f68dc17812d"

full_root="$FORAINET_FINETUNE_ROOT/full"
test ! -e "$full_root"
mkdir "$full_root"
mkdir "$full_root/training"
mkdir "$full_root/config_stage"

benchmark_root="$(readlink -f "$FORAINET_BENCHMARK_ROOT")"
benchmark_git_common_raw="$(
  git -C "$benchmark_root" rev-parse --git-common-dir
)"
case "$benchmark_git_common_raw" in
  /*) benchmark_git_common="$benchmark_git_common_raw" ;;
  *) benchmark_git_common="$(readlink -f "$benchmark_root/$benchmark_git_common_raw")" ;;
esac
upstream_root="$(readlink -f "$FORAINET_UPSTREAM_ROOT")"
checkpoint="$(readlink -f "$FORAINET_CHECKPOINT")"
image="$(readlink -f "$FORAINET_IMAGE")"
qualification_root="$(readlink -f "$FORAINET_QUALIFICATION_ROOT")"
finetune_root="$(readlink -f "$FORAINET_FINETUNE_ROOT")"
full_root="$(readlink -f "$full_root")"
data_root="$finetune_root/data_set1_5classes"
pointcloud_root="$upstream_root/PointCloudSegmentation"
official_model_config="$pointcloud_root/conf/models/panoptic/FORpartseg_3heads.yaml"
staged_model_config="$full_root/config_stage/FORpartseg_3heads.yaml"
completion_marker="$full_root/training.complete"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-12}"

module load apptainer/1.3.6
python3 \
  "$benchmark_root/methods/forainet/scripts/provenance/stage_finetune_model_config.py" \
  --official-config "$official_model_config" \
  --checkpoint "$checkpoint" \
  --output-yaml "$staged_model_config" \
  --metadata-json "$full_root/config_stage/model_config.json"

apptainer exec --cleanenv \
  --bind "$benchmark_root:$benchmark_root:ro" \
  --bind "$finetune_root:$finetune_root" \
  "$image" python3.8 \
    "$benchmark_root/methods/forainet/scripts/provenance/snapshot_finetune_checkpoints.py" \
    --checkpoint "$full_root/training/PointGroup-PAPER.pt" \
    --output-dir "$full_root/candidates" \
    --completion-marker "$completion_marker" \
    --poll-seconds 5 \
    --timeout-seconds 260000 &
snapshot_pid=$!

set +e
apptainer exec --nv --cleanenv \
  --env "OMP_NUM_THREADS=$OMP_NUM_THREADS" \
  --bind "$benchmark_root:$benchmark_root:ro" \
  --bind "$benchmark_git_common:$benchmark_git_common:ro" \
  --bind "$upstream_root:$upstream_root:ro" \
  --bind "$checkpoint:$checkpoint:ro" \
  --bind "$qualification_root:$qualification_root:ro" \
  --bind "$finetune_root:$finetune_root" \
  --bind "$staged_model_config:$official_model_config:ro" \
  --pwd "$pointcloud_root" \
  "$image" python3.8 train.py \
    task=panoptic \
    data=panoptic/treeins_set1_treemix3d \
    models=panoptic/FORpartseg_3heads \
    model_name=PointGroup-PAPER \
    training=treeins_set1_mixtree \
    job_name="$FORAINET_FINETUNE_RUN_ID-full" \
    data.dataroot="$data_root" \
    training.epochs=150 \
    training.batch_size=4 \
    training.num_workers=0 \
    training.wandb.log=false \
    training.tensorboard.log=false \
    debugging.early_break=false \
    visualization.activate=false \
    eval_frequency=1 \
    pretty_print=true \
    hydra.run.dir="$full_root/training"
training_status=$?
set -e
if [[ "$training_status" -eq 0 ]]; then
  printf 'complete\n' > "$completion_marker"
else
  printf 'failed_exit_%s\n' "$training_status" > "$completion_marker"
fi

set +e
wait "$snapshot_pid"
snapshot_status=$?
set -e
if [[ "$training_status" -ne 0 ]]; then
  exit "$training_status"
fi
if [[ "$snapshot_status" -ne 0 ]]; then
  exit "$snapshot_status"
fi

full_checkpoint="$full_root/training/PointGroup-PAPER.pt"
candidate_index="$full_root/candidates/index.json"
test -s "$full_checkpoint"
test -s "$candidate_index"
apptainer exec --cleanenv \
  --bind "$benchmark_root:$benchmark_root:ro" \
  --bind "$checkpoint:$checkpoint:ro" \
  --bind "$finetune_root:$finetune_root" \
  "$image" python3.8 \
    "$benchmark_root/methods/forainet/scripts/provenance/validate_finetune_full.py" \
    --initial-checkpoint "$checkpoint" \
    --full-checkpoint "$full_checkpoint" \
    --candidate-index "$candidate_index" \
    --candidate-root "$full_root/candidates" \
    --expected-data-root "$data_root" \
    --output-json "$full_root/validation.json"

python3 - "$full_root/validation.json" "$full_root/final_gate.json" \
  "$FORAINET_EXPECTED_BENCHMARK_COMMIT" "$FORAINET_FINETUNE_RUN_ID" <<'PY'
import hashlib
import json
import sys
source, output, commit, run_id = sys.argv[1:]
payload = json.load(open(source, encoding="utf-8"))
assert payload["status"] == "verified"
assert payload["held_out_access"] is False
digest = hashlib.sha256(open(source, "rb").read()).hexdigest()
gate = {
    "schema": "forainet_finetune_full_final_gate_v1",
    "status": "complete",
    "run_id": run_id,
    "benchmark_commit": commit,
    "validation_sha256": digest,
    "candidate_epochs": payload["candidate_epochs"],
    "held_out_access": False,
    "next_gate": "canonical_five_plot_candidate_validation",
}
open(output, "w", encoding="utf-8").write(
    json.dumps(gate, indent=2, sort_keys=True) + "\n"
)
PY
