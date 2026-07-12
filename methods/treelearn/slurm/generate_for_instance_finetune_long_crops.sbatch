#!/bin/bash -l
#SBATCH --job-name=treelearn_long_crops
#SBATCH --partition=nodes
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=12
#SBATCH --mem=128G
#SBATCH --output=logs/treelearn_for_instance/treelearn_long_crops_%A_%a.out
#SBATCH --error=logs/treelearn_for_instance/treelearn_long_crops_%A_%a.err
set -euo pipefail
cd "$HOME/scratch/tree-seg-benchmark"
test "${TREELEARN_FINETUNE_LONG_CONFIRMED:-0}" = 1
test "$(git rev-parse HEAD)" = "$TREELEARN_EXPECTED_BENCHMARK_COMMIT"
test -z "$(git status --porcelain)"
module purge
module load miniforge3/25.3.0-python3.12.10
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$TREELEARN_ENV"
python methods/treelearn/scripts/verify_for_instance_finetune_long_contract.py \
  --freeze "$TREELEARN_LONG_FREEZE" \
  --treelearn-repo "$TREELEARN_REPO"
python methods/treelearn/scripts/normalise_for_instance_finetune_long_plot.py \
  --freeze "$TREELEARN_LONG_FREEZE" \
  --task-index "$SLURM_ARRAY_TASK_ID" \
  --treelearn-repo "$TREELEARN_REPO"
CROP_CONFIG=$(python - "$TREELEARN_LONG_FREEZE" "$SLURM_ARRAY_TASK_ID" <<'PY'
import json,sys
p=json.load(open(sys.argv[1]))
print(next(r["crop_config"] for r in p["plots"] if int(r["task_index"]) == int(sys.argv[2])))
PY
)
CROP_ROOT=$(python - "$TREELEARN_LONG_FREEZE" "$SLURM_ARRAY_TASK_ID" <<'PY'
import json,sys
p=json.load(open(sys.argv[1]))
print(next(r["crop_root"] for r in p["plots"] if int(r["task_index"]) == int(sys.argv[2])))
PY
)
CROP_SEED=$(python - "$TREELEARN_LONG_FREEZE" "$SLURM_ARRAY_TASK_ID" <<'PY'
import json,sys
p=json.load(open(sys.argv[1]))
print(next(r["crop_seed"] for r in p["plots"] if int(r["task_index"]) == int(sys.argv[2])))
PY
)
cd "$TREELEARN_REPO"
export PYTHONHASHSEED="$CROP_SEED"
python "$HOME/scratch/tree-seg-benchmark/methods/treelearn/scripts/run_treelearn_crop_generation_seeded.py" \
  --treelearn-repo "$TREELEARN_REPO" \
  --config "$CROP_CONFIG" \
  --seed "$CROP_SEED"
test "$(find "$CROP_ROOT" -maxdepth 1 -type f -name '*.npz' | wc -l)" -eq 1500
cd "$HOME/scratch/tree-seg-benchmark"
python methods/treelearn/scripts/inventory_for_instance_finetune_long_crops.py \
  --freeze "$TREELEARN_LONG_FREEZE" \
  --task-index "$SLURM_ARRAY_TASK_ID"
echo status=treelearn-long-plot-crops-completed
echo held_out_test_accessed=false
