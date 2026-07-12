#!/bin/bash -l
#SBATCH --job-name=treelearn_smoke
#SBATCH --partition=gpu-l40s-low
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=12
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --output=logs/treelearn_for_instance/treelearn_one_plot_smoke_%j.out
#SBATCH --error=logs/treelearn_for_instance/treelearn_one_plot_smoke_%j.err

set -euo pipefail

cd ~/scratch/tree-seg-benchmark
mkdir -p logs/treelearn_for_instance

if [[ "${TREELEARN_SMOKE_CONFIRMED:-0}" != "1" ]]; then
  echo "Refusing to run TreeLearn smoke test."
  echo "Set TREELEARN_SMOKE_CONFIRMED=1 after checking the one-plot route."
  exit 2
fi

module purge
module load miniforge3/25.3.0-python3.12.10
source "$(conda info --base)/etc/profile.d/conda.sh"

TREELEARN_ENV="${TREELEARN_ENV:-$HOME/fastscratch/venvs/treelearn}"
TREELEARN_REPO="${TREELEARN_REPO:-$HOME/fastscratch/external/TreeLearn}"
TREELEARN_CHECKPOINT="${TREELEARN_CHECKPOINT:-$HOME/fastscratch/treelearn_checkpoints/model_weights_20241213.pth}"
TREELEARN_DATASET_ROOT="${TREELEARN_DATASET_ROOT:-$HOME/data/datasets/for_instance/FORinstance_dataset}"
TREELEARN_RUN_ID="${TREELEARN_RUN_ID:?TREELEARN_RUN_ID is required}"
TREELEARN_EXPECTED_BENCHMARK_COMMIT="${TREELEARN_EXPECTED_BENCHMARK_COMMIT:?TREELEARN_EXPECTED_BENCHMARK_COMMIT is required}"
TREELEARN_EXPECTED_CHECKPOINT_MD5="${TREELEARN_EXPECTED_CHECKPOINT_MD5:?TREELEARN_EXPECTED_CHECKPOINT_MD5 is required}"
TREELEARN_EXPECTED_CHECKPOINT_SHA256="${TREELEARN_EXPECTED_CHECKPOINT_SHA256:?TREELEARN_EXPECTED_CHECKPOINT_SHA256 is required}"

if [[ ! -x "$TREELEARN_ENV/bin/python" ]]; then
  echo "Missing TreeLearn environment: $TREELEARN_ENV"
  exit 1
fi
if [[ ! -d "$TREELEARN_REPO" ]]; then
  echo "Missing TreeLearn checkout: $TREELEARN_REPO"
  exit 1
fi
if [[ ! -f "$TREELEARN_CHECKPOINT" ]]; then
  echo "Missing TreeLearn checkpoint: $TREELEARN_CHECKPOINT"
  exit 1
fi
if [[ ! -d "$TREELEARN_DATASET_ROOT" ]]; then
  echo "Missing FOR-instance dataset root: $TREELEARN_DATASET_ROOT"
  exit 1
fi
test -f "$TREELEARN_DATASET_ROOT/data_split_metadata.csv"

ACTUAL_BENCHMARK_COMMIT=$(git rev-parse HEAD)
if [[ "$ACTUAL_BENCHMARK_COMMIT" != "$TREELEARN_EXPECTED_BENCHMARK_COMMIT" ]]; then
  echo "Benchmark checkout changed after submission."
  exit 1
fi
if [[ -n "$(git status --porcelain)" ]]; then
  echo "Benchmark checkout is dirty."
  exit 1
fi

ACTUAL_CHECKPOINT_MD5=$(md5sum "$TREELEARN_CHECKPOINT" | cut -d ' ' -f 1)
if [[ "$ACTUAL_CHECKPOINT_MD5" != "$TREELEARN_EXPECTED_CHECKPOINT_MD5" ]]; then
  echo "TreeLearn checkpoint MD5 does not match the official default."
  exit 1
fi
ACTUAL_CHECKPOINT_SHA256=$(sha256sum "$TREELEARN_CHECKPOINT" | cut -d ' ' -f 1)
if [[ "$ACTUAL_CHECKPOINT_SHA256" != "$TREELEARN_EXPECTED_CHECKPOINT_SHA256" ]]; then
  echo "TreeLearn checkpoint SHA-256 does not match the frozen submission."
  exit 1
fi

conda activate "$TREELEARN_ENV"

echo "hostname=$(hostname)"
echo "date_utc=$(date -u)"
echo "job_id=${SLURM_JOB_ID:-manual}"
echo "partition=${SLURM_JOB_PARTITION:-unknown}"
echo "cpus_per_task=${SLURM_CPUS_PER_TASK:-12}"
echo "memory_request=128G"
echo "gpu_request=gpu:1"
echo "expected_runtime=30-90 minutes"
echo "treelearn_env=$TREELEARN_ENV"
echo "treelearn_repo=$TREELEARN_REPO"
echo "treelearn_checkpoint=$TREELEARN_CHECKPOINT"
echo "treelearn_checkpoint_md5=$ACTUAL_CHECKPOINT_MD5"
echo "treelearn_checkpoint_sha256=$ACTUAL_CHECKPOINT_SHA256"
echo "treelearn_run_id=$TREELEARN_RUN_ID"
echo "benchmark_commit=$ACTUAL_BENCHMARK_COMMIT"
echo "dataset_root=$TREELEARN_DATASET_ROOT"
echo "plot=CULS/plot_1_annotated.las"
echo "success_criteria=imports, checkpoint load, full-forest output, aligned adapted NPZ/LAS, positive predictions"
echo "failure_indicators=missing env/repo/checkpoint, CUDA/spconv import failure, OOM, missing output, row mismatch, zero positive predictions"

python - "$TREELEARN_REPO" <<'PY'
from pathlib import Path
import sys
import spconv
import torch
import tree_learn

repo = Path(sys.argv[1]).resolve()
package = Path(tree_learn.__file__).resolve()
print(f"torch={torch.__version__}")
print(f"python={sys.version.split()[0]}")
print(f"torch_cuda={torch.version.cuda}")
print(f"cuda_available={torch.cuda.is_available()}")
print(f"cuda_device_count={torch.cuda.device_count()}")
print(f"spconv={getattr(spconv, '__version__', 'unknown')}")
print(f"tree_learn={package}")
if not torch.cuda.is_available():
    raise SystemExit("CUDA is not available in the active TreeLearn environment.")
if sys.version_info[:2] != (3, 10):
    raise SystemExit(f"Expected Python 3.10, found {sys.version.split()[0]}")
if torch.__version__.split("+")[0] != "2.0.0":
    raise SystemExit(f"Expected PyTorch 2.0.0, found {torch.__version__}")
if torch.version.cuda != "11.8":
    raise SystemExit(f"Expected PyTorch CUDA 11.8, found {torch.version.cuda}")
if not package.is_relative_to(repo):
    raise SystemExit(f"tree_learn imported from {package}, not pinned repo {repo}")
PY

python methods/treelearn/scripts/run_for_instance_one_plot_smoke.py \
  --config methods/treelearn/configs/for_instance_one_plot_smoke.yml \
  --dataset-root "$TREELEARN_DATASET_ROOT" \
  --treelearn-repo "$TREELEARN_REPO" \
  --checkpoint "$TREELEARN_CHECKPOINT" \
  --run-id "$TREELEARN_RUN_ID"

echo "TreeLearn one-plot smoke test completed."
