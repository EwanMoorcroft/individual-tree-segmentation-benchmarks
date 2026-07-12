#!/bin/bash -l
#SBATCH --job-name=treelearn_setup
#SBATCH --partition=gpu-l40s-low
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --output=logs/treelearn_for_instance/treelearn_setup_%j.out
#SBATCH --error=logs/treelearn_for_instance/treelearn_setup_%j.err

set -euo pipefail

cd "$HOME/scratch/tree-seg-benchmark"
mkdir -p logs/treelearn_for_instance

if [[ "${TREELEARN_SETUP_CONFIRMED:-0}" != "1" ]]; then
  echo "Refusing TreeLearn environment setup."
  echo "Set TREELEARN_SETUP_CONFIRMED=1 after reviewing the pinned setup."
  exit 2
fi

module purge
module load miniforge3/25.3.0-python3.12.10
source "$(conda info --base)/etc/profile.d/conda.sh"

TREELEARN_ENV="${TREELEARN_ENV:-$HOME/fastscratch/venvs/treelearn}"
TREELEARN_REPO="${TREELEARN_REPO:-$HOME/fastscratch/external/TreeLearn}"
TREELEARN_CHECKPOINT="${TREELEARN_CHECKPOINT:-$HOME/fastscratch/treelearn_checkpoints/model_weights_20241213.pth}"
TREELEARN_COMMIT="fd240ce7caa4c444fe3418aca454dc578bc557d4"
TREELEARN_CHECKPOINT_DATASET="model_weights_20241213"
TREELEARN_CHECKPOINT_MD5="56a3d78f689ae7f1190906b975700311"
TREELEARN_SETUP_RESUME_PARTIAL="${TREELEARN_SETUP_RESUME_PARTIAL:-0}"
ENV_MARKER="$TREELEARN_ENV/.treelearn_setup_complete"

trap 'status=$?; if [[ $status -ne 0 ]]; then rm -f "$ENV_MARKER"; fi' EXIT

if [[ -e "$TREELEARN_REPO" ]]; then
  test -d "$TREELEARN_REPO/.git"
  ACTUAL_COMMIT=$(git -C "$TREELEARN_REPO" rev-parse HEAD)
  test "$ACTUAL_COMMIT" = "$TREELEARN_COMMIT"
  test -z "$(git -C "$TREELEARN_REPO" status --porcelain)"
  echo "reusing_treelearn_repo=$TREELEARN_REPO"
else
  mkdir -p "$(dirname "$TREELEARN_REPO")"
  git clone https://github.com/ecker-lab/TreeLearn.git "$TREELEARN_REPO"
  git -C "$TREELEARN_REPO" checkout --detach "$TREELEARN_COMMIT"
fi

INSTALL_ENV=0
if [[ -f "$ENV_MARKER" && "$TREELEARN_SETUP_RESUME_PARTIAL" != "1" ]]; then
  test -x "$TREELEARN_ENV/bin/python"
  echo "reusing_verified_treelearn_env=$TREELEARN_ENV"
elif [[ -e "$TREELEARN_ENV" ]]; then
  if [[ ! -d "$TREELEARN_ENV/conda-meta" ]]; then
    echo "Existing TreeLearn path is not a Conda prefix: $TREELEARN_ENV"
    echo "Choose a new TREELEARN_ENV path."
    exit 1
  fi
  if [[ "$TREELEARN_SETUP_RESUME_PARTIAL" != "1" ]]; then
    echo "TreeLearn environment exists without a completed setup marker."
    echo "Set TREELEARN_SETUP_RESUME_PARTIAL=1 to resume the idempotent installation."
    exit 1
  fi
  INSTALL_ENV=1
  echo "resuming_partial_treelearn_env=$TREELEARN_ENV"
else
  mkdir -p "$(dirname "$TREELEARN_ENV")"
  conda create -p "$TREELEARN_ENV" python=3.10 -y
  INSTALL_ENV=1
fi

if [[ "$INSTALL_ENV" = "1" ]]; then
  conda install -p "$TREELEARN_ENV" \
    pytorch==2.0.0 torchvision==0.15.0 torchaudio==2.0.0 pytorch-cuda=11.8 \
    -c pytorch -c nvidia -y
  conda install -p "$TREELEARN_ENV" mkl=2024.0.0 -y
  conda install -p "$TREELEARN_ENV" \
    nvidia/label/cuda-11.8.0::cuda-toolkit -y
  conda install -p "$TREELEARN_ENV" -c conda-forge tensorboard tensorboardx -y
  "$TREELEARN_ENV/bin/pip" install \
    -r "$TREELEARN_REPO/setup/requirements.txt"
  "$TREELEARN_ENV/bin/pip" install -e "$TREELEARN_REPO"
fi

if [[ -e "$TREELEARN_CHECKPOINT" ]]; then
  test -f "$TREELEARN_CHECKPOINT"
  echo "reusing_treelearn_checkpoint=$TREELEARN_CHECKPOINT"
else
  mkdir -p "$(dirname "$TREELEARN_CHECKPOINT")"
  "$TREELEARN_ENV/bin/python" "$TREELEARN_REPO/tree_learn/util/download.py" \
    --dataset_name "$TREELEARN_CHECKPOINT_DATASET" \
    --root_folder "$(dirname "$TREELEARN_CHECKPOINT")"
fi

echo "$TREELEARN_CHECKPOINT_MD5  $TREELEARN_CHECKPOINT" | md5sum --check --status

"$TREELEARN_ENV/bin/python" -c \
  'import pathlib, sys, open3d, spconv, torch, tree_learn; repo=pathlib.Path(sys.argv[1]).resolve(); package=pathlib.Path(tree_learn.__file__).resolve(); assert sys.version_info[:2] == (3, 10); assert torch.__version__.split("+")[0] == "2.0.0"; assert torch.version.cuda == "11.8"; assert torch.cuda.is_available(); assert package.is_relative_to(repo); print("python_treelearn={}".format(package)); print("torch={}".format(torch.__version__)); print("cuda={}".format(torch.version.cuda)); print("spconv={}".format(getattr(spconv, "__version__", "unknown"))); print("open3d={}".format(open3d.__version__))' \
  "$TREELEARN_REPO"

test -z "$(git -C "$TREELEARN_REPO" status --porcelain)"
touch "$ENV_MARKER"

SETUP_RECORD="results/metadata/treelearn_for_instance/environment_setups/${SLURM_JOB_ID:-manual}"
mkdir -p "$SETUP_RECORD"
conda list -p "$TREELEARN_ENV" --explicit > "$SETUP_RECORD/conda-explicit.txt"
"$TREELEARN_ENV/bin/pip" freeze > "$SETUP_RECORD/pip-freeze.txt"
git -C "$TREELEARN_REPO" status --short > "$SETUP_RECORD/treelearn-git-status.txt"
git -C "$TREELEARN_REPO" rev-parse HEAD > "$SETUP_RECORD/treelearn-commit.txt"
sha256sum "$TREELEARN_CHECKPOINT" > "$SETUP_RECORD/checkpoint-sha256.txt"

echo "treelearn_commit=$TREELEARN_COMMIT"
echo "treelearn_env=$TREELEARN_ENV"
echo "treelearn_repo=$TREELEARN_REPO"
echo "treelearn_checkpoint=$TREELEARN_CHECKPOINT"
echo "treelearn_checkpoint_sha256=$(sha256sum "$TREELEARN_CHECKPOINT" | cut -d ' ' -f 1)"
echo "setup_record=$SETUP_RECORD"
echo "status=treelearn-setup-verified"
trap - EXIT
