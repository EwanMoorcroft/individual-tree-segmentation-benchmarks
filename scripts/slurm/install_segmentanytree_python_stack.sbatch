#!/bin/bash -l
#SBATCH --job-name=for_sat_stack
#SBATCH --partition=gpu-l40s
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=logs/segmentanytree_for_instance/%x_%j.out
#SBATCH --error=logs/segmentanytree_for_instance/%x_%j.err

set -euo pipefail

cd "$HOME/scratch/tree-seg-benchmark"
module purge
module load apptainer/1.3.6
mkdir -p logs/segmentanytree_for_instance

IMAGE="$HOME/scratch/containers/segment-any-tree_latest.sif"
USERBASE="$HOME/fastscratch/segmentanytree_pyuser_v1"
mkdir -p "$USERBASE"

apptainer exec --nv \
  --bind "$USERBASE:/sat_pyuser" \
  --env "PYTHONUSERBASE=/sat_pyuser" \
  "$IMAGE" \
  python3 -m pip install --user --force-reinstall \
    "numpy==1.24.4" \
    "pandas==1.5.3" \
    "scipy==1.10.1" \
    "scikit-learn==1.3.2"

apptainer exec --nv \
  --bind "$USERBASE:/sat_pyuser:ro" \
  --env "PYTHONUSERBASE=/sat_pyuser" \
  --env "PYTHONPATH=/sat_pyuser/lib/python3.8/site-packages:/home/nibio/mutable-outside-world" \
  "$IMAGE" \
  python3 -c '
import numpy
import pandas
import scipy
import sklearn
import torch
import torch_geometric
from sklearn.neighbors import KDTree
import MinkowskiEngine
import torch_points3d
import eval
del KDTree, MinkowskiEngine, torch_points3d, eval
print("numpy", numpy.__version__)
print("pandas", pandas.__version__)
print("scipy", scipy.__version__)
print("sklearn", sklearn.__version__)
print("torch", torch.__version__)
print("torch_geometric", torch_geometric.__version__)
print("PACKAGE_PREFLIGHT_OK")
'
