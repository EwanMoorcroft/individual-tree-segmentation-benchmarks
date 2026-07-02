#!/bin/bash -l
#SBATCH --job-name=for_sat_stack_test
#SBATCH --partition=gpu-l40s
#SBATCH --gres=gpu:1
#SBATCH --time=00:15:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --output=logs/segmentanytree_for_instance/%x_%j.out
#SBATCH --error=logs/segmentanytree_for_instance/%x_%j.err

set -euo pipefail

cd "$HOME/scratch/tree-seg-benchmark"
module purge
module load apptainer/1.3.6
mkdir -p logs/segmentanytree_for_instance

IMAGE="$HOME/scratch/containers/segment-any-tree_latest.sif"
USERBASE="$HOME/fastscratch/segmentanytree_pyuser_v1"

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
assert numpy.__version__ == "1.24.4"
assert pandas.__version__ == "1.5.3"
assert scipy.__version__ == "1.10.1"
assert sklearn.__version__ == "1.3.2"
assert torch.cuda.is_available()
print("PACKAGE_PREFLIGHT_OK")
'
