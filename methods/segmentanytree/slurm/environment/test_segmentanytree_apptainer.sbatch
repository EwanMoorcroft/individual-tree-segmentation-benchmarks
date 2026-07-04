#!/bin/bash -l
#SBATCH --job-name=for_sat_container
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
test -f "$IMAGE"
nvidia-smi
apptainer exec --nv "$IMAGE" bash -lc '
cd /home/nibio/mutable-outside-world
python3 --version
test -f run_inference.sh
python3 -c "import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))"
'
