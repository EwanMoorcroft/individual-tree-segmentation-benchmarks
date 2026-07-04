#!/bin/bash -l
#SBATCH --job-name=for_tls2trees
#SBATCH --partition=nodes
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --output=logs/for_instance_tls2trees/%x_%j.out
#SBATCH --error=logs/for_instance_tls2trees/%x_%j.err

set -euo pipefail

module purge
module load miniforge3/25.3.0-python3.12.10
source ~/fastscratch/venvs/treebench/bin/activate

cd "$HOME/scratch/tree-seg-benchmark"
mkdir -p \
  data/predictions/tls2trees/for_instance/CULS \
  results/metadata/for_instance_tls2trees/runs/CULS

python methods/tls2trees/scripts/runtime/run_tls2trees_for_instance_plot.py \
  --config methods/tls2trees/configs/for_instance_accuracy.yml \
  --plot-path CULS/plot_1_annotated.las \
  --collection CULS \
  --plot-name plot_1_annotated
