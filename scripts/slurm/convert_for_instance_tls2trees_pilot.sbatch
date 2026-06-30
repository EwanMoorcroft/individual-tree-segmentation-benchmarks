#!/bin/bash -l
#SBATCH --job-name=for_convert
#SBATCH --partition=nodes
#SBATCH --time=00:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --output=logs/for_instance_tls2trees/%x_%j.out
#SBATCH --error=logs/for_instance_tls2trees/%x_%j.err

set -euo pipefail

module purge
module load miniforge3/25.3.0-python3.12.10
source ~/fastscratch/venvs/treebench/bin/activate

cd "$HOME/scratch/tree-seg-benchmark"

INPUT="$HOME/data/datasets/for_instance/FORinstance_dataset/CULS/plot_1_annotated.las"
OUTPUT="data/interim/tls2trees/for_instance/CULS/plot_1_annotated"
METADATA="results/metadata/for_instance_tls2trees/conversions/CULS/plot_1_annotated_conversion.json"

mkdir -p "$OUTPUT" "$(dirname "$METADATA")"

python scripts/data/convert_for_instance_to_tls2trees_ply.py \
  --input-las "$INPUT" \
  --output-dir "$OUTPUT" \
  --tile-name 001 \
  --reference-classes 4 6 \
  --metadata-output "$METADATA"
