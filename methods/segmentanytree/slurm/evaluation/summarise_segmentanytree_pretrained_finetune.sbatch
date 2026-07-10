#!/bin/bash -l
#SBATCH --job-name=for_sat_pre_ft_summary
#SBATCH --partition=nodes
#SBATCH --time=00:10:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --output=logs/segmentanytree_for_instance/%x_%j.out
#SBATCH --error=logs/segmentanytree_for_instance/%x_%j.err

set -euo pipefail

PROJECT_ROOT="${SEGMENTANYTREE_PROJECT_ROOT:-$HOME/scratch/tree-seg-benchmark}"
PRETRAINED="${SEGMENTANYTREE_PRETRAINED_METRICS_ROOT:?Set pretrained metrics root.}"
FINETUNED="${SEGMENTANYTREE_FINETUNED_METRICS_ROOT:?Set fine-tuned metrics root.}"
OUTPUT="${SEGMENTANYTREE_PRETRAINED_FINETUNE_SUMMARY:?Set summary output.}"

cd "$PROJECT_ROOT"
module purge
module load miniforge3/25.3.0-python3.12.10
source "$HOME/fastscratch/venvs/treebench/bin/activate"
python methods/segmentanytree/scripts/evaluation/summarise_segmentanytree_variants.py \
  --variant "published_pretrained=$PRETRAINED" \
  --variant "fine_tuned_on_dev=$FINETUNED" \
  --expected-plots 11 \
  --expected-split test \
  --require-any-predictions \
  --output "$OUTPUT"
