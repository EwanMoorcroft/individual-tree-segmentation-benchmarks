#!/bin/bash -l
#SBATCH --job-name=for_evaluate
#SBATCH --partition=nodes
#SBATCH --time=01:00:00
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

PREDICTIONS="data/predictions/tls2trees/for_instance/CULS/plot_1_annotated"
REFERENCE="$HOME/data/datasets/for_instance/FORinstance_dataset/CULS/plot_1_annotated.las"
METADATA="results/metadata/for_instance_tls2trees/evaluations/CULS/plot_1_annotated_metrics.json"
TABLE_ROOT="results/tables/for_instance_tls2trees/CULS"

mkdir -p "$(dirname "$METADATA")" "$TABLE_ROOT"

python scripts/evaluation/instance_iou_f1.py \
  --plot-name CULS_plot_1_annotated \
  --predicted-instance-dir "$PREDICTIONS" \
  --reference-labelled-point-cloud "$REFERENCE" \
  --reference-label-field treeID \
  --reference-semantic-field classification \
  --reference-classes 4 6 \
  --ignored-reference-classes 0 1 2 3 5 \
  --ignore-reference-labels 0,-1 \
  --iou-threshold 0.5 \
  --coordinate-tolerance 0.02 \
  --output-json "$METADATA" \
  --output-metrics-csv "$TABLE_ROOT/plot_1_annotated_metrics.csv" \
  --output-matches-csv "$TABLE_ROOT/plot_1_annotated_matches.csv" \
  --output-unmatched-predictions-csv "$TABLE_ROOT/plot_1_annotated_unmatched_predictions.csv" \
  --output-unmatched-references-csv "$TABLE_ROOT/plot_1_annotated_unmatched_references.csv"
