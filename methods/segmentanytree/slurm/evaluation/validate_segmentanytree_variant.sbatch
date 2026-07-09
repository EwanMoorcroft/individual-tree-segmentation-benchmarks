#!/bin/bash -l
#SBATCH --job-name=for_sat_variant_gate
#SBATCH --partition=nodes
#SBATCH --time=00:10:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --output=logs/segmentanytree_for_instance/%x_%j.out
#SBATCH --error=logs/segmentanytree_for_instance/%x_%j.err

set -euo pipefail

PROJECT_ROOT="$HOME/scratch/tree-seg-benchmark"
LABEL="${SEGMENTANYTREE_VARIANT_LABEL:?Set SEGMENTANYTREE_VARIANT_LABEL.}"
METRICS_ROOT="${SEGMENTANYTREE_VARIANT_METRICS_ROOT:?Set SEGMENTANYTREE_VARIANT_METRICS_ROOT.}"
EXPECTED_PLOTS="${SEGMENTANYTREE_VARIANT_EXPECTED_PLOTS:?Set SEGMENTANYTREE_VARIANT_EXPECTED_PLOTS.}"
EXPECTED_SPLIT="${SEGMENTANYTREE_VARIANT_EXPECTED_SPLIT:-}"
SUMMARY="${SEGMENTANYTREE_VARIANT_SUMMARY:?Set SEGMENTANYTREE_VARIANT_SUMMARY.}"

cd "$PROJECT_ROOT"
module purge
module load miniforge3/25.3.0-python3.12.10
source "$HOME/fastscratch/venvs/treebench/bin/activate"

args=(
  --variant "$LABEL=$METRICS_ROOT"
  --expected-plots "$EXPECTED_PLOTS"
  --output "$SUMMARY"
)
if [[ -n "$EXPECTED_SPLIT" ]]; then
  args+=(--expected-split "$EXPECTED_SPLIT")
fi
if [[ "${SEGMENTANYTREE_VARIANT_REQUIRE_PREDICTIONS:-0}" == "1" ]]; then
  args+=(--require-any-predictions)
fi
python methods/segmentanytree/scripts/evaluation/summarise_segmentanytree_variants.py "${args[@]}"
