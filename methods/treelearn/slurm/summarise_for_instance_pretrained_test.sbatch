#!/bin/bash -l
#SBATCH --job-name=treelearn_pre_sum
#SBATCH --partition=nodes
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=logs/treelearn_for_instance/treelearn_pre_test_sum_%j.out
#SBATCH --error=logs/treelearn_for_instance/treelearn_pre_test_sum_%j.err
set -euo pipefail
cd "$HOME/scratch/tree-seg-benchmark"
test "${TREELEARN_PRETRAINED_TEST_CONFIRMED:-0}" = 1
test "$(git rev-parse HEAD)" = "$TREELEARN_TEST_BENCHMARK_COMMIT"
test -z "$(git status --porcelain)"
source "$HOME/fastscratch/venvs/treebench/bin/activate"
python methods/treelearn/scripts/summarise_for_instance_finetuned_test.py \
  --manifest "$TREELEARN_TEST_FREEZE" \
  --run-id "$TREELEARN_TEST_RUN_ID" \
  --evaluation-root "$TREELEARN_TEST_TABLE_ROOT/per_plot" \
  --metadata-root "$TREELEARN_TEST_METADATA_ROOT" \
  --output-root "$TREELEARN_TEST_TABLE_ROOT" \
  --expected-benchmark-commit "$TREELEARN_TEST_BENCHMARK_COMMIT"
