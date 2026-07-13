#!/bin/bash -l
#SBATCH --job-name=treelearn_test_prep
#SBATCH --partition=nodes
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=logs/treelearn_for_instance/treelearn_test_prep_%j.out
#SBATCH --error=logs/treelearn_for_instance/treelearn_test_prep_%j.err
set -euo pipefail
cd "$HOME/scratch/tree-seg-benchmark"
test "${TREELEARN_FINETUNED_TEST_CONFIRMED:-0}" = 1
test "$(git rev-parse HEAD)" = "$TREELEARN_TEST_BENCHMARK_COMMIT"
test -z "$(git status --porcelain)"
source "$HOME/fastscratch/venvs/treebench/bin/activate"
python methods/treelearn/scripts/prepare_for_instance_finetuned_test.py \
  --selected-freeze "$TREELEARN_TEST_SELECTED_FREEZE" \
  --dataset-root "$TREELEARN_DATASET_ROOT" \
  --metadata-csv "$TREELEARN_DATASET_ROOT/data_split_metadata.csv" \
  --expected-run-id "$TREELEARN_TEST_RUN_ID" \
  --output-json "$TREELEARN_TEST_FREEZE" \
  --output-csv "$TREELEARN_TEST_MANIFEST"
