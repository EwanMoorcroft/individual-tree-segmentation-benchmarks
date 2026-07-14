#!/bin/bash -l
#SBATCH --job-name=treelearn_test_gate
#SBATCH --partition=nodes
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --output=logs/treelearn_for_instance/treelearn_test_gate_%j.out
#SBATCH --error=logs/treelearn_for_instance/treelearn_test_gate_%j.err
set -euo pipefail
cd "$HOME/scratch/tree-seg-benchmark"
test "${TREELEARN_FINETUNED_TEST_CONFIRMED:-0}" = 1
test "$(git rev-parse HEAD)" = "$TREELEARN_TEST_BENCHMARK_COMMIT"
test -z "$(git status --porcelain)"
source "$HOME/fastscratch/venvs/treebench/bin/activate"
python methods/treelearn/scripts/validate_for_instance_finetuned_test.py \
  --test-freeze "$TREELEARN_TEST_FREEZE" \
  --run-summary "$TREELEARN_TEST_RUN_SUMMARY" \
  --final-summary "$TREELEARN_TEST_FINAL_SUMMARY" \
  --retention-manifest "$TREELEARN_TEST_RETENTION_MANIFEST" \
  --output "$TREELEARN_TEST_COMPLETION_GATE" \
  --expected-benchmark-commit "$TREELEARN_TEST_BENCHMARK_COMMIT"
