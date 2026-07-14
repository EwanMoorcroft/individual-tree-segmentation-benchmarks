#!/bin/bash -l
#SBATCH --job-name=treelearn_pre_recover_prep
#SBATCH --partition=nodes
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --output=logs/treelearn_for_instance/treelearn_pre_recover_prep_%j.out
#SBATCH --error=logs/treelearn_for_instance/treelearn_pre_recover_prep_%j.err
set -euo pipefail
cd "$HOME/scratch/tree-seg-benchmark"
test "${TREELEARN_PRETRAINED_TEST_RECOVERY_CONFIRMED:-0}" = 1
test "$(git rev-parse HEAD)" = "$TREELEARN_TEST_BENCHMARK_COMMIT"
test -z "$(git status --porcelain)"
source "$HOME/fastscratch/venvs/treebench/bin/activate"
PRIOR_ARGS=()
if [[ -n "${TREELEARN_TEST_PRIOR_RECOVERY_MANIFEST:-}" ]]; then
  PRIOR_ARGS=(--prior-recovery-manifest "$TREELEARN_TEST_PRIOR_RECOVERY_MANIFEST")
fi
python methods/treelearn/scripts/prepare_for_instance_pretrained_test_recovery.py \
  --manifest "$TREELEARN_TEST_FREEZE" \
  --runtime-root "$TREELEARN_TEST_RUNTIME_ROOT" \
  --prediction-root "$TREELEARN_TEST_PREDICTION_ROOT" \
  --metadata-root "$TREELEARN_TEST_METADATA_ROOT" \
  --table-root "$TREELEARN_TEST_TABLE_ROOT" \
  --archive-root "$TREELEARN_TEST_RECOVERY_ARCHIVE" \
  --output "$TREELEARN_TEST_RECOVERY_MANIFEST" \
  --original-benchmark-commit "$TREELEARN_TEST_ORIGINAL_BENCHMARK_COMMIT" \
  --recovery-benchmark-commit "$TREELEARN_TEST_BENCHMARK_COMMIT" \
  "${PRIOR_ARGS[@]}"
