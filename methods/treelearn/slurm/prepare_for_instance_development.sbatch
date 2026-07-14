#!/bin/bash -l
#SBATCH --job-name=treelearn_dev_prep
#SBATCH --partition=nodes
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=logs/treelearn_for_instance/treelearn_dev_prep_%j.out
#SBATCH --error=logs/treelearn_for_instance/treelearn_dev_prep_%j.err

set -euo pipefail

cd "$HOME/scratch/tree-seg-benchmark"
mkdir -p logs/treelearn_for_instance

if [[ "${TREELEARN_FULL_DEV_CONFIRMED:-0}" != "1" \
      || "${TREELEARN_DEV_ALIGNMENT_REVIEW_CONFIRMED:-0}" != "1" ]]; then
  echo "Refusing TreeLearn development preparation without both confirmations."
  exit 2
fi

module purge
module load miniforge3/25.3.0-python3.12.10
source "$HOME/fastscratch/venvs/treebench/bin/activate"

TREELEARN_DEV_RUN_ID="${TREELEARN_DEV_RUN_ID:?TREELEARN_DEV_RUN_ID is required}"
TREELEARN_DATASET_ROOT="${TREELEARN_DATASET_ROOT:-$HOME/data/datasets/for_instance/FORinstance_dataset}"
TREELEARN_ACCEPTED_SMOKE_RUN_ID="${TREELEARN_ACCEPTED_SMOKE_RUN_ID:?TREELEARN_ACCEPTED_SMOKE_RUN_ID is required}"
TREELEARN_EXPECTED_BENCHMARK_COMMIT="${TREELEARN_EXPECTED_BENCHMARK_COMMIT:?TREELEARN_EXPECTED_BENCHMARK_COMMIT is required}"
TREELEARN_DEV_ACCEPTANCE="${TREELEARN_DEV_ACCEPTANCE:?TREELEARN_DEV_ACCEPTANCE is required}"
TREELEARN_DEV_MANIFEST="${TREELEARN_DEV_MANIFEST:?TREELEARN_DEV_MANIFEST is required}"
TREELEARN_DEV_MANIFEST_JSON="${TREELEARN_DEV_MANIFEST_JSON:?TREELEARN_DEV_MANIFEST_JSON is required}"

EXPECTED_SMOKE_RUN_ID="treelearn_for-instance_published_pretrained_dev_smoke_20260712_135205"
if [[ "$TREELEARN_ACCEPTED_SMOKE_RUN_ID" != "$EXPECTED_SMOKE_RUN_ID" ]]; then
  echo "Accepted smoke run differs from the frozen development contract."
  exit 2
fi

test "$(git rev-parse HEAD)" = "$TREELEARN_EXPECTED_BENCHMARK_COMMIT"
test -z "$(git status --porcelain)"
test -d "$TREELEARN_DATASET_ROOT"
test -f "$TREELEARN_DATASET_ROOT/data_split_metadata.csv"

SMOKE_METADATA_ROOT="results/metadata/treelearn_for_instance/one_plot_smokes/$TREELEARN_ACCEPTED_SMOKE_RUN_ID"
SMOKE_TABLE_ROOT="results/tables/treelearn_for_instance/one_plot_smokes/$TREELEARN_ACCEPTED_SMOKE_RUN_ID"
SMOKE_INFERENCE="$SMOKE_METADATA_ROOT/CULS_plot_1_annotated_inference.json"
SMOKE_METRICS="$SMOKE_TABLE_ROOT/metrics.json"
test -f "$SMOKE_INFERENCE"
test -f "$SMOKE_METRICS"

python methods/treelearn/scripts/prepare_for_instance_development_acceptance.py \
  --config methods/treelearn/configs/for_instance_development.yml \
  --smoke-inference "$SMOKE_INFERENCE" \
  --smoke-metrics "$SMOKE_METRICS" \
  --output "$TREELEARN_DEV_ACCEPTANCE" \
  --manual-alignment-review-confirmed

python methods/treelearn/scripts/prepare_for_instance_development_manifest.py \
  --dataset-root "$TREELEARN_DATASET_ROOT" \
  --metadata-csv "$TREELEARN_DATASET_ROOT/data_split_metadata.csv" \
  --output-csv "$TREELEARN_DEV_MANIFEST" \
  --output-json "$TREELEARN_DEV_MANIFEST_JSON"

python -c 'import csv,sys; rows=list(csv.DictReader(open(sys.argv[1], encoding="utf-8"))); assert len(rows) == 21; assert all(row["split"] == "dev" for row in rows)' "$TREELEARN_DEV_MANIFEST"

echo "run_id=$TREELEARN_DEV_RUN_ID"
echo "accepted_smoke=$TREELEARN_ACCEPTED_SMOKE_RUN_ID"
echo "development_manifest=$TREELEARN_DEV_MANIFEST"
echo "status=treelearn-development-prepared"
echo "held_out_test_accessed=false"
