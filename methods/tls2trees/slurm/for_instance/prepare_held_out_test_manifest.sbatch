#!/bin/bash -l
#SBATCH --job-name=tls2trees_dt_test_inventory
#SBATCH --partition=nodes
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --no-requeue
#SBATCH --output=logs/tls2trees_for_instance/%x_%j.out
#SBATCH --error=logs/tls2trees_for_instance/%x_%j.err

set -euo pipefail
cd "${TLS2TREES_PROJECT_ROOT:-$HOME/scratch/tree-seg-benchmark}"
test "${TLS2TREES_HELD_OUT_TEST_CONFIRMED:-0}" = "1"
test "$(git rev-parse HEAD)" = "$TLS2TREES_EXPECTED_BENCHMARK_COMMIT"
test -z "$(git status --porcelain)"
test "$(sha256sum "$TLS2TREES_FINAL_SELECTION_JSON" | awk '{print $1}')" = "$TLS2TREES_FINAL_SELECTION_SHA256"
test ! -e "$TLS2TREES_MANIFEST_JSON"
test ! -e "$TLS2TREES_MANIFEST_CSV"
test ! -e "$TLS2TREES_MANIFEST_SHA256_FILE"

module purge
module load miniforge3/25.3.0-python3.12.10
# shellcheck disable=SC1090
source "$TLS2TREES_TREEBENCH_ENV/bin/activate"
export PYTHONNOUSERSITE=1
python "$TLS2TREES_MANIFEST_CLI" build \
  --dataset-root "$TLS2TREES_DATASET_ROOT" \
  --metadata-csv "$TLS2TREES_SPLIT_METADATA_CSV" \
  --split test --allow-held-out-test \
  --output-json "$TLS2TREES_MANIFEST_JSON" \
  --output-csv "$TLS2TREES_MANIFEST_CSV"
python "$TLS2TREES_MANIFEST_CLI" validate \
  --manifest-json "$TLS2TREES_MANIFEST_JSON" \
  --expected-split test --allow-held-out-test >/dev/null
python -c '
import json,sys
p=json.load(open(sys.argv[1])); assert len(p["plots"]) == 11
assert sum(int(r["point_count"]) for r in p["plots"]) == 49709922
assert sum(int(r["reference_tree_count"]) for r in p["plots"]) == 323
' "$TLS2TREES_MANIFEST_JSON"
sha256sum "$TLS2TREES_MANIFEST_JSON" > "$TLS2TREES_MANIFEST_SHA256_FILE"
echo "status=held_out_test_manifest_prepared"
echo "plots=11"
echo "points=49709922"
echo "reference_trees=323"
echo "manifest_sha256=$(awk '{print $1}' "$TLS2TREES_MANIFEST_SHA256_FILE")"
echo "held_out_test_accessed=true"
