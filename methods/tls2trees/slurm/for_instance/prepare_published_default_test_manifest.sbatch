#!/bin/bash -l
#SBATCH --job-name=tls2trees_pd_test_manifest
#SBATCH --partition=nodes
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --no-requeue
#SBATCH --output=logs/tls2trees_for_instance/%x_%j.out
#SBATCH --error=logs/tls2trees_for_instance/%x_%j.err

set -euo pipefail
cd "${TLS2TREES_PD_TEST_PROJECT_ROOT:-$HOME/scratch/tree-seg-benchmark}"
# shellcheck disable=SC1091
source methods/tls2trees/slurm/for_instance/published_default_test_common.sh
pd_test_load_contract
test ! -e "$TLS2TREES_PD_TEST_MANIFEST_JSON"
test ! -e "$TLS2TREES_PD_TEST_MANIFEST_CSV"
test ! -e "$TLS2TREES_PD_TEST_MANIFEST_SHA256_FILE"
pd_test_load_treebench

python - "$TLS2TREES_PD_TEST_WORKFLOW_CONFIG" "$TLS2TREES_PD_TEST_PUBLISHED_CONFIG" <<'PY'
import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd() / "methods/tls2trees/scripts/runtime"))
from published_default_test_common import validate_frozen_configuration

validate_frozen_configuration(sys.argv[1], sys.argv[2])
PY

mkdir -p "$(dirname "$TLS2TREES_PD_TEST_MANIFEST_JSON")"
if [[ "$TLS2TREES_PD_TEST_CACHE_AVAILABLE" == "1" ]]; then
  pd_test_require_file "$TLS2TREES_PD_TEST_CACHE_STATE_FILE" "cache state"
  [[ "$(sha256sum "$TLS2TREES_PD_TEST_CACHE_STATE_FILE" | awk '{print $1}')" == "$TLS2TREES_PD_TEST_CACHE_STATE_SHA256" ]] || \
    pd_test_die "cache state changed"
  pd_test_require_file "$TLS2TREES_PD_TEST_CACHE_MANIFEST_JSON" "cache manifest JSON"
  pd_test_require_file "$TLS2TREES_PD_TEST_CACHE_MANIFEST_CSV" "cache manifest CSV"
  [[ "$(sha256sum "$TLS2TREES_PD_TEST_CACHE_MANIFEST_JSON" | awk '{print $1}')" == "$TLS2TREES_PD_TEST_CACHE_MANIFEST_SHA256" ]] || \
    pd_test_die "cache manifest changed"
  python "$TLS2TREES_PD_TEST_MANIFEST_CLI" validate \
    --manifest-json "$TLS2TREES_PD_TEST_CACHE_MANIFEST_JSON" \
    --expected-split test --allow-held-out-test >/dev/null
  cp "$TLS2TREES_PD_TEST_CACHE_MANIFEST_JSON" "$TLS2TREES_PD_TEST_MANIFEST_JSON"
  cp "$TLS2TREES_PD_TEST_CACHE_MANIFEST_CSV" "$TLS2TREES_PD_TEST_MANIFEST_CSV"
  MANIFEST_SOURCE=verified_semantic_cache
else
  python "$TLS2TREES_PD_TEST_MANIFEST_CLI" build \
    --dataset-root "$TLS2TREES_PD_TEST_DATASET_ROOT" \
    --metadata-csv "$TLS2TREES_PD_TEST_SPLIT_METADATA_CSV" \
    --split test --allow-held-out-test \
    --output-json "$TLS2TREES_PD_TEST_MANIFEST_JSON" \
    --output-csv "$TLS2TREES_PD_TEST_MANIFEST_CSV"
  MANIFEST_SOURCE=dataset_metadata
fi

python "$TLS2TREES_PD_TEST_MANIFEST_CLI" validate \
  --manifest-json "$TLS2TREES_PD_TEST_MANIFEST_JSON" \
  --expected-split test --allow-held-out-test >/dev/null
python - "$TLS2TREES_PD_TEST_MANIFEST_JSON" "$TLS2TREES_PD_TEST_WORKFLOW_CONFIG" <<'PY'
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd() / "methods/tls2trees/scripts/runtime"))
from published_default_test_common import load_yaml, validate_exact_manifest

manifest = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
workflow, _ = load_yaml(sys.argv[2])
validate_exact_manifest(manifest, workflow)
PY
sha256sum "$TLS2TREES_PD_TEST_MANIFEST_JSON" > "$TLS2TREES_PD_TEST_MANIFEST_SHA256_FILE"

echo "status=published_default_test_manifest_prepared"
echo "manifest_source=$MANIFEST_SOURCE"
echo "plots=11"
echo "points=49709922"
echo "reference_trees=323"
echo "manifest_sha256=$(awk '{print $1}' "$TLS2TREES_PD_TEST_MANIFEST_SHA256_FILE")"
echo "held_out_test_accessed=true"
