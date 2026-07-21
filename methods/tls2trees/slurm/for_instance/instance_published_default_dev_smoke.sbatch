#!/bin/bash -l
#SBATCH --job-name=tls2trees_smoke_instance
#SBATCH --partition=nodes
#SBATCH --time=08:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --output=logs/tls2trees_for_instance/%x_%j.out
#SBATCH --error=logs/tls2trees_for_instance/%x_%j.err

set -euo pipefail

cd "${TLS2TREES_PROJECT_ROOT:-$HOME/scratch/tree-seg-benchmark}"
# shellcheck disable=SC1091
source methods/tls2trees/slurm/for_instance/published_default_dev_smoke_common.sh
tls2trees_load_contract
tls2trees_validate_contract
tls2trees_load_method_env
tls2trees_resolve_stage0_context
tls2trees_stage_banner instance

tls2trees_require_file "$TLS2TREES_INSTANCE_CLI" "instance runner"
tls2trees_require_file "$PLOT_ROOT/metadata/semantic_run.json" "semantic metadata"
python "$TLS2TREES_ENV_VALIDATOR" \
  --tls2trees-repo "$TLS2TREES_REPO" \
  --setup-marker-json "$TLS2TREES_METHOD_ENV_MARKER"
INSTANCE_RECOVERY="${TLS2TREES_INSTANCE_RECOVERY_CONFIRMED:-0}"
RECOVERY_ARGUMENTS=()
if [[ "$INSTANCE_RECOVERY" == "1" ]]; then
  tls2trees_require_file "$PLOT_ROOT/metadata/instance_run.json" "failed instance metadata"
  tls2trees_require_dir "$PLOT_ROOT/predictions/raw" "failed raw prediction root"
  RECOVERY_ARGUMENTS+=(--resume-failed-empty-output)
elif [[ "$INSTANCE_RECOVERY" != "0" ]]; then
  tls2trees_die "TLS2TREES_INSTANCE_RECOVERY_CONFIRMED must be 0 or 1"
elif [[ -e "$PLOT_ROOT/predictions" || -e "$PLOT_ROOT/metadata/instance_run.json" ]]; then
  tls2trees_die "refusing existing instance output under $PLOT_ROOT"
fi

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
python "$TLS2TREES_INSTANCE_CLI" \
  --manifest-json "$MANIFEST_JSON" \
  --task-index "$TASK_INDEX" \
  --output-root "$OUTPUT_ROOT" \
  --run-id "$RUN_ID" \
  --tls2trees-repo "$TLS2TREES_REPO" \
  --config "$TLS2TREES_METHOD_CONFIG" \
  --variant published_default \
  --split development \
  "${RECOVERY_ARGUMENTS[@]}"

test -f "$PLOT_ROOT/metadata/instance_run.json"
test -d "$PLOT_ROOT/predictions/raw"
echo "plot_root=$PLOT_ROOT"
echo "memory_request=96G"
echo "expected_runtime=30 minutes to 8 hours; calibrate from this smoke"
echo "success_criteria=completed instance metadata and explicit raw leaf-off and leaf-on files"
echo "failure_indicators=OOM, missing stems, graph failure, missing target pattern"
echo "held_out_test_accessed=false"
echo "finished_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
