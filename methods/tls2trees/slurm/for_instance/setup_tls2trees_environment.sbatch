#!/bin/bash -l
#SBATCH --job-name=tls2trees_env_setup
#SBATCH --partition=gpu-l40s-low
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --no-requeue
#SBATCH --output=logs/tls2trees_for_instance/%x_%j.out
#SBATCH --error=logs/tls2trees_for_instance/%x_%j.err

set -euo pipefail

if [[ "${TLS2TREES_SETUP_CONFIRMED:-0}" != "1" ]]; then
  echo "Refusing TLS2trees environment setup." >&2
  echo "Set TLS2TREES_SETUP_CONFIRMED=1 after reviewing the pinned setup." >&2
  exit 2
fi

PROJECT_ROOT="${TLS2TREES_PROJECT_ROOT:-$HOME/scratch/tree-seg-benchmark}"
UPSTREAM_REPO="${TLS2TREES_UPSTREAM_REPO:-$PROJECT_ROOT/external/TLS2trees}"
METHOD_ENV="${TLS2TREES_METHOD_ENV:-$HOME/fastscratch/venvs/tls2trees}"
RESUME_PARTIAL="${TLS2TREES_SETUP_RESUME_PARTIAL:-0}"
VALIDATE_PARTIAL_ONLY="${TLS2TREES_SETUP_VALIDATE_PARTIAL_ONLY:-0}"
EXPECTED_COMMIT="ca12cb73b2c736d80b020e8025f8d975d42e6f01"
EXPECTED_MODEL_SHA256="1a8bb6372394600f7c4b15f76beb98c32cb47ed25f8f729a84117ccfa410e72b"
VALIDATOR="$PROJECT_ROOT/methods/tls2trees/scripts/runtime/validate_tls2trees_environment.py"
ENV_MARKER="$METHOD_ENV/.tls2trees_setup_complete.json"
CONDA_CHANNEL_RECORD="${METHOD_ENV}.tls2trees_conda_channel"
PYTORCH_WHEEL_INDEX="https://download.pytorch.org/whl/torch_stable.html"
PYG_WHEEL_INDEX="https://data.pyg.org/whl/torch-1.9.0+cu111.html"
CONDA_CUDA_RUNTIME="cudatoolkit=11.1.1=h6406543_8"
REQUESTED_CONDA_CHANNEL="${TLS2TREES_CONDA_CHANNEL:-}"
REQUESTED_CONDA_CHANNEL="${REQUESTED_CONDA_CHANNEL%/}"
CONDA_CHANNEL="${REQUESTED_CONDA_CHANNEL:-https://conda.anaconda.org/conda-forge}"
CONDA_CHANNEL="${CONDA_CHANNEL%/}"

tls2trees_validate_conda_channel() {
  local channel="$1"
  case "$channel" in
    https://conda.anaconda.org/conda-forge|https://prefix.dev/conda-forge)
      ;;
    *)
      echo "Unsupported TLS2trees Conda channel: $channel" >&2
      echo "Use an allow-listed HTTPS conda-forge source." >&2
      return 1
      ;;
  esac
}

tls2trees_validate_conda_channel "$CONDA_CHANNEL"

if [[ "$RESUME_PARTIAL" != "0" && "$RESUME_PARTIAL" != "1" ]]; then
  echo "TLS2TREES_SETUP_RESUME_PARTIAL must be 0 or 1." >&2
  exit 2
fi
if [[ "$VALIDATE_PARTIAL_ONLY" != "0" && "$VALIDATE_PARTIAL_ONLY" != "1" ]]; then
  echo "TLS2TREES_SETUP_VALIDATE_PARTIAL_ONLY must be 0 or 1." >&2
  exit 2
fi
if [[ "$VALIDATE_PARTIAL_ONLY" == "1" && "$RESUME_PARTIAL" != "1" ]]; then
  echo "Validation-only recovery requires TLS2TREES_SETUP_RESUME_PARTIAL=1." >&2
  exit 2
fi

tls2trees_load_recorded_conda_channel() {
  local recorded_channel
  if [[ ! -f "$CONDA_CHANNEL_RECORD" ]]; then
    echo "TLS2trees Conda channel record is missing: $CONDA_CHANNEL_RECORD" >&2
    return 1
  fi
  IFS= read -r recorded_channel < "$CONDA_CHANNEL_RECORD"
  recorded_channel="${recorded_channel%/}"
  tls2trees_validate_conda_channel "$recorded_channel"
  if [[ -n "$REQUESTED_CONDA_CHANNEL" && \
    "$REQUESTED_CONDA_CHANNEL" != "$recorded_channel" ]]; then
    echo "Requested TLS2trees Conda channel does not match the prefix record." >&2
    echo "requested=$REQUESTED_CONDA_CHANNEL" >&2
    echo "recorded=$recorded_channel" >&2
    return 1
  fi
  CONDA_CHANNEL="$recorded_channel"
}

tls2trees_preflight_conda_channel() {
  local subdir
  local repodata_url
  for subdir in linux-64 noarch; do
    repodata_url="$CONDA_CHANNEL/$subdir/repodata.json.zst"
    if ! curl --fail --silent --show-error --location --head \
      --connect-timeout 10 --max-time 30 "$repodata_url" >/dev/null; then
      echo "TLS2trees Conda channel preflight failed: $repodata_url" >&2
      return 1
    fi
  done
  echo "conda_channel_preflight=passed"
  echo "conda_channel=$CONDA_CHANNEL"
}

tls2trees_export_cuda_runtime() {
  local library_dir="$METHOD_ENV/lib"
  case ":${LD_LIBRARY_PATH:-}:" in
    *":$library_dir:"*)
      ;;
    *)
      export LD_LIBRARY_PATH="$library_dir${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
      ;;
  esac
}

cd "$PROJECT_ROOT"
mkdir -p logs/tls2trees_for_instance "$HOME/fastscratch/venvs"
test -d .git
test -d "$UPSTREAM_REPO/.git"
test -f "$VALIDATOR"
test "$(git -C "$UPSTREAM_REPO" rev-parse HEAD)" = "$EXPECTED_COMMIT"
test -z "$(git -C "$UPSTREAM_REPO" status --porcelain)"
test -z "$(git status --porcelain)"
BENCHMARK_COMMIT=$(git rev-parse HEAD)
echo "$EXPECTED_MODEL_SHA256  $UPSTREAM_REPO/tls2trees/fsct/model/model.pth" | \
  sha256sum --check --status

MIN_FREE_BYTES="${TLS2TREES_SETUP_MIN_FREE_BYTES:-21474836480}"
if [[ ! "$MIN_FREE_BYTES" =~ ^[1-9][0-9]*$ ]]; then
  echo "TLS2TREES_SETUP_MIN_FREE_BYTES must be a positive integer." >&2
  exit 2
fi
FREE_BYTES=$(df -PB1 "$HOME/fastscratch/venvs" | awk 'NR == 2 {print $4}')
if [[ ! "$FREE_BYTES" =~ ^[0-9]+$ ]] || ((FREE_BYTES < MIN_FREE_BYTES)); then
  echo "Need at least $MIN_FREE_BYTES free bytes under fastscratch; found ${FREE_BYTES:-unknown}." >&2
  exit 2
fi

module purge
module load miniforge3/25.3.0-python3.12.10
source "$(conda info --base)/etc/profile.d/conda.sh"
export CONDA_NUMBER_CHANNEL_NOTICES=0
export PYTHONNOUSERSITE=1
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-$HOME/fastscratch/pip-cache}"
mkdir -p "$PIP_CACHE_DIR"

PREFIX_MODIFIED=0
trap 'status=$?; rm -f "$ENV_MARKER.partial" "$CONDA_CHANNEL_RECORD.partial"; if [[ $status -ne 0 && "$PREFIX_MODIFIED" == "1" ]]; then rm -f "$ENV_MARKER"; fi' EXIT

INSTALL_ENV=0
CREATE_ENV=0
if [[ -f "$ENV_MARKER" && "$RESUME_PARTIAL" != "1" ]]; then
  test -x "$METHOD_ENV/bin/python"
  tls2trees_load_recorded_conda_channel
  echo "reusing_verified_tls2trees_env=$METHOD_ENV"
elif [[ -e "$METHOD_ENV" ]]; then
  if [[ ! -d "$METHOD_ENV/conda-meta" ]]; then
    echo "Existing TLS2trees path is not a Conda prefix: $METHOD_ENV" >&2
    echo "Choose a new TLS2TREES_METHOD_ENV path." >&2
    exit 2
  fi
  tls2trees_load_recorded_conda_channel
  if [[ "$RESUME_PARTIAL" != "1" ]]; then
    echo "TLS2trees environment exists without a valid setup marker." >&2
    echo "Inspect the prior setup log, then set TLS2TREES_SETUP_RESUME_PARTIAL=1 to resume." >&2
    exit 2
  fi
  if [[ "$VALIDATE_PARTIAL_ONLY" == "1" ]]; then
    test -x "$METHOD_ENV/bin/python"
    echo "validating_existing_partial_tls2trees_env=$METHOD_ENV"
    echo "network_install_steps=skipped"
  else
    INSTALL_ENV=1
    echo "resuming_partial_tls2trees_env=$METHOD_ENV"
  fi
else
  if [[ "$VALIDATE_PARTIAL_ONLY" == "1" ]]; then
    echo "Validation-only recovery requires an existing partial Conda prefix." >&2
    exit 2
  fi
  if [[ -f "$CONDA_CHANNEL_RECORD" ]]; then
    tls2trees_load_recorded_conda_channel
  fi
  INSTALL_ENV=1
  CREATE_ENV=1
fi

if [[ "$INSTALL_ENV" == "1" ]]; then
  # Both a new setup and a reviewed partial resume need the pinned CUDA runtime.
  tls2trees_preflight_conda_channel
  if [[ "$CREATE_ENV" == "1" ]]; then
    if [[ ! -f "$CONDA_CHANNEL_RECORD" ]]; then
      printf '%s\n' "$CONDA_CHANNEL" > "$CONDA_CHANNEL_RECORD.partial"
      mv "$CONDA_CHANNEL_RECORD.partial" "$CONDA_CHANNEL_RECORD"
    fi
  fi
  PREFIX_MODIFIED=1
  if [[ "$CREATE_ENV" == "1" ]]; then
    conda create \
      --override-channels \
      --channel "$CONDA_CHANNEL" \
      -p "$METHOD_ENV" \
      python=3.9 pip "$CONDA_CUDA_RUNTIME" -y
  else
    conda install \
      --override-channels \
      --channel "$CONDA_CHANNEL" \
      --freeze-installed \
      -p "$METHOD_ENV" \
      "$CONDA_CUDA_RUNTIME" -y
  fi

  tls2trees_export_cuda_runtime

  "$METHOD_ENV/bin/python" -m pip install --disable-pip-version-check \
    "pip==23.3.2" \
    "setuptools==68.2.2" \
    "wheel==0.41.3"

  "$METHOD_ENV/bin/python" -m pip install --disable-pip-version-check \
    "numpy==1.21.1" \
    "pandas==1.2.5" \
    "scipy==1.7.1" \
    "scikit-learn==0.24.2" \
    "networkx==2.6.2" \
    "matplotlib==3.4.3" \
    "tqdm==4.62.1" \
    "laspy==2.0.2" \
    "PyYAML==6.0.1" \
    "requests==2.26.0" \
    "Jinja2==3.0.1" \
    "MarkupSafe==2.0.1" \
    "googledrivedownloader==0.4" \
    "rdflib==6.0.0" \
    "python-louvain==0.15"

  "$METHOD_ENV/bin/python" -m pip install --disable-pip-version-check \
    "torch==1.9.0+cu111" \
    --find-links "$PYTORCH_WHEEL_INDEX"

  "$METHOD_ENV/bin/python" -m pip install --disable-pip-version-check \
    --no-index \
    --find-links "$PYG_WHEEL_INDEX" \
    "torch-cluster==1.5.9" \
    "torch-scatter==2.0.8" \
    "torch-sparse==0.6.11" \
    "torch-spline-conv==1.2.1"

  "$METHOD_ENV/bin/python" -m pip install --disable-pip-version-check \
    --no-deps \
    "torch-geometric==1.7.2"
fi

tls2trees_export_cuda_runtime
"$METHOD_ENV/bin/python" -m pip check

SETUP_RECORD="$PROJECT_ROOT/results/metadata/tls2trees/for_instance/environment_setups/${SLURM_JOB_ID:-manual}"
if [[ -e "$SETUP_RECORD" ]]; then
  echo "Refusing existing setup record: $SETUP_RECORD" >&2
  exit 2
fi
mkdir -p "$SETUP_RECORD"
VALIDATION_JSON="$SETUP_RECORD/validation.json"

"$METHOD_ENV/bin/python" "$VALIDATOR" \
  --tls2trees-repo "$UPSTREAM_REPO" \
  --require-cuda \
  --output-json "$VALIDATION_JSON"

TLS2TREES_REPO="$UPSTREAM_REPO" \
PYTHONPATH="$UPSTREAM_REPO:$UPSTREAM_REPO/tls2trees" \
PYTHONNOUSERSITE=1 \
  "$METHOD_ENV/bin/python" \
  methods/tls2trees/scripts/runtime/patches/semantic_patched.py --help >/dev/null
TLS2TREES_REPO="$UPSTREAM_REPO" \
PYTHONPATH="$UPSTREAM_REPO:$UPSTREAM_REPO/tls2trees" \
PYTHONNOUSERSITE=1 \
  "$METHOD_ENV/bin/python" \
  methods/tls2trees/scripts/runtime/patches/instance_patched.py --help >/dev/null

conda list -p "$METHOD_ENV" --explicit > "$SETUP_RECORD/conda-explicit.txt"
printf '%s\n' "$CONDA_CHANNEL" > "$SETUP_RECORD/conda-channel.txt"
printf '%s\n' "$CONDA_CUDA_RUNTIME" > "$SETUP_RECORD/cuda-runtime.txt"
"$METHOD_ENV/bin/python" -m pip freeze > "$SETUP_RECORD/pip-freeze.txt"
printf '%s\n' "$BENCHMARK_COMMIT" > "$SETUP_RECORD/benchmark-commit.txt"
git -C "$UPSTREAM_REPO" rev-parse HEAD > "$SETUP_RECORD/tls2trees-commit.txt"
sha256sum "$UPSTREAM_REPO/tls2trees/fsct/model/model.pth" > \
  "$SETUP_RECORD/model-sha256.txt"
git -C "$UPSTREAM_REPO" status --short > "$SETUP_RECORD/tls2trees-git-status.txt"

cp "$VALIDATION_JSON" "$ENV_MARKER.partial"
mv "$ENV_MARKER.partial" "$ENV_MARKER"
test -f "$ENV_MARKER"
test -z "$(git status --porcelain)"

echo "status=TLS2TREES_ENVIRONMENT_SETUP_VALIDATED"
echo "tls2trees_env=$METHOD_ENV"
echo "environment_marker=$ENV_MARKER"
echo "environment_marker_sha256=$(sha256sum "$ENV_MARKER" | awk '{print $1}')"
echo "setup_record=$SETUP_RECORD"
echo "conda_channel=$CONDA_CHANNEL"
echo "cuda_runtime=$CONDA_CUDA_RUNTIME"
echo "benchmark_commit=$BENCHMARK_COMMIT"
echo "upstream_commit=$EXPECTED_COMMIT"
echo "model_sha256=$EXPECTED_MODEL_SHA256"
echo "No benchmark job was submitted."
trap - EXIT
