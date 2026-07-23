#!/usr/bin/env bash

set -euo pipefail

: "${FF3D_ENV_ROOT:?Missing FF3D_ENV_ROOT}"

TOOLCHAIN="$FF3D_ENV_ROOT/toolchain"
VENV="$FF3D_ENV_ROOT/venv"
SOURCE_ROOT="$FF3D_ENV_ROOT/source"
CACHE_ROOT="$FF3D_ENV_ROOT/cache"
BUILD_HOME="$FF3D_ENV_ROOT/home"

printf 'stage=rootless_builder_start environment_root=%s\n' "$FF3D_ENV_ROOT"
test -d "$FF3D_ENV_ROOT"
test ! -e "$TOOLCHAIN"
test ! -e "$VENV"
test ! -e "$SOURCE_ROOT"

mkdir "$SOURCE_ROOT" "$CACHE_ROOT" "$BUILD_HOME"

export HOME="$BUILD_HOME"
export CONDA_PKGS_DIRS="$CACHE_ROOT/conda_pkgs"
export CONDA_ENVS_PATH="$CACHE_ROOT/conda_envs"
export PIP_CACHE_DIR="$CACHE_ROOT/pip"
export TORCH_EXTENSIONS_DIR="$CACHE_ROOT/torch_extensions"
export CUDA_CACHE_PATH="$CACHE_ROOT/cuda"
export XDG_CACHE_HOME="$CACHE_ROOT/xdg"
export MPLCONFIGDIR="$CACHE_ROOT/matplotlib"

mkdir -p \
  "$CONDA_PKGS_DIRS" \
  "$CONDA_ENVS_PATH" \
  "$PIP_CACHE_DIR" \
  "$TORCH_EXTENSIONS_DIR" \
  "$CUDA_CACHE_PATH" \
  "$XDG_CACHE_HOME" \
  "$MPLCONFIGDIR"

echo "stage=conda_toolchain_create"
/opt/conda/bin/conda create --yes \
  --prefix "$TOOLCHAIN" \
  --override-channels \
  --channel conda-forge \
  ffmpeg=4.4 \
  gcc_linux-64=9 \
  gxx_linux-64=9 \
  git=2.40 \
  glib \
  libglvnd \
  ninja=1.11.1 \
  openblas=0.3.21 \
  pkg-config=0.29.2 \
  xorg-libsm=1.2.4 \
  xorg-libxext=1.3.4 \
  xorg-libxrender

# Conda compiler activation scripts reference variables before assigning them.
# Limit nounset suspension to activation, then restore the fail-closed shell.
set +u
eval "$(/opt/conda/bin/conda shell.bash hook)"
conda activate "$TOOLCHAIN"
set -u

echo "stage=virtual_environment_create"
/opt/conda/bin/python -m venv --system-site-packages "$VENV"
source "$VENV/bin/activate"

export PATH="$VENV/bin:$TOOLCHAIN/bin:/usr/local/cuda/bin:/opt/conda/bin:$PATH"
export LD_LIBRARY_PATH="$TOOLCHAIN/lib:/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}"
export TORCH_CUDA_ARCH_LIST="8.0"
export FORCE_CUDA=1
export MAX_JOBS="${SLURM_CPUS_PER_TASK:-8}"
export PYTHONDONTWRITEBYTECODE=1
export CPATH="$TOOLCHAIN/include:${CPATH:-}"
export LIBRARY_PATH="$TOOLCHAIN/lib:${LIBRARY_PATH:-}"

echo "stage=blas_toolchain_validate"
test -f "$TOOLCHAIN/include/cblas.h"
test -e "$TOOLCHAIN/lib/libopenblas.so"

clone_exact() {
  local repository="$1"
  local revision="$2"
  local destination="$3"

  git init "$destination"
  git -C "$destination" remote add origin "$repository"
  git -C "$destination" fetch --depth=1 origin "$revision"
  git -C "$destination" checkout --detach FETCH_HEAD
  test -z "$(git -C "$destination" status --porcelain)"
}

clone_exact \
  https://github.com/SmartForest-no/ForestFormer3D.git \
  6a75c3735e4a4108d02ee944a8b93177f2360a4f \
  "$SOURCE_ROOT/ForestFormer3D"
test "$(git -C "$SOURCE_ROOT/ForestFormer3D" rev-parse HEAD)" = \
  "6a75c3735e4a4108d02ee944a8b93177f2360a4f"

clone_exact \
  https://github.com/open-mmlab/mmdetection3d.git \
  22aaa47fdb53ce1870ff92cb7e3f96ae38d17f61 \
  "$SOURCE_ROOT/mmdetection3d"
test "$(git -C "$SOURCE_ROOT/mmdetection3d" rev-parse HEAD)" = \
  "22aaa47fdb53ce1870ff92cb7e3f96ae38d17f61"

clone_exact \
  https://github.com/NVIDIA/MinkowskiEngine.git \
  02fc608bea4c0549b0a7b00ca1bf15dee4a0b228 \
  "$SOURCE_ROOT/MinkowskiEngine"
test "$(git -C "$SOURCE_ROOT/MinkowskiEngine" rev-parse HEAD)" = \
  "02fc608bea4c0549b0a7b00ca1bf15dee4a0b228"

clone_exact \
  https://github.com/rusty1s/pytorch_scatter.git \
  refs/tags/2.0.9 \
  "$SOURCE_ROOT/pytorch_scatter"

clone_exact \
  https://github.com/Karbo123/segmentator.git \
  76efe46d03dd27afa78df972b17d07f2c6cfb696 \
  "$SOURCE_ROOT/segmentator"
test "$(git -C "$SOURCE_ROOT/segmentator" rev-parse HEAD)" = \
  "76efe46d03dd27afa78df972b17d07f2c6cfb696"

echo "stage=python_dependencies_install"
python -m pip install debugpy

python -m pip install --no-deps \
  mmengine==0.7.3 \
  mmdet==3.0.0 \
  mmsegmentation==1.0.0 \
  "$SOURCE_ROOT/mmdetection3d"

python -m pip install \
  mmcv==2.0.0 \
  -f https://download.openmmlab.com/mmcv/dist/cu116/torch1.13.0/index.html \
  --no-deps

python -m pip install \
  "$SOURCE_ROOT/MinkowskiEngine" \
  -v \
  --no-deps \
  --install-option="--blas=openblas" \
  --install-option="--blas_include_dirs=$TOOLCHAIN/include" \
  --install-option="--blas_library_dirs=$TOOLCHAIN/lib" \
  --install-option="--force_cuda"

python -m pip install "$SOURCE_ROOT/pytorch_scatter"

SEGMENTATOR_BUILD="$SOURCE_ROOT/segmentator/csrc/build"
mkdir "$SEGMENTATOR_BUILD"
echo "stage=segmentator_build"
(
  cd "$SEGMENTATOR_BUILD"
  cmake .. \
    -DCMAKE_PREFIX_PATH="$(python -c 'import torch; print(torch.utils.cmake_prefix_path())')" \
    -DPYTHON_INCLUDE_DIR="$(python -c 'from distutils.sysconfig import get_python_inc; print(get_python_inc())')" \
    -DPYTHON_LIBRARY="$(python -c 'import distutils.sysconfig as sysconfig; print(sysconfig.get_config_var("LIBDIR") + "/libpython3.10.so")')" \
    -DCMAKE_INSTALL_PREFIX="$(python -c 'from distutils.sysconfig import get_python_lib; print(get_python_lib())')"
  make -j2
  make install
)

python -m pip install --no-deps \
  spconv-cu116==2.3.6 \
  addict==2.4.0 \
  yapf==0.33.0 \
  termcolor==2.3.0 \
  packaging==23.1 \
  numpy==1.24.1 \
  rich==13.3.5 \
  opencv-python==4.7.0.72 \
  pycocotools==2.0.6 \
  Shapely==1.8.5 \
  scipy==1.10.1 \
  terminaltables==3.1.10 \
  numba==0.57.0 \
  llvmlite==0.40.0 \
  pccm==0.4.7 \
  ccimport==0.4.2 \
  pybind11==2.10.4 \
  ninja==1.11.1 \
  lark==1.1.5 \
  cumm-cu116==0.4.9 \
  pyquaternion==0.9.9 \
  lyft-dataset-sdk==0.0.8 \
  pandas==2.0.1 \
  python-dateutil==2.8.2 \
  matplotlib==3.5.2 \
  pyparsing==3.0.9 \
  cycler==0.11.0 \
  kiwisolver==1.4.4 \
  scikit-learn==1.2.2 \
  joblib==1.2.0 \
  threadpoolctl==3.1.0 \
  cachetools==5.3.0 \
  nuscenes-devkit==1.1.10 \
  trimesh==3.21.6 \
  open3d==0.17.0 \
  plotly==5.18.0 \
  dash==2.14.2 \
  plyfile==1.0.2 \
  flask==3.0.0 \
  werkzeug==3.0.1 \
  click==8.1.7 \
  blinker==1.7.0 \
  itsdangerous==2.1.2 \
  importlib_metadata==2.1.2 \
  zipp==3.17.0 \
  tensorboard==2.15.1 \
  tensorboard-data-server==0.7.2 \
  protobuf \
  absl-py \
  future \
  MarkupSafe==2.0.1 \
  markdown \
  grpcio \
  google-auth-oauthlib \
  google-auth \
  requests-oauthlib \
  oauthlib

python -m pip install --no-deps --no-cache-dir torch-points-kernels==0.7.0
python -m pip uninstall -y torch-cluster || true
python -m pip install --no-deps --no-cache-dir torch-cluster

SITE_PACKAGES="$(
  python -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])'
)"
FOREST_SOURCE="$SOURCE_ROOT/ForestFormer3D"

echo "stage=official_replacements_install"
test "$(sha256sum "$FOREST_SOURCE/replace_mmdetection_files/loops.py" | cut -d ' ' -f 1)" = \
  "df3b0d6688ae4f911fa6cbe8b1afb90520b1d147b3da800ee22075993a0bae27"
test "$(sha256sum "$FOREST_SOURCE/replace_mmdetection_files/base_model.py" | cut -d ' ' -f 1)" = \
  "9fb88239dd8eeddadbe6c909dc6bd5d613d3bbd487e272592972c856e56e233d"
test "$(sha256sum "$FOREST_SOURCE/replace_mmdetection_files/transforms_3d.py" | cut -d ' ' -f 1)" = \
  "c1a34b5a2ce006739fd1b810fdbe9cfc12f4b443acf389df71c6228aad690be9"

cp "$FOREST_SOURCE/replace_mmdetection_files/loops.py" \
  "$SITE_PACKAGES/mmengine/runner/loops.py"
cp "$FOREST_SOURCE/replace_mmdetection_files/base_model.py" \
  "$SITE_PACKAGES/mmengine/model/base_model/base_model.py"
cp "$FOREST_SOURCE/replace_mmdetection_files/transforms_3d.py" \
  "$SITE_PACKAGES/mmdet3d/datasets/transforms/transforms_3d.py"

python -m pip freeze > "$FF3D_ENV_ROOT/pip_freeze.txt"
/opt/conda/bin/conda list --explicit --prefix "$TOOLCHAIN" \
  > "$FF3D_ENV_ROOT/conda_explicit.txt"

printf '%s\n' "6a75c3735e4a4108d02ee944a8b93177f2360a4f" \
  > "$FF3D_ENV_ROOT/forestformer3d_commit.txt"
printf '%s\n' "22aaa47fdb53ce1870ff92cb7e3f96ae38d17f61" \
  > "$FF3D_ENV_ROOT/mmdetection3d_commit.txt"
printf '%s\n' "02fc608bea4c0549b0a7b00ca1bf15dee4a0b228" \
  > "$FF3D_ENV_ROOT/minkowski_engine_commit.txt"
git -C "$SOURCE_ROOT/pytorch_scatter" rev-parse HEAD \
  > "$FF3D_ENV_ROOT/pytorch_scatter_commit.txt"
printf '%s\n' "76efe46d03dd27afa78df972b17d07f2c6cfb696" \
  > "$FF3D_ENV_ROOT/segmentator_commit.txt"

echo "status=forestformer3d-rootless-environment-built"
