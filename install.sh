#!/usr/bin/env bash
# One-shot install: PyTorch, Python deps, CUDA compiler, Chamfer/EMD, PointNet++, KNN_CUDA.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

PYTORCH_INDEX="${PYTORCH_INDEX:-https://download.pytorch.org/whl/cu132}"
CUDA_TOOLKIT_INDEX="${CUDA_TOOLKIT_INDEX:-https://pypi.org/simple}"
POINTNET2_GIT="${POINTNET2_GIT:-https://github.com/erikwijmans/Pointnet2_PyTorch.git}"
KNN_GIT="${KNN_GIT:-https://github.com/unique1i/knn_cuda.git}"

log() { printf '\n==> %s\n' "$*"; }

try_conda_activate() {
  local cand
  for cand in \
    "${CONDA_EXE:+${CONDA_EXE%/bin/conda}/etc/profile.d/conda.sh}" \
    "$HOME/miniforge3/etc/profile.d/conda.sh" \
    "$HOME/mambaforge/etc/profile.d/conda.sh" \
    "$HOME/anaconda3/etc/profile.d/conda.sh" \
    "$HOME/miniconda3/etc/profile.d/conda.sh"; do
    if [ -n "${cand:-}" ] && [ -f "$cand" ]; then
      # shellcheck source=/dev/null
      source "$cand"
      return 0
    fi
  done
  return 1
}

ensure_env() {
  if command -v python >/dev/null 2>&1 && python -c "import sys; raise SystemExit(0 if sys.version_info[:2] == (3, 10) else 1)"; then
    log "Using $(command -v python) ($(python -V))"
    return
  fi
  if try_conda_activate; then
    conda create -n spanet python=3.10 -y >/dev/null
    conda activate spanet
    log "Activated conda env spanet ($(python -V))"
    return
  fi
  echo "Need Python 3.10. Create an env first, for example:"
  echo "  conda create -n spanet python=3.10 -y && conda activate spanet"
  exit 1
}

ensure_torch() {
  if python -c "import torch; assert torch.cuda.is_available()" >/dev/null 2>&1; then
    log "PyTorch CUDA already available: $(python -c 'import torch; print(torch.__version__, torch.version.cuda)')"
    return
  fi
  log "Installing torch / torchvision from ${PYTORCH_INDEX}"
  python -m pip install torch torchvision --index-url "${PYTORCH_INDEX}"
  python -c "import torch; assert torch.cuda.is_available(), 'CUDA torch install failed'"
}

ensure_python_deps() {
  log "Installing Python dependencies"
  python -m pip install -r "${ROOT}/requirements.txt"
  python -m pip install --no-deps timm==1.0.12
}

setup_cuda_home() {
  log "Ensuring nvcc and CCCL headers"
  python -m pip install "cuda-toolkit[nvcc,cccl]" \
    || python -m pip install "cuda-toolkit[nvcc,cccl]" -i "${CUDA_TOOLKIT_INDEX}"

  CUDA_HOME="$(python - <<'PY'
import os, shutil, sys
from pathlib import Path

def has_nvcc(root: Path) -> bool:
    return (root / "bin" / "nvcc").is_file()

env = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")
if env and has_nvcc(Path(env)):
    print(env)
    raise SystemExit
which = shutil.which("nvcc")
if which:
    print(str(Path(which).resolve().parent.parent))
    raise SystemExit
site = Path(sys.prefix) / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}" / "site-packages"
for cand in (site / "nvidia" / "cu13", site / "nvidia" / "cuda_nvcc", Path(sys.prefix)):
    if has_nvcc(cand):
        print(cand)
        raise SystemExit
print("")
PY
)"
  if [ -z "${CUDA_HOME}" ]; then
    echo "nvcc not found. Install with:"
    echo "  pip install \"cuda-toolkit[nvcc,cccl]\" -i https://pypi.org/simple"
    exit 1
  fi
  export CUDA_HOME
  export PATH="${CUDA_HOME}/bin:${PATH}"
  export CPATH="${CUDA_HOME}/include:${CPATH:-}"
  if [ ! -e "${CUDA_HOME}/include/nv/target" ]; then
    echo "Missing ${CUDA_HOME}/include/nv/target"
    echo "  pip install nvidia-cuda-cccl -i https://pypi.org/simple"
    exit 1
  fi
  python - <<'PY'
import os
from pathlib import Path
lib = Path(os.environ["CUDA_HOME"]) / "lib"
lib64 = Path(os.environ["CUDA_HOME"]) / "lib64"
if lib.is_dir() and not lib64.exists():
    lib64.symlink_to("lib")
for so in lib.glob("lib*.so.*"):
    unversioned = lib / f"{so.name.split('.so.')[0]}.so"
    if not unversioned.exists():
        unversioned.symlink_to(so.name)
cudart = lib / "libcudart.so"
if not cudart.exists():
    raise SystemExit(f"still missing {cudart}")
print(f"linker can use {cudart} -> {cudart.resolve().name}")
PY
  export LIBRARY_PATH="${CUDA_HOME}/lib:${LIBRARY_PATH:-}"
  export LD_LIBRARY_PATH="${CUDA_HOME}/lib:${LD_LIBRARY_PATH:-}"
}

first_wheel() {
  local dir="$1"
  local wheel
  shopt -s nullglob
  for wheel in "${dir}"/*.whl; do
    printf '%s' "$wheel"
    return 0
  done
  return 0
}

patch_knn_cuda() {
  local root="$1"
  python - "$root" <<'PY'
import sys
from pathlib import Path
root = Path(sys.argv[1])
cpp = root / "knn_cuda" / "csrc" / "cuda" / "knn.cpp"
if not cpp.is_file():
    raise SystemExit(f"missing {cpp}")
src = cpp.read_text()
src = src.replace("x.device().type() == at::Device::Type::CUDA", "x.is_cuda()")
src = src.replace(".data<float>()", ".data_ptr<float>()")
src = src.replace(".data<long>()", ".data_ptr<long>()")
cpp.write_text(src)
PY
}

install_pointnet2() {
  local local_dir="${ROOT}/pointnet2_ops"
  local wheel
  wheel="$(first_wheel "${local_dir}")"
  if [ -n "${wheel}" ]; then
    log "Installing pointnet2_ops from ${wheel}"
    python -m pip install --no-build-isolation "${wheel}" \
      || echo "pointnet2_ops wheel failed; SPA-Net will use the PyTorch FPS fallback."
    return
  fi
  if [ -f "${local_dir}/setup.py" ]; then
    log "Installing pointnet2_ops from ${local_dir}"
    python -m pip install --no-build-isolation "${local_dir}" \
      || echo "pointnet2_ops failed to compile; SPA-Net will use the PyTorch FPS fallback."
    return
  fi
  log "pointnet2_ops not in repo; cloning ${POINTNET2_GIT}"
  local src="${BUILD_DIR}/Pointnet2_PyTorch"
  if git clone --depth 1 "${POINTNET2_GIT}" "${src}"; then
    python - "${src}/pointnet2_ops_lib/setup.py" "${ARCH}" <<'PY'
import sys
from pathlib import Path
setup_py, arch = Path(sys.argv[1]), sys.argv[2]
text = setup_py.read_text()
old = 'os.environ["TORCH_CUDA_ARCH_LIST"] = "3.7+PTX;5.0;6.0;6.1;6.2;7.0;7.5"'
new = f'os.environ["TORCH_CUDA_ARCH_LIST"] = "{arch}"'
if old in text:
    setup_py.write_text(text.replace(old, new, 1))
PY
    python -m pip install --no-build-isolation "${src}/pointnet2_ops_lib" \
      || echo "pointnet2_ops failed to compile; SPA-Net will use the PyTorch FPS fallback."
  else
    echo "Could not clone Pointnet2_PyTorch; SPA-Net will use the PyTorch FPS fallback."
  fi
}

install_knn_cuda() {
  local local_dir="${ROOT}/KNN_CUDA"
  local wheel
  wheel="$(first_wheel "${local_dir}")"
  if [ -n "${wheel}" ]; then
    log "Installing KNN_CUDA from ${wheel}"
    python -m pip install --no-deps --no-build-isolation "${wheel}" \
      || echo "KNN_CUDA wheel failed; SPA-Net will use the PyTorch KNN fallback."
    python -c "from knn_cuda import KNN; print('KNN_CUDA import ok')" \
      || echo "KNN_CUDA installed but JIT compile failed; SPA-Net will use the PyTorch KNN fallback."
    return
  fi
  if [ -f "${local_dir}/setup.py" ]; then
    log "Installing KNN_CUDA from ${local_dir}"
    python -m pip install --no-deps --no-build-isolation "${local_dir}" \
      || echo "KNN_CUDA failed to install; SPA-Net will use the PyTorch KNN fallback."
    python -c "from knn_cuda import KNN; print('KNN_CUDA import ok')" \
      || echo "KNN_CUDA installed but JIT compile failed; SPA-Net will use the PyTorch KNN fallback."
    return
  fi
  log "KNN_CUDA not in repo; cloning ${KNN_GIT}"
  local src="${BUILD_DIR}/knn_cuda"
  if git clone --depth 1 "${KNN_GIT}" "${src}"; then
    patch_knn_cuda "${src}"
    python -m pip install --no-deps --no-build-isolation "${src}" \
      || echo "KNN_CUDA failed to install; SPA-Net will use the PyTorch KNN fallback."
    python -c "from knn_cuda import KNN; print('KNN_CUDA import ok')" \
      || echo "KNN_CUDA installed but JIT compile failed; SPA-Net will use the PyTorch KNN fallback."
  else
    echo "Could not clone unique1i/knn_cuda; SPA-Net will use the PyTorch KNN fallback."
  fi
}

ensure_env
ensure_torch
ensure_python_deps
setup_cuda_home

ARCH="$(python -c 'import torch; print("%d.%d" % torch.cuda.get_device_capability())')"
export TORCH_CUDA_ARCH_LIST="${ARCH}"
export FORCE_CUDA=1
log "Building CUDA ops for GPU arch ${TORCH_CUDA_ARCH_LIST} (CUDA_HOME=${CUDA_HOME})"

log "Installing Chamfer Distance and EMD"
python -m pip install --no-build-isolation "${ROOT}/extensions/chamfer_dist"
python -m pip install --no-build-isolation "${ROOT}/extensions/emd"

BUILD_DIR="$(mktemp -d)"
cleanup() { rm -rf "${BUILD_DIR}"; }
trap cleanup EXIT

install_pointnet2
install_knn_cuda

python - <<'PY'
import torch
print("torch", torch.__version__, "cuda", torch.version.cuda, "gpu", torch.cuda.get_device_name(0))
import chamfer
print("chamfer ok")
from utils import point_ops
print("pointnet2_ops:", "installed" if point_ops._pn2 else "pytorch fallback")
print("KNN_CUDA:", "installed" if point_ops._CudaKNN else "pytorch fallback")
PY
