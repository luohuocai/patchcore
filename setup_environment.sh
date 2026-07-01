#!/usr/bin/env bash
set -u

cd "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

usage() {
    cat <<'EOF'
Usage:
  bash setup_environment.sh
  bash setup_environment.sh cpu
  bash setup_environment.sh cu121
  bash setup_environment.sh cu118

Default mode is auto:
  - NVIDIA driver detected: try CUDA 12.1 PyTorch wheels.
  - No NVIDIA driver: install CPU PyTorch wheels.
EOF
}

fail() {
    echo
    echo "========================================"
    echo "Setup failed."
    echo "Check the error message above."
    echo "If PyTorch/CUDA failed, try:"
    echo "  bash setup_environment.sh cpu"
    echo "  bash setup_environment.sh cu121"
    echo "  bash setup_environment.sh cu118"
    echo "========================================"
    exit 1
}

echo "========================================"
echo "PatchCore environment setup"
echo "Project directory: $(pwd)"
echo "========================================"
echo

case "${1:-}" in
    help|--help|-h)
        usage
        exit 0
        ;;
esac

PYTHON_CMD=""
if command -v python3 >/dev/null 2>&1; then
    PYTHON_CMD="python3"
elif command -v python >/dev/null 2>&1; then
    PYTHON_CMD="python"
fi

if [ -z "$PYTHON_CMD" ]; then
    echo "ERROR: Python was not found."
    echo "Install Python 3.10 or 3.11 first, then run this file again."
    exit 1
fi

echo "[1/7] Checking Python..."
"$PYTHON_CMD" --version
"$PYTHON_CMD" - <<'PY'
import sys
print("Python executable:", sys.executable)
raise SystemExit(0 if sys.version_info >= (3, 8) else 1)
PY
if [ "$?" -ne 0 ]; then
    echo "ERROR: Python 3.8 or newer is required. Python 3.10 or 3.11 is recommended."
    exit 1
fi

echo
echo "[2/7] Creating virtual environment..."
if [ ! -x ".venv/bin/python" ]; then
    "$PYTHON_CMD" -m venv .venv
    if [ "$?" -ne 0 ]; then
        echo "ERROR: Failed to create .venv."
        echo "On Ubuntu/Debian, install venv support first, for example:"
        echo "  sudo apt update"
        echo "  sudo apt install python3-venv"
        fail
    fi
else
    echo "Existing .venv found. Reusing it."
fi

echo
echo "[3/7] Activating virtual environment..."
# shellcheck disable=SC1091
source ".venv/bin/activate" || fail

echo
echo "[4/7] Upgrading pip tooling..."
python -m pip install --upgrade pip setuptools wheel || fail

TORCH_PROFILE="${1:-auto}"

install_torch_cpu() {
    python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
}

install_torch_cu118() {
    python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
}

install_torch_cu121() {
    python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
}

echo
echo "[5/7] Installing PyTorch profile: $TORCH_PROFILE"
case "$TORCH_PROFILE" in
    cpu)
        install_torch_cpu || fail
        ;;
    cu118)
        install_torch_cu118 || fail
        ;;
    cu121)
        install_torch_cu121 || fail
        ;;
    auto)
        if command -v nvidia-smi >/dev/null 2>&1; then
            echo "NVIDIA driver detected. Trying CUDA 12.1 PyTorch wheels."
            install_torch_cu121 || {
                echo "CUDA 12.1 PyTorch install failed. Falling back to CPU PyTorch wheels."
                install_torch_cpu || fail
            }
        else
            echo "NVIDIA driver was not detected. Installing CPU PyTorch wheels."
            install_torch_cpu || fail
        fi
        ;;
    *)
        echo "ERROR: Unknown PyTorch profile \"$TORCH_PROFILE\"."
        usage
        exit 1
        ;;
esac

echo
echo "[6/7] Installing project dependencies..."
python -m pip install -r requirements.txt || fail
python -m pip install -r requirements_dev.txt || fail
python -m pip install -e . || fail

echo
echo "[7/7] Verifying imports..."
python - <<'PY'
import torch
import torchvision
import faiss
import numpy
import sklearn
import scipy
import timm
import patchcore

print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("cuda version:", torch.version.cuda)
print("patchcore import: OK")
PY
if [ "$?" -ne 0 ]; then
    fail
fi

echo
echo "========================================"
echo "Setup completed successfully."
echo "To use this environment later, run:"
echo "  source .venv/bin/activate"
echo "========================================"
