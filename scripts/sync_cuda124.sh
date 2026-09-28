#!/bin/sh
set -eu

PROJECT_ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
CUDA_ENV=${LDA_CUDA_ENV:-"$PROJECT_ROOT/.venv-cu124"}
UV_CACHE=${LDA_UV_CACHE:-"${XDG_CACHE_HOME:-$HOME/.cache}/laya-dynamics-agent/uv"}

cd "$PROJECT_ROOT"
UV_PROJECT_ENVIRONMENT="$CUDA_ENV" UV_CACHE_DIR="$UV_CACHE" uv sync --extra test --extra laya
UV_CACHE_DIR="$UV_CACHE" uv pip install \
  --python "$CUDA_ENV/bin/python" \
  --index-url https://download.pytorch.org/whl/cu124 \
  "torch==2.6.0"
LAYA_DEVICE=cuda "$CUDA_ENV/bin/lda" doctor --require-cuda
