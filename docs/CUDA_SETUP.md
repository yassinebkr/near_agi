# Stable CUDA Setup for Laya Campaigns

Large campaigns must use an explicit GPU policy. Set `LAYA_DEVICE=cuda`; the process then fails before loading Laya if PyTorch cannot initialize CUDA. `LAYA_DEVICE=auto` is reserved for exploratory work where a visible CPU fallback is acceptable.

The current reference machine reports an NVIDIA driver capability of CUDA 12.4 while its environment contains PyTorch `2.14.0+cu130`. That combination cannot initialize CUDA. The stable target is a dedicated Python 3.13 environment with a CUDA 12.4 PyTorch wheel, currently pinned for this machine to `torch==2.6.0`.

## Validated reference stack

Validated locally on 2026-09-28:

```text
Python: 3.13.15
PyTorch: 2.6.0+cu124
CUDA available: true
Laya: 0.3.21
Laya load time: 5.2 s
Three-task Laya predictor time: 1.4 s
```

The previous CPU run took approximately 36.8 seconds of predictor time. The CUDA smoke is therefore roughly 26 times faster. The checkpoint-temperature warning remains and still means that zero-shot confidence values are uncalibrated; it is independent of CUDA.

## Rebuild the reference environment

Keep the existing environment until the replacement passes every check. The idempotent setup script creates or repairs a separate CUDA environment and runs the strict preflight:

```bash
cd near_agi
export LDA_CUDA_ENV="${LDA_CUDA_ENV:-$(pwd)/.venv-cu124}"
export LDA_UV_CACHE="${LDA_UV_CACHE:-${XDG_CACHE_HOME:-$HOME/.cache}/laya-dynamics-agent/uv}"
./scripts/sync_cuda124.sh
```

The script performs the following explicit operations:

```bash
cd near_agi

UV_PROJECT_ENVIRONMENT="$LDA_CUDA_ENV" \
UV_CACHE_DIR="$LDA_UV_CACHE" \
uv sync --extra test --extra laya

uv pip install \
  --python "$LDA_CUDA_ENV/bin/python" \
  --index-url https://download.pytorch.org/whl/cu124 \
  "torch==2.6.0"
```

A normal `uv sync --extra laya` currently resolves PyTorch 2.14 with CUDA 13 and can overwrite the compatible wheel. Re-run `./scripts/sync_cuda124.sh` after any dependency sync; its final strict preflight prevents an incompatible environment from being used silently.

Do not promote this environment merely because installation succeeds. The script runs this strict preflight:

```bash
cd near_agi
LAYA_DEVICE=cuda \
"$LDA_CUDA_ENV/bin/lda" doctor --require-cuda
```

The command must report `torch_cuda: true`, `cuda_ready: true`, and exit with status zero. Then run a no-API Laya smoke:

```bash
LAYA_DEVICE=cuda \
"$LDA_CUDA_ENV/bin/lda" benchmark \
  --suite smoke \
  --provider deterministic \
  --seeds 0 \
  --with-laya
```

Record the exact PyTorch version, CUDA build, GPU name, Laya load time, and predictor latency. Only after this succeeds should the environment be used for an OpenRouter campaign. Keep `.venv` as the known CPU-capable rollback until the CUDA candidate is validated.

## Runtime contract

- `LAYA_DEVICE=cuda`: campaign mode; CUDA is mandatory and CPU fallback is forbidden.
- `LAYA_DEVICE=cpu`: intentional CPU baseline.
- `LAYA_DEVICE=auto`: exploratory mode; selects CUDA when available and otherwise reports CPU.

Every benchmark report records the resolved device, checkpoint, and model load time under `laya_runtime`.
