# Stable CUDA Setup for Laya Campaigns

Large campaigns must use an explicit GPU policy. Set `LAYA_DEVICE=cuda`; the process then fails before loading Laya if PyTorch cannot initialize CUDA. `LAYA_DEVICE=auto` is reserved for exploratory work where a visible CPU fallback is acceptable.

The current reference machine reports an NVIDIA driver capability of CUDA 12.4 while its environment contains PyTorch `2.14.0+cu130`. That combination cannot initialize CUDA. The stable target is a dedicated Python 3.13 environment with a CUDA 12.4 PyTorch wheel, currently pinned for this machine to `torch==2.6.0`.

## Rebuild the reference environment

Keep the existing environment until the replacement passes every check. Create a separate candidate environment:

```bash
cd /home/kwestog/Documents/code/near_agi

UV_PROJECT_ENVIRONMENT=/mnt/fast-ssd/laya-dynamics-agent/.venv-cu124 \
UV_CACHE_DIR=/mnt/fast-ssd/uv-cache \
uv sync --extra test --extra laya

uv pip install \
  --python /mnt/fast-ssd/laya-dynamics-agent/.venv-cu124/bin/python \
  --index-url https://download.pytorch.org/whl/cu124 \
  "torch==2.6.0"
```

Do not promote this environment merely because installation succeeds. Run the strict preflight:

```bash
cd /home/kwestog/Documents/code/near_agi
LAYA_DEVICE=cuda \
/mnt/fast-ssd/laya-dynamics-agent/.venv-cu124/bin/lda doctor --require-cuda
```

The command must report `torch_cuda: true`, `cuda_ready: true`, and exit with status zero. Then run a no-API Laya smoke:

```bash
LAYA_DEVICE=cuda \
/mnt/fast-ssd/laya-dynamics-agent/.venv-cu124/bin/lda benchmark \
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
