#!/bin/sh
set -eu

root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
workspace=${NEBIUS_WORKSPACE:-/data/laya-posttrain}
python_bin=${PYTHON_BIN:-"$workspace/venv/bin/python"}
dataset=${POSTTRAIN_DATA_ROOT:-"$workspace/data/v001"}
base=${LAYA_BASE:-"$workspace/base-english"}

test -x "$python_bin" || { echo "Missing Nebius virtual environment" >&2; exit 2; }
test -f "$base/model.safetensors" || { echo "Missing base checkpoint: $base" >&2; exit 2; }
test -f "$dataset/raw/manifest.json" || { echo "Missing dataset manifest" >&2; exit 2; }
available_kib=$(df -Pk "$workspace" | awk 'NR==2 {print $4}')
test "$available_kib" -ge 52428800 || { echo "At least 50 GiB free is required" >&2; exit 2; }

"$python_bin" -m laya_dynamics_agent.posttrain verify-data --dataset-dir "$dataset/raw"
"$python_bin" - <<'PY'
import torch
assert torch.cuda.is_available(), "CUDA unavailable"
props = torch.cuda.get_device_properties(0)
assert props.total_memory >= 40 * 2**30, f"expected >=40 GiB VRAM, found {props.total_memory / 2**30:.1f}"
print({"gpu": props.name, "vram_gib": round(props.total_memory / 2**30, 2), "torch": torch.__version__})
PY

sha256sum "$base/model.safetensors" "$dataset/raw/manifest.json"
printf 'Preflight passed. No training was started.\n'
