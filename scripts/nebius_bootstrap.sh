#!/bin/sh
set -eu

root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
workspace=${NEBIUS_WORKSPACE:-/data/laya-posttrain}
venv="$workspace/venv"

test -d /data || { echo "/data persistent disk is not mounted" >&2; exit 2; }
if ! command -v screen >/dev/null 2>&1 || ! command -v rsync >/dev/null 2>&1; then
  sudo apt-get update
  sudo DEBIAN_FRONTEND=noninteractive apt-get install -y screen rsync
fi
mkdir -p "$workspace/checkpoints" "$workspace/data"
python3 -m venv "$venv"
"$venv/bin/python" -m pip install --upgrade pip
"$venv/bin/python" -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
"$venv/bin/python" -m pip install -r "$root/configs/train/requirements-nebius.txt"
"$venv/bin/python" -m pip install --no-deps \
  "laya @ git+https://github.com/NandhaKishorM/laya.git@9d955671415fc19f069b9cc998928075c1f255ec"
"$venv/bin/python" -m pip install --no-deps -e "$root"
"$venv/bin/python" -m pip freeze > "$workspace/venv-freeze.txt"
"$venv/bin/python" -c 'import torch; assert torch.cuda.is_available(); print(torch.__version__, torch.cuda.get_device_name(0))'
