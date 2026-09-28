#!/usr/bin/env bash
set -euo pipefail

root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
python_bin=${PYTHON_BIN:-python}
output=${1:-"$root/data/posttrain/v002bis/raw"}

"$python_bin" -m laya_dynamics_agent.training_data_v2bis --output "$output"
cp "$output/manifest.json" "$root/configs/train/v002bis.dataset-manifest.json"
