#!/bin/sh
set -eu

root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
python_bin=${PYTHON_BIN:-python}
output=${1:-"$root/data/posttrain/v002/raw"}

cd "$root"
"$python_bin" -m laya_dynamics_agent.training_data_v2 --output "$output"
"$python_bin" -m laya_dynamics_agent.posttrain verify-data --dataset-dir "$output"
cp "$output/manifest.json" "$root/configs/train/v002.dataset-manifest.json"
printf 'Dataset v2 ready: %s\n' "$output"
