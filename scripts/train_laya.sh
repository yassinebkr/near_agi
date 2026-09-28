#!/bin/sh
set -eu

if [ "${POSTTRAIN_APPROVED:-}" != "YES" ]; then
  echo "Refusing to train. Export POSTTRAIN_APPROVED=YES after reviewing the manifest and cost gate." >&2
  exit 2
fi

root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
python_bin=${PYTHON_BIN:-"$root/.venv/bin/python"}
data_root=${POSTTRAIN_DATA_ROOT:-"$root/data/posttrain/v001"}
base=${LAYA_BASE:-/data/laya-posttrain/base-english}
output=${LAYA_OUTPUT:-/data/laya-posttrain/checkpoints/laya-dynamics-v001}

exec "$python_bin" -m laya_dynamics_agent.posttrain train \
  --items "$data_root/tokenized/train.pt" \
  --base "$base" \
  --output "$output" \
  --epochs "${EPOCHS:-4}" \
  --micro-batch "${MICRO_BATCH:-4}" \
  --gradient-accumulation "${GRADIENT_ACCUMULATION:-8}" \
  --checkpoint-seconds "${CHECKPOINT_SECONDS:-600}" \
  --max-wall-seconds "${MAX_WALL_SECONDS:-21600}" \
  --max-steps "${MAX_STEPS:-0}"
