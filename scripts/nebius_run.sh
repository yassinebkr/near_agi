#!/bin/sh
set -eu

if [ "${NEBIUS_TRAIN_APPROVED:-}" != "YES" ]; then
  echo "Refusing paid training. Export NEBIUS_TRAIN_APPROVED=YES only after checking the live hourly price." >&2
  exit 2
fi

root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
workspace=${NEBIUS_WORKSPACE:-/data/laya-posttrain}
python_bin="$workspace/venv/bin/python"
dataset=${POSTTRAIN_DATA_ROOT:-"$workspace/data/v001"}
base=${LAYA_BASE:-"$workspace/base-english"}
output=${LAYA_OUTPUT:-"$workspace/checkpoints/laya-dynamics-v001"}

shutdown_vm() {
  trap - EXIT HUP INT TERM
  if [ "${AUTO_SHUTDOWN:-1}" = "1" ]; then
    sudo shutdown -h now || true
  fi
}
trap shutdown_vm EXIT HUP INT TERM

"$root/scripts/nebius_preflight.sh"

if [ ! -f "$dataset/tokenized/manifest.json" ]; then
  "$python_bin" -m laya_dynamics_agent.posttrain preprocess \
    --dataset-dir "$dataset/raw" --checkpoint "$base" --output-dir "$dataset/tokenized"
fi

POSTTRAIN_APPROVED=YES PYTHON_BIN="$python_bin" POSTTRAIN_DATA_ROOT="$dataset" \
LAYA_BASE="$base" LAYA_OUTPUT="$output" MAX_WALL_SECONDS="${MAX_WALL_SECONDS:-21600}" \
  "$root/scripts/train_laya.sh"

if [ -f "$output/model.safetensors" ]; then
  "$python_bin" -m laya_dynamics_agent.posttrain calibrate \
    --items "$dataset/tokenized/calibration.pt" --checkpoint "$output"
  mkdir -p "$output/evaluation"
  "$python_bin" -m laya_dynamics_agent.posttrain evaluate \
    --items "$dataset/tokenized/development.pt" --checkpoint "$base" \
    --output "$output/evaluation/base-development.json"
  "$python_bin" -m laya_dynamics_agent.posttrain evaluate \
    --items "$dataset/tokenized/development.pt" --checkpoint "$output" \
    --output "$output/evaluation/candidate-development.json"
  gate_status=0
  "$python_bin" -m laya_dynamics_agent.posttrain gate \
    --base-report "$output/evaluation/base-development.json" \
    --candidate-report "$output/evaluation/candidate-development.json" \
    --output "$output/evaluation/promotion-gate.json" || gate_status=$?
  "$root/scripts/verify_laya_checkpoint.sh" "$output"
  (cd "$output" && find . -type f ! -name SHA256SUMS -print0 | sort -z | xargs -0 sha256sum > SHA256SUMS)
  exit "$gate_status"
else
  echo "Training paused with a resumable checkpoint; final calibration was not run." >&2
fi
