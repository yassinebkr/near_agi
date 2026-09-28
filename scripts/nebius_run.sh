#!/bin/sh
set -eu

if [ "${NEBIUS_TRAIN_APPROVED:-}" != "YES" ]; then
  echo "Refusing paid training. Export NEBIUS_TRAIN_APPROVED=YES only after checking the live hourly price." >&2
  exit 2
fi

root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
workspace=${NEBIUS_WORKSPACE:-/data/laya-posttrain}
python_bin="$workspace/venv/bin/python"
dataset=${POSTTRAIN_DATA_ROOT:-"$workspace/data/v002bis"}
base=${LAYA_BASE:-"$workspace/base-english"}
output=${LAYA_OUTPUT:-"$workspace/checkpoints/laya-dynamics-v002bis"}

phase() {
  printf '[pipeline %s] %s\n' "$(date -u +%H:%M:%S)" "$1"
}

shutdown_vm() {
  trap - EXIT HUP INT TERM
  if [ "${AUTO_SHUTDOWN:-1}" = "1" ]; then
    sudo shutdown -h now || true
  fi
}
trap shutdown_vm EXIT HUP INT TERM

phase "Running safety and hardware preflight"
"$root/scripts/nebius_preflight.sh"
phase "Preflight passed"

if [ ! -f "$dataset/tokenized/manifest.json" ]; then
  phase "Preprocessing the frozen dataset"
  "$python_bin" -m laya_dynamics_agent.posttrain preprocess \
    --dataset-dir "$dataset/raw" --checkpoint "$base" --output-dir "$dataset/tokenized"
  phase "Dataset preprocessing complete"
else
  phase "Using the existing verified tokenized dataset"
fi

phase "Starting or resuming Laya training"
POSTTRAIN_APPROVED=YES PYTHON_BIN="$python_bin" POSTTRAIN_DATA_ROOT="$dataset" \
LAYA_BASE="$base" LAYA_OUTPUT="$output" MAX_WALL_SECONDS="${MAX_WALL_SECONDS:-21600}" \
  "$root/scripts/train_laya.sh"

if [ -f "$output/model.safetensors" ]; then
  phase "Training complete; calibrating the exported checkpoint"
  "$python_bin" -m laya_dynamics_agent.posttrain calibrate \
    --items "$dataset/tokenized/calibration.pt" --checkpoint "$output"
  mkdir -p "$output/evaluation"
  phase "Evaluating the base checkpoint on the development split"
  "$python_bin" -m laya_dynamics_agent.posttrain evaluate \
    --items "$dataset/tokenized/development.pt" --checkpoint "$base" \
    --output "$output/evaluation/base-development.json"
  phase "Evaluating the candidate checkpoint on the development split"
  "$python_bin" -m laya_dynamics_agent.posttrain evaluate \
    --items "$dataset/tokenized/development.pt" --checkpoint "$output" \
    --output "$output/evaluation/candidate-development.json"
  gate_status=0
  phase "Applying the promotion gate"
  "$python_bin" -m laya_dynamics_agent.posttrain gate \
    --base-report "$output/evaluation/base-development.json" \
    --candidate-report "$output/evaluation/candidate-development.json" \
    --output "$output/evaluation/promotion-gate.json" || gate_status=$?
  phase "Verifying the final checkpoint and writing checksums"
  "$root/scripts/verify_laya_checkpoint.sh" "$output"
  (cd "$output" && find . -type f ! -name SHA256SUMS -print0 | sort -z | xargs -0 sha256sum > SHA256SUMS)
  if [ "$gate_status" -eq 0 ]; then
    phase "Offline promotion gate passed"
    if [ "${RUN_RUNTIME_BENCHMARKS:-0}" = "1" ]; then
      phase "Starting checkpointed smoke, challenge and final runtime pipeline"
      "$root/scripts/nebius_runtime_benchmarks.sh"
    fi
    phase "Pipeline complete; all enabled gates passed"
  else
    phase "Pipeline complete; promotion gate failed"
  fi
  exit "$gate_status"
else
  phase "Training paused with a resumable checkpoint; calibration was not run"
fi
