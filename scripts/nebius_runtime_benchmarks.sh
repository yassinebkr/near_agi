#!/bin/sh
set -eu

root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
workspace=${NEBIUS_WORKSPACE:-/data/laya-posttrain}
python_bin="$workspace/venv/bin/python"
lda="$workspace/venv/bin/lda"
base=${LAYA_BASE:-"$workspace/base-english"}
fine=${LAYA_OUTPUT:-"$workspace/checkpoints/laya-dynamics-v002bis"}
provider=${BENCHMARK_PROVIDER:-openrouter}
model=${OPENROUTER_MODEL:-openai/gpt-5.6-sol}
state_dir="$workspace/runtime-eval/v002bis"
mkdir -p "$state_dir"

# Screen sessions start in the caller's working directory. The benchmark
# currently resolves prompt assets relative to the repository root.
cd "$root"

test -n "${OPENROUTER_API_KEY:-}" || {
  echo "OPENROUTER_API_KEY is required for automatic runtime gates." >&2
  exit 2
}
test -f "$base/model.safetensors" || { echo "Base checkpoint missing: $base" >&2; exit 2; }
test -f "$fine/model.safetensors" || { echo "Fine-tuned checkpoint missing: $fine" >&2; exit 2; }

phase() {
  printf '[evaluation %s] %s\n' "$(date -u +%H:%M:%S)" "$1"
}

run_suite() {
  suite=$1
  seeds=$2
  campaign="benchmark-v002bis-$suite"
  report="$root/reports/$campaign/metrics.json"
  gate="$state_dir/$suite-gate.json"
  resume_arg=
  if [ -f "$root/reports/$campaign/checkpoint.json" ]; then
    resume_arg=--resume
  fi
  phase "Starting $suite (campaign $campaign)"
  LAYA_DEVICE=cuda "$lda" benchmark \
    --suite "$suite" \
    --provider "$provider" \
    --model "$model" \
    --seeds "$seeds" \
    --base-laya-checkpoint "$base" \
    --finetuned-laya-checkpoint "$fine" \
    --fresh-candidate-cache \
    --campaign-id "$campaign" \
    $resume_arg
  "$python_bin" -m laya_dynamics_agent.runtime_gate \
    --report "$report" --output "$gate"
  phase "$suite gate passed"
}

run_suite smoke 0
run_suite challenge 0
run_suite final 0,1,2,3,4
phase "All runtime gates and final-v001 completed"
