#!/bin/sh
set -eu
model_dir=${1:-"${LAYA_CHECKPOINT:-}"}
expected=${LAYA_EXPECTED_SHA256:-}
test -n "$model_dir" || { echo "Pass a checkpoint path or set LAYA_CHECKPOINT" >&2; exit 2; }
test -f "$model_dir/model.safetensors"
actual="$(sha256sum "$model_dir/model.safetensors" | cut -d ' ' -f 1)"
if [ -n "$expected" ]; then
  test "$actual" = "$expected"
fi
test -f "$model_dir/rl_agent_config.json"
test -f "$model_dir/tokenizer/tokenizer.json"
printf 'checkpoint_ok path=%s sha256=%s\n' "$model_dir" "$actual"
