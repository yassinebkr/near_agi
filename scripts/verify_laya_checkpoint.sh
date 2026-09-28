#!/bin/sh
set -eu
model_dir=${1:-"${LAYA_CHECKPOINT:-/mnt/fast-ssd/models/laya/base-english}"}
expected=${LAYA_EXPECTED_SHA256:-}
if [ -z "$expected" ] && [ "$model_dir" = "/mnt/fast-ssd/models/laya/base-english" ]; then
  expected="891102d372688fc2a094dac56a384bc537b87c63f21f9f3dac0be2b7cbc8d86c"
fi
test -f "$model_dir/model.safetensors"
actual="$(sha256sum "$model_dir/model.safetensors" | cut -d ' ' -f 1)"
if [ -n "$expected" ]; then
  test "$actual" = "$expected"
fi
test -f "$model_dir/rl_agent_config.json"
test -f "$model_dir/tokenizer/tokenizer.json"
printf 'checkpoint_ok path=%s sha256=%s\n' "$model_dir" "$actual"
