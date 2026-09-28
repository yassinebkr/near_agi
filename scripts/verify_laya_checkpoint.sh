#!/bin/sh
set -eu
model_dir="${LAYA_CHECKPOINT:-/mnt/fast-ssd/models/laya/base-english}"
expected="891102d372688fc2a094dac56a384bc537b87c63f21f9f3dac0be2b7cbc8d86c"
test -f "$model_dir/model.safetensors"
actual="$(sha256sum "$model_dir/model.safetensors" | cut -d ' ' -f 1)"
test "$actual" = "$expected"
test -f "$model_dir/rl_agent_config.json"
test -f "$model_dir/tokenizer/tokenizer.json"
printf 'checkpoint_ok path=%s sha256=%s\n' "$model_dir" "$actual"

