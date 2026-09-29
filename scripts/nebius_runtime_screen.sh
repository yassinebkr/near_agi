#!/bin/sh
set -eu

test -n "${OPENROUTER_API_KEY:-}" || {
  echo "OPENROUTER_API_KEY is required for runtime benchmarks." >&2
  exit 2
}
command -v screen >/dev/null 2>&1 || {
  echo "screen is required." >&2
  exit 2
}

root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
workspace=${NEBIUS_WORKSPACE:-/data/laya-posttrain}
session=${LAYA_RUNTIME_SCREEN_SESSION:-laya-runtime}
log_dir="$workspace/logs"
mkdir -p "$log_dir"
log_file="$log_dir/runtime-$(date -u +%Y%m%dT%H%M%SZ).log"

if screen -list | grep -q "[.]$session[[:space:]]"; then
  echo "A screen session named $session already exists." >&2
  exit 2
fi

screen -DmS "$session" -L -Logfile "$log_file" sh -c '
  trap "sudo shutdown -h now || true" EXIT HUP INT TERM
  "$1"
' sh "$root/scripts/nebius_runtime_benchmarks.sh"

printf '[launcher] runtime session=%s | log=%s\n' "$session" "$log_file"
