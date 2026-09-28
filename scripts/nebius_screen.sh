#!/bin/sh
set -eu

if [ "${NEBIUS_TRAIN_APPROVED:-}" != "YES" ]; then
  echo "Refusing paid training. Set NEBIUS_TRAIN_APPROVED=YES after checking the live price." >&2
  exit 2
fi
if [ "$#" -ne 1 ] || { [ "$1" != "smoke" ] && [ "$1" != "full" ]; }; then
  echo "usage: NEBIUS_TRAIN_APPROVED=YES $0 smoke|full" >&2
  exit 2
fi
command -v screen >/dev/null 2>&1 || { echo "screen is required; install it with: sudo apt-get install -y screen" >&2; exit 2; }

mode=$1
root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
workspace=${NEBIUS_WORKSPACE:-/data/laya-posttrain}
session=${LAYA_SCREEN_SESSION:-laya}
log_dir="$workspace/logs"
mkdir -p "$log_dir"
log_file="$log_dir/${mode}-$(date -u +%Y%m%dT%H%M%SZ).log"

if screen -list | grep -q "[.]$session[[:space:]]"; then
  echo "A screen session named $session already exists. Attach with: screen -r $session" >&2
  exit 2
fi

if [ "$mode" = "smoke" ]; then
  auto_shutdown=0
  max_steps=3
  runtime_benchmarks=0
else
  auto_shutdown=1
  max_steps=0
  runtime_benchmarks=1
  test -n "${OPENROUTER_API_KEY:-}" || { echo "Export OPENROUTER_API_KEY before starting the full screen session." >&2; exit 2; }
fi

printf '[launcher] mode=%s | session=%s | log=%s\n' "$mode" "$session" "$log_file"
screen -DmS "$session" -L -Logfile "$log_file" \
  env NEBIUS_TRAIN_APPROVED=YES AUTO_SHUTDOWN="$auto_shutdown" MAX_STEPS="$max_steps" RUN_RUNTIME_BENCHMARKS="$runtime_benchmarks" \
  MAX_WALL_SECONDS="${MAX_WALL_SECONDS:-21600}" "$root/scripts/nebius_run.sh"
printf '[launcher] started. Attach to the original live output with: screen -r %s\n' "$session"
printf '[launcher] detach without stopping it with: Ctrl+A then D\n'
