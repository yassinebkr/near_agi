#!/bin/sh
set -eu

if [ "$#" -ne 1 ]; then
  echo "usage: $0 user@vm" >&2
  exit 2
fi

root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
remote=$1
dataset=${POSTTRAIN_DATA_ROOT:-"$root/data/posttrain/v002bis"}
base=${LAYA_BASE:-}

test -n "$base" || { echo "Set LAYA_BASE to the local base-checkpoint directory" >&2; exit 2; }
test -f "$dataset/raw/manifest.json" || { echo "Build the dataset first" >&2; exit 2; }
rsync -az --delete --rsync-path="mkdir -p /data/laya-posttrain/repo && rsync" --exclude .git --exclude .env --exclude data --exclude reports --exclude logs \
  "$root/" "$remote:/data/laya-posttrain/repo/"
rsync -az --rsync-path="mkdir -p /data/laya-posttrain/data/v002bis/raw && rsync" "$dataset/raw/" "$remote:/data/laya-posttrain/data/v002bis/raw/"
rsync -az --rsync-path="mkdir -p /data/laya-posttrain/base-english && rsync" "$base/" "$remote:/data/laya-posttrain/base-english/"
printf 'Staged code, frozen raw dataset, and base checkpoint. No training was started.\n'
