#!/bin/sh
set -eu

if [ "$#" -ne 1 ]; then
  echo "usage: $0 user@vm" >&2
  exit 2
fi

root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
remote=$1
dataset=${POSTTRAIN_DATA_ROOT:-"$root/data/posttrain/v001"}
base=${LAYA_BASE:-/mnt/fast-ssd/models/laya/base-english}

test -f "$dataset/raw/manifest.json" || { echo "Build the dataset first" >&2; exit 2; }
ssh "$remote" "mkdir -p /data/laya-posttrain/repo /data/laya-posttrain/data/v001/raw /data/laya-posttrain/base-english"
rsync -az --delete --exclude .git --exclude .env --exclude data --exclude reports --exclude logs \
  "$root/" "$remote:/data/laya-posttrain/repo/"
rsync -az "$dataset/raw/" "$remote:/data/laya-posttrain/data/v001/raw/"
rsync -az "$base/" "$remote:/data/laya-posttrain/base-english/"
printf 'Staged code, frozen raw dataset, and base checkpoint. No training was started.\n'
