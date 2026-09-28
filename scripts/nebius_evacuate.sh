#!/bin/sh
set -eu

if [ "$#" -ne 1 ]; then
  echo "usage: $0 user@vm" >&2
  exit 2
fi

root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
remote=$1
destination=${LAYA_EVACUATE_TO:-"$root/checkpoints/laya-dynamics-v002bis"}
mkdir -p "$destination"
rsync -az --partial "$remote:/data/laya-posttrain/checkpoints/laya-dynamics-v002bis/" "$destination/"
(cd "$destination" && sha256sum -c SHA256SUMS)
"$root/scripts/verify_laya_checkpoint.sh" "$destination"
printf 'Artifacts verified at %s. Delete the VM, disk, image/snapshot and static IP in Nebius.\n' "$destination"
