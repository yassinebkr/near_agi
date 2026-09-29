# Nebius Laya post-training and recovery runbook

## Current status

V2bis training, calibration, offline promotion, smoke, and challenge are complete. The frozen final campaign is paused after 705 of 1,800 episodes because the OpenRouter budget was exhausted. The complete cloud workspace was evacuated before VM and system-disk deletion.

No Nebius resource is needed while the API budget remains unavailable. The next cloud action is restoration and exact benchmark resume, not retraining.

## Safety invariants

- Nebius resources are created manually after an explicit price and budget check.
- Use one H100-class CUDA VM with enough system-disk capacity for the restored workspace.
- Keep public IPs, SSH keys, local backup paths, and credentials out of Git.
- Transfer VM payloads only with `rsync`.
- Run long jobs inside logged `screen` sessions.
- Never regenerate final candidates or change the final campaign identifier.
- Verify the evacuated copy before deleting any cloud resource.
- Guest shutdown can precede the Nebius console state update. Confirm `STOPPED` in the control plane.

## Archived state required for resume

The restored `/data/laya-posttrain` tree must contain at least:

```text
repo/reports/benchmark-v002bis-final/checkpoint.json
repo/data/candidate_cache/openrouter-openai_gpt-5.6-sol-benchmark-v002bis-final.json
repo/data/trajectories.sqlite3
repo/logs/
checkpoints/laya-dynamics-v002bis/model.safetensors
base-english/model.safetensors
venv/
```

The benchmark checkpoint must report 705 completed episodes. The v2bis model SHA-256 is:

```text
19eb33a1a2ad62e019325b29f3796b2e507fb4723ca6bd4ac22b3de726458fbf
```

`resume.pt` is not expected under the completed v2bis export. Benchmark recovery uses `checkpoint.json`. The large `laya-dynamics-v002/resume.pt` belongs to the interrupted v2 training and must never be copied into the v2bis directory.

## Create and restore a VM

Check the current Nebius price and available balance. Create an Ubuntu NVIDIA H100 VM with a system disk large enough for the complete workspace. A separate data disk is optional. The validated deployment used the persistent system disk directly.

Record the current public IP only in the shell:

```bash
VM_HOST=USER@CURRENT_VM_IP
SSH_KEY=/path/to/private_key
BACKUP_ROOT=/path/to/evacuated/laya-posttrain
```

Install the transfer and terminal tools if needed:

```bash
ssh -i "$SSH_KEY" -o IdentitiesOnly=yes "$VM_HOST" '
  sudo apt-get update
  sudo apt-get install -y rsync screen
  sudo mkdir -p /data/laya-posttrain
  sudo chown "$USER":"$USER" /data/laya-posttrain
'
```

Restore the complete workspace:

```bash
RSYNC_RSH="ssh -i $SSH_KEY -o IdentitiesOnly=yes" \
rsync -aH --partial "$BACKUP_ROOT/" "$VM_HOST:/data/laya-posttrain/"
```

Do not use `scp`, archive streaming, or a newly generated dataset.

## Preflight

```bash
ssh -t -i "$SSH_KEY" -o IdentitiesOnly=yes "$VM_HOST"
cd /data/laya-posttrain/repo
scripts/nebius_preflight.sh
```

Then verify:

```bash
jq -r '[.status, (.results|length), .results[-1].run_id] | @tsv' \
  reports/benchmark-v002bis-final/checkpoint.json

sha256sum ../checkpoints/laya-dynamics-v002bis/model.safetensors
```

Require CUDA, the expected model hash, the frozen dataset hashes, the final cache, and exactly 705 completed episodes before spending API credit.

## Resume the final campaign

Add sufficient OpenRouter credit or raise the key's total limit first. Keep the key in the process environment:

```bash
cd /data/laya-posttrain/repo
export OPENROUTER_API_KEY=your_key_here
export OPENROUTER_MODEL=openai/gpt-5.6-sol
export OPENROUTER_MAX_TOKENS=4096

scripts/nebius_runtime_screen.sh
screen -r laya-runtime
```

Detach with `Ctrl+A`, then `D`. The launcher records the original human-readable stream under `/data/laya-posttrain/logs/runtime-*.log`.

The runtime script rechecks smoke and challenge from their checkpoints, then resumes `benchmark-v002bis-final`. Completed episodes are printed as `checkpoint replay`. The first live request must retry episode 706. If the configuration differs from the saved checkpoint, resume fails closed.

The launcher requests guest shutdown after success or failure. SSH can become unavailable before Nebius changes the instance state from `RUNNING` to `STOPPED`. Wait for the console transition before deciding that shutdown failed.

## Training procedure for a future independent recipe

The original v2bis training commands remain available for audit:

```bash
NEBIUS_TRAIN_APPROVED=YES scripts/nebius_screen.sh smoke
NEBIUS_TRAIN_APPROVED=YES MAX_WALL_SECONDS=21600 scripts/nebius_screen.sh full
```

Do not run them for the current checkpoint. Any future training iteration requires a new dataset version, new checkpoint name, new hidden final suite, explicit budget approval, and an updated preregistration.

## Evacuate and teardown

Synchronize the complete workspace, not only the model:

```bash
RSYNC_RSH="ssh -i $SSH_KEY -o IdentitiesOnly=yes" \
rsync -aH --partial "$VM_HOST:/data/laya-posttrain/" "$BACKUP_ROOT/"

RSYNC_RSH="ssh -i $SSH_KEY -o IdentitiesOnly=yes" \
rsync -aHnci "$VM_HOST:/data/laya-posttrain/" "$BACKUP_ROOT/"
```

The checksum dry run must report no differences. Verify the model hash, final checkpoint count, cache, logs, reports, and databases locally. Then delete the VM, system disk, snapshots, images, and static IPs. Confirm in Nebius that no billable resource remains.
