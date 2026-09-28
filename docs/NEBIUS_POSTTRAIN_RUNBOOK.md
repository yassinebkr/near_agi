# Nebius Laya post-training runbook

## Scope

This procedure runs domain post-training on one preemptible NVIDIA H100. It never creates Nebius resources; VM, disk and IP creation remain deliberate console actions because they are billable. Training follows Laya's published single-GPU RLCD recipe at pinned revision `9d955671415fc19f069b9cc998928075c1f255ec`.

## Safety invariants

- `final-v001` identifiers are rejected during generation and preprocessing.
- Raw splits and resume checkpoints are bound to SHA-256 hashes.
- Development templates are disjoint from training templates; calibration receives no gradient updates.
- Paid training requires `NEBIUS_TRAIN_APPROVED=YES`.
- Atomic checkpoints are written to persistent storage every 600 seconds and on graceful interruption.
- The VM shuts down after completion, failure or interruption when `AUTO_SHUTDOWN=1`.
- A stopped VM can still incur storage and IP charges. Evacuate, verify, then delete every resource.

## 1. Freeze the local dataset

```bash
cd /home/kwestog/Documents/code/near_agi

PYTHON_BIN=/mnt/fast-ssd/laya-dynamics-agent/.venv-cu124/bin/python \
  scripts/build_dataset.sh

python -m json.tool configs/train/v001.dataset-manifest.json
```

The raw build contains 36,000 training, 5,400 validation, 5,400 calibration and 10,800 unseen-template development sequences. Deterministic rare-label oversampling expands only the training split to 58,800 sequences; the evaluation splits keep their natural distribution. Do not rebuild it after training begins.

## 2. Create resources manually

Check the live Nebius price immediately before creation. Create one preemptible H100 in `eu-north1`, an Ubuntu 24.04 NVIDIA image, and a 100 GiB persistent network SSD mounted at `/data`.

The compute ceiling is $16 before tax. The console quote recorded on 2026-09-28 is $2.16 per hour for one H100. The configured six-hour maximum costs $12.96 and leaves $3.04 for bootstrap time and variance. Nebius billing is authoritative.

Do not create a reusable image or snapshot.

## 3. Stage and preflight

From the local machine:

```bash
scripts/nebius_stage.sh ubuntu@VM_IP
```

`nebius_stage.sh` uses `rsync`; install it on both endpoints if either machine lacks it. Configure the SSH key in `~/.ssh/config` or load it into `ssh-agent` before staging.

On the VM:

```bash
ssh ubuntu@VM_IP
screen -RR laya
cd /data/laya-posttrain/repo
scripts/nebius_bootstrap.sh
scripts/nebius_preflight.sh
```

Preflight verifies CUDA, at least 40 GiB VRAM, 50 GiB free disk, dataset hashes and the base checkpoint. It does not train.

## 4. Three-step paid smoke

After checking the live price and balance:

```bash
cd /data/laya-posttrain/repo

NEBIUS_TRAIN_APPROVED=YES \
AUTO_SHUTDOWN=0 \
MAX_STEPS=3 \
scripts/nebius_run.sh
```

Inspect `/data/laya-posttrain/checkpoints/laya-dynamics-v001/train-log.jsonl`. Require finite loss, safe peak VRAM and a valid `resume.pt`. This smoke deliberately pauses before calibration.

The training terminal prints concise human-readable progress. The JSONL file remains the machine-readable source of record. From a second local terminal, follow it live with:

```bash
ssh -i SSH_KEY -o IdentitiesOnly=yes USER@VM_IP \
  'tail -n 20 -F /data/laya-posttrain/checkpoints/laya-dynamics-v001/train-log.jsonl'
```

For a readable rendering of the same events when `jq` is installed locally:

```bash
ssh -i SSH_KEY -o IdentitiesOnly=yes USER@VM_IP \
  'tail -n 20 -F /data/laya-posttrain/checkpoints/laya-dynamics-v001/train-log.jsonl' \
  | jq -r '"[train] \(.event) | step=\(.step // "-") | epoch=\(.epoch // "-") | loss=\(.loss // "-") | peak_vram=\(.peak_reserved_gib // "-") GiB | elapsed=\((.elapsed_seconds // 0) | floor)s"'
```

## 5. Production run

The same output directory resumes the accepted smoke:

```bash
cd /data/laya-posttrain/repo

NEBIUS_TRAIN_APPROVED=YES \
AUTO_SHUTDOWN=1 \
MAX_STEPS=0 \
MAX_WALL_SECONDS=21600 \
scripts/nebius_run.sh
```

The pipeline preprocesses once, trains four epochs, calibrates on the dedicated split, removes inherited option-bucket temperatures, verifies the checkpoint, writes `SHA256SUMS`, and powers off.

After preemption, attach the same disk to a compatible H100 VM, mount it at `/data`, and rerun the identical command. Dataset or seed mismatches fail closed.

## 6. Evacuate and teardown

From the local machine:

```bash
scripts/nebius_evacuate.sh ubuntu@VM_IP
```

The default verified destination is `/mnt/fast-ssd/laya-dynamics-agent/checkpoints/laya-dynamics-v001`.

After local verification, delete the VM, network disk, every image or snapshot, and any static public IP. Confirm that the Nebius project has no billable resources.

## 7. Promotion

Evaluate base and fine-tuned checkpoints on the disjoint development set and visible challenge suite first. Promote only after lower transition error, fewer unnecessary actions, zero unsafe-action regression and the pre-registered development success threshold. Only then add the fine-tuned benchmark arm and run the single compliant `final-v001` comparison.
