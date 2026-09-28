# Nebius Laya post-training runbook

## Scope

This procedure runs domain post-training on one preemptible NVIDIA H100. It never creates Nebius resources. VM, disk and IP creation remain deliberate console actions because they are billable. Training follows Laya's published single-GPU RLCD recipe at pinned revision `9d955671415fc19f069b9cc998928075c1f255ec`.

## Safety invariants

- `final-v001` identifiers are rejected during generation and preprocessing.
- Raw splits and resume checkpoints are bound to SHA-256 hashes.
- Development templates are disjoint from training templates. Calibration receives no gradient updates.
- Paid training requires `NEBIUS_TRAIN_APPROVED=YES`.
- Atomic checkpoints are written to persistent storage every 600 seconds and on graceful interruption.
- The VM shuts down after completion, failure or interruption when `AUTO_SHUTDOWN=1`.
- A stopped VM can still incur storage and IP charges. Evacuate, verify, then delete every resource.

Before restarting any paid VM, require green tests and a completed local validation benchmark whose report contains wall-clock, fresh/cache provenance, component timings, paired ratios and reconstruction-completeness fields. Record the gate summary before training.

## 1. Freeze the local dataset

```bash
cd near_agi

PYTHON_BIN="${PYTHON_BIN:-python}" \
  scripts/build_dataset_v2bis.sh

python -m json.tool configs/train/v002bis.dataset-manifest.json
```

Dataset generation requires `OPENROUTER_API_KEY` and uses `openai/gpt-5.6-sol`. GPT generates five contrastive candidate actions at each reachable anchor state. Every candidate is executed on an independent simulator clone; GPT never supplies a label. The planned build uses 200/30/30/60 scenario groups and five anchors per group, approximately 1,600 planner calls and 48,000 labeled question sequences when every response contains five unique candidates. The exact committed manifest is authoritative.

Current and historical action identifiers are excluded from model input but retained in audit records. Scenario groups cannot cross splits, and development uses separate template families. Generation fails if tool or outcome diversity gates are not met. Do not freeze or stage the corpus until the manifest, usage, candidate cache and manual trajectory audit pass. Do not rebuild it after training begins.

## 2. Create resources manually

Check the live Nebius price immediately before creation. Create one preemptible H100 in `eu-north1` with an Ubuntu 24.04 NVIDIA image and a persistent 100 GiB system disk. This setup keeps the system disk after compute shutdown; create `/data/laya-posttrain` on that filesystem. A separate network disk is optional, not required.

The compute ceiling is $16 before tax. The console quote recorded on 2026-09-28 is $2.16 per hour for one H100. The configured six-hour maximum costs $12.96 and leaves $3.04 for bootstrap time and variance. Nebius billing is authoritative.

Do not create a reusable image or snapshot.

## 3. Stage and preflight

After a stopped VM is restarted, obtain its current status and public IP from Nebius before SSH. Keep the IP in a shell variable only. Never commit it. Verify or install the transfer and terminal tools on the VM, then stage from the local machine:

```bash
VM_HOST=USER@CURRENT_VM_IP
SSH_KEY=/path/to/private_key

ssh -i "$SSH_KEY" -o IdentitiesOnly=yes "$VM_HOST" '
  if ! command -v rsync >/dev/null || ! command -v screen >/dev/null
  then
    sudo apt-get update
    sudo apt-get install -y rsync screen
  fi
'

RSYNC_RSH="ssh -i $SSH_KEY -o IdentitiesOnly=yes" \
  scripts/nebius_stage.sh "$VM_HOST"
```

`nebius_stage.sh` transfers every payload with `rsync`; it does not use `scp` or archive streaming. SSH remains the interactive control channel for bootstrap and `screen`. Configure the SSH key explicitly as above or load it into `ssh-agent`.

On the VM:

```bash
ssh -i SSH_KEY -o IdentitiesOnly=yes USER@VM_IP
cd /data/laya-posttrain/repo
scripts/nebius_bootstrap.sh
scripts/nebius_preflight.sh
```

Preflight verifies CUDA, at least 40 GiB VRAM, 50 GiB free disk, dataset hashes and the base checkpoint. It does not train.

## 4. Three-step paid smoke

After checking the live price and balance:

```bash
cd /data/laya-posttrain/repo
NEBIUS_TRAIN_APPROVED=YES scripts/nebius_screen.sh smoke
screen -r laya
```

The attached screen displays the original pipeline output. It labels preflight, preprocessing, training, calibration, evaluation and promotion-gate phases. Training progress is formatted as concise human-readable lines with step, epoch, loss, cross-entropy, throughput, peak VRAM and elapsed time. Detach without stopping the job with `Ctrl+A`, then `D`, and reattach later with `screen -r laya`.

The complete original terminal stream is also recorded under `/data/laya-posttrain/logs/smoke-*.log`. The structured `/data/laya-posttrain/checkpoints/laya-dynamics-v002bis/train-log.jsonl` remains the audit source. After the session finishes, require finite loss, safe peak VRAM and a valid `resume.pt`. This smoke deliberately pauses before calibration. Never copy or resume `/data/laya-posttrain/checkpoints/laya-dynamics-v002/resume.pt` into this directory: its dataset hash and model representation belong to the interrupted v2 run.

## 5. Production run

The same output directory resumes the accepted smoke:

```bash
cd /data/laya-posttrain/repo
export OPENROUTER_API_KEY=your_key_here
export OPENROUTER_MODEL=openai/gpt-5.6-sol
NEBIUS_TRAIN_APPROVED=YES MAX_WALL_SECONDS=21600 scripts/nebius_screen.sh full
screen -r laya
```

The pipeline preprocesses once, trains four epochs, calibrates on the dedicated split, verifies the offline promotion gate, then runs checkpointed four-arm smoke and challenge gates. Only two passes unlock the single `final-v001` campaign. It writes `SHA256SUMS` and powers off after success, gate failure, interruption or error.

Each benchmark writes `reports/<campaign>/checkpoint.json` atomically after every completed episode. Re-running the full screen command resumes the fixed v2bis campaign, reuses its campaign-scoped candidate cache, skips completed episodes and restarts only an interrupted episode. The OpenRouter key remains in the process environment and is never written to a repository file or checkpoint.

After preemption, attach the same disk to a compatible H100 VM, mount it at `/data`, and rerun the identical command. Dataset or seed mismatches fail closed.

## 6. Evacuate and teardown

From the local machine:

```bash
scripts/nebius_evacuate.sh ubuntu@VM_IP
```

Set `LAYA_EVACUATE_TO` to an external local storage directory. If it is unset, artifacts are verified under the ignored local `checkpoints/` directory.

After local verification, delete the VM, network disk, every image or snapshot, and any static public IP. Confirm that the Nebius project has no billable resources.

## 7. Promotion

Evaluate base and fine-tuned checkpoints on the disjoint development set and visible challenge suite first. Promote only after lower transition error, fewer unnecessary actions, zero unsafe-action regression and the pre-registered development success threshold. Only then add the fine-tuned benchmark arm and run the single compliant `final-v001` comparison.
