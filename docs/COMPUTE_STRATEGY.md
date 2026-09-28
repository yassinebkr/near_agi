# Compute strategy

Local VRAM is a routing constraint, not a blocker for Milestone 2. The training pipeline must produce the same dataset manifest, configuration, seeds, metrics, and checkpoint manifest regardless of execution backend.

## Separate execution scopes

1. **Phase 1 inference and benchmarks: local only** — run the complete Laya checkpoint on the RTX 3070 Ti and record latency, VRAM and RAM. Nebius is not an inference backend for this project.
2. **Milestone 2 training smoke: local** — validate preprocessing, loss, one optimizer step, checkpoint writing and reload on the smallest useful sample.
3. **Milestone 2 full fine-tuning/post-training: Nebius when required** — reuse BeyondVRAM's measured cloud-training lifecycle when the reference run does not fit locally or would take unreasonably long. Provisioning remains an explicit, user-authorized billable action.

BeyondVRAM also contains local inference/offload research, but that work is not presented as a Laya training backend and is not the reason Nebius is included here.

## Portability contract

Every job consumes a frozen dataset hash plus `configs/train/*.yaml` and emits:

- base repository, revision, and weight SHA-256;
- source Git commit and dirty-state flag;
- exact Python, Laya, PyTorch, CUDA, driver, and container/image versions;
- hardware topology, duration, peak VRAM/RAM, and failure/OOM record;
- seed, optimizer, precision, batch/accumulation, checkpointing and offload settings;
- checkpoint SHA-256, calibration temperatures, raw validation/calibration/test metrics;
- provider and instance type, with cost when a cloud backend is used.

Cloud output must be copied back to `/mnt/fast-ssd/laya-dynamics-agent/` and verified by checksum before the instance is stopped. Secrets, provider credentials, generated datasets, and large checkpoints never enter Git.

## Decision rule

First reproduce a tiny official-style training step locally. Then estimate the complete run from measured memory and throughput. Use Nebius only for fine-tuning/post-training if the local run is impossible or its projected wall time is unreasonable. Training location must not change splits or evaluation criteria. All inference evaluation remains local unless a future protocol explicitly introduces and justifies another inference environment.

## Audited integration

BeyondVRAM was audited at commit `28d0a005b72f46b1dc360fcda1afc010f2766e14`; see `docs/BEYONDVRAM_AUDIT.md`. For Nebius we reuse only its verified fine-tuning/post-training operations: persistent outputs, resumable checkpoints, SIGTERM handling, evacuation, cost tracking and teardown. We do not attribute any Nebius inference experiment to BeyondVRAM.

