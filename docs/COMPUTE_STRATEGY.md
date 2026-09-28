# Compute strategy

Local VRAM is a routing constraint, not a blocker for Milestone 2. The training pipeline must produce the same dataset manifest, configuration, seeds, metrics, and checkpoint manifest regardless of execution backend.

## Execution tiers

1. **Local native GPU** — preferred for inference, smoke training, profiling, and the smallest reproducible run on the RTX 3070 Ti 8 GB.
2. **Local Beyond VRAM** — reuse the user's existing GitHub work for measured CPU/RAM/NVMe offload experiments. Integration starts only after its repository URL and supported training contract are recorded; no unverified assumption is made that an inference offload engine supports Laya backpropagation.
3. **Nebius cloud** — optional for the reference fine-tune when local execution is too slow or cannot fit optimizer/activation state. Provisioning is an explicit, user-authorized operation because it creates billable resources.

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

First reproduce a tiny official-style training step locally. Then benchmark local native versus the verified Beyond VRAM path. Use Nebius only if local execution is impossible or its projected wall time is unreasonable. Backend choice must not change splits or evaluation criteria.

## Open integration item

Record the exact URL and commit of the user's Beyond VRAM repository before adapting it. Its APIs and numerical guarantees have not yet been inspected in this repository.

