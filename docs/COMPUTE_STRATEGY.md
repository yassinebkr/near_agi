# Compute strategy

Hardware placement must preserve the same datasets, prompts, seeds, caches, checkpoints, and evaluation rules. Compute changes execution cost and throughput. It does not change the experimental contract.

## Validated execution scopes

- **Local development:** deterministic tests, dataset audits, CUDA inference smoke, report inspection, and checkpoint loading use the RTX 3070 Ti 8 GiB machine.
- **Nebius training:** the completed v2bis full-parameter run used one H100 80 GB because it provided an available and economical path for the frozen recipe.
- **Nebius runtime evaluation:** smoke, challenge, and the resumable final campaign ran beside the trained checkpoint on the same H100. This is now an explicit evaluation backend, and every report records its device and timing context.
- **Offline storage:** complete cloud workspaces are evacuated with rsync and verified before VM and disk deletion. Large datasets, caches, reports, and checkpoints remain outside Git.

Local and Nebius latency measurements belong to different hardware contexts. They may be compared within a campaign when all arms share that context. They must not be combined as if hardware were identical.

## Portability contract

Every training job consumes a frozen dataset manifest and training config. It records:

- source revision and dirty-state flag;
- base checkpoint and dataset hashes;
- Python, Laya, PyTorch, CUDA, and driver versions;
- hardware, duration, throughput, peak VRAM, and interruption state;
- seed, precision, batch, accumulation, checkpoint, and optimizer settings;
- exported checkpoint hash, calibration, evaluation, and gate reports;
- provider usage and cloud cost when applicable.

Every runtime campaign additionally records exact provider/model, prompt versions, task-manifest hash, Laya checkpoints, seeds, cache provenance, component latency, and atomic resume state.

## Current state

The v2bis training and runtime transfer gates are complete. `final-v001` is paused after 705 of 1,800 episodes because the OpenRouter key budget was exhausted. The cloud workspace and model were evacuated. No GPU VM is required until API budget is available to resume the same campaign.

Future compute is provisioned only after an explicit cost check and approval. A new backend must pass preflight and a small reproducibility comparison before it can replace a validated backend.
