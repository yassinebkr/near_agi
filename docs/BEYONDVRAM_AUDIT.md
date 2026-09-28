# BeyondVRAM integration audit

Source: https://github.com/yassinebkr/beyondvram

Audited commit: `28d0a005b72f46b1dc360fcda1afc010f2766e14` (2026-09-28).

## Reusable work

BeyondVRAM is directly relevant as an experimental and operations reference: it targets the same RTX 3070 Ti 8 GiB / 32 GiB RAM class, records failures rather than estimating them, pins artifacts, and contains a measured Nebius training workflow. Its Track 5 contributes useful patterns for this project:

- deterministic corpus filtering and held-out split discipline;
- resumable training state (adapter, optimizer, scheduler and data cursor);
- periodic and SIGTERM-triggered checkpoints for preemptible instances;
- persistent/network-volume outputs and evacuation before teardown;
- explicit cost, wall-time, environment and artifact reporting;
- cloud gates and automatic shutdown to limit idle cost.

## Important boundary

BeyondVRAM has two distinct bodies of work that must not be conflated. Its over-VRAM inference, quantization, placement and measurement experiments run locally. Separately, Nebius was used only for fine-tuning/post-training: a model-specific bf16 LoRA job on a large-VRAM GPU. No BeyondVRAM inference run on Nebius is claimed. The repository explicitly records its attempted local QLoRA path for a 30B model as infeasible on 32 GiB RAM because CPU-dispatched modules remained fp32.

Therefore it is not a drop-in Laya trainer or proof that Laya fine-tuning can spill transparently to NVMe. We reuse the research discipline, checkpoint/resume lifecycle and Nebius operations. Laya training itself must adapt the official Laya decision-head/RLCD notebook and be validated first with a one-step smoke run.

## Planned adaptation

1. Keep Laya's official preprocessing, decision loss and temperature fitting.
2. Port BeyondVRAM's time-based/SIGTERM checkpoint contract around that loop.
3. Write all mutable cloud outputs to a persistent volume.
4. Pin source, base checkpoint, dataset and container revisions.
5. Run a tiny local training smoke before provisioning cloud compute.
6. If local training is impractical, run the unchanged experiment on an explicitly authorized Nebius instance and evacuate verified artifacts to `/mnt/fast-ssd/laya-dynamics-agent/`.
