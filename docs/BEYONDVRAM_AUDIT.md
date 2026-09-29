# BeyondVRAM integration audit

Source: https://github.com/yassinebkr/beyondvram

Audited commit: `28d0a005b72f46b1dc360fcda1afc010f2766e14` on 2026-09-28.

## Reused practices

BeyondVRAM supplied the operational pattern for the Nebius work:

- deterministic corpus manifests and held-out split discipline;
- resumable optimizer, scheduler, RNG, and data-cursor state;
- periodic and SIGTERM-triggered checkpoints;
- persistent outputs and checksum verification before teardown;
- explicit cost, wall-time, environment, and artifact reporting;
- fail-closed gates and automatic guest shutdown.

These practices were adapted to Laya's full-parameter RLCD training loop. The v2bis run used a persistent Nebius system disk, a logged `screen` session, ten-minute resumable checkpoints, a frozen dataset manifest, post-training calibration, offline promotion, runtime gates, and rsync evacuation.

## Boundary

BeyondVRAM's local over-VRAM inference work and its Nebius post-training work are separate. This project does not claim that Laya training spills transparently to NVMe, and it does not attribute a Nebius inference experiment to BeyondVRAM. The Laya objective, preprocessing, calibration, and checkpoint format come from the upstream Laya implementation.

The interrupted v2 run and completed v2bis run confirmed that the reusable contribution is operational discipline rather than a drop-in trainer. Dataset representation and evaluation validity remained project-specific responsibilities.

## Outcome of the adaptation

The completed v2bis training ran 5,648 updates and 46,371,505 tokens on one H100. It exported a standard Laya checkpoint, passed the corrected offline gate, then passed four-arm smoke and challenge runtime gates. The frozen final campaign is paused at 705 of 1,800 episodes because the OpenRouter budget was exhausted. Its complete state was evacuated before cloud teardown.
