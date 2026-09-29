# Implementation plan

## Current objective

The implemented POC tests whether a specialised Laya transition predictor can select GPT-generated actions more reliably than base Laya while preserving fast local selection. The sandbox is the source of transition truth. Laya predicts outcomes and never executes tools.

The active experiment is the registered four-arm `final-v001` campaign. It is paused at 705 of 1,800 episodes because the OpenRouter budget was exhausted. Its campaign checkpoint, candidate cache, logs, reports, databases, environment, and model have been evacuated for an exact future resume.

## Implemented system

- deterministic sandbox with typed `navigate`, `answer`, and `observe` actions;
- direct GPT and GPT candidate-generation planners;
- campaign-scoped candidate cache with generation provenance;
- heuristic, base-Laya, and fine-tuned-Laya candidate selectors;
- explicit utility policy and loop guards;
- SQLite plus JSONL transition persistence;
- atomic episode-level campaign checkpoints and resume;
- component, selector, observed wall, and reconstructed end-to-end latency telemetry;
- smoke, challenge, and frozen `final-v001` suites;
- v2bis executed-counterfactual corpus, training, calibration, and promotion pipeline;
- Nebius staging, preflight, `screen` logs, graceful checkpoints, evacuation, and shutdown.

## Experimental boundary

`final-v001` contains 90 tasks from three held-out templates and uses seeds `0,1,2,3,4`. Four arms produce 1,800 planned episodes:

1. `direct_gpt`
2. `candidates_heuristic`
3. `candidates_base_laya`
4. `candidates_finetuned_laya`

Candidate-based arms share byte-equivalent lists for identical states. Direct GPT is generated independently. The final report must be complete and protocol-compliant before any research conclusion is made.

## Completed stages

1. Freeze schemas, hashes, protocols, allowlist, and compact formatting.
2. Implement the sandbox, deterministic labels, storage, planners, policies, and Laya adapter.
3. Add cache-aware latency instrumentation and paired reporting.
4. Freeze `final-v001` before v2bis training.
5. Record v1 as a runtime-transfer failure and interrupt v2 after identifying action-ID leakage.
6. Generate and freeze the v2bis executed-counterfactual corpus.
7. Train and calibrate v2bis on one Nebius H100.
8. Correct tie-aware selection evaluation without changing weights.
9. Pass four-arm smoke and challenge runtime gates.
10. Start the registered final campaign and preserve its resumable state after API-budget exhaustion.

## Next stages

1. Restore the evacuated workspace when OpenRouter budget is available.
2. Resume the same campaign identifier, cache, checkpoints, prompts, model, and five seeds.
3. Verify the 1,800-episode report and protocol-compliance verdict.
4. Analyse paired success, steps, safety, unnecessary actions, prediction quality, selector latency, observed latency, and reconstructed end-to-end latency.
5. Publish either the positive or negative bounded-domain result without tuning against final-v001.
6. Design a new external computer-use benchmark before making broader claims.

Parallel episode pools are future engineering work. The sequential implementation remains the reference protocol. A parallel mode must reproduce actions and quality metrics against `--workers 1`, preserve paired candidate cache semantics, and report latency under contention separately.

Shopify integration, OAuth, DAgger, browser environments, and latent rollouts remain outside the current registered experiment.
