# Evaluation protocol (pre-registered draft)

## Experimental contract

The primary live comparison uses one exact model id for both GPT arms: direct GPT action selection versus GPT candidate generation followed by a local transition selector. Candidate lists are cached by provider/model, prompt version, seed, candidate limit, and canonical state hash. Cache schema v2 stores generation timestamp, latency, provider, model, state hash, and original token/cost usage. Legacy entries remain replayable but are explicitly reported as having unknown usage provenance. Therefore heuristic and Laya selectors receive byte-equivalent candidates whenever they reach the same state. A direct GPT action is generated independently and is not mislabeled as “the first candidate”.

Clean live campaigns must use `--fresh-candidate-cache`. This selects a new campaign-scoped cache without deleting previous evidence. Cache hits can still occur inside that campaign when the heuristic and Laya arms reach the same canonical state. These internal hits are required for a paired comparison and are not contamination from an earlier campaign.

The smoke suite currently contains three deterministic tasks from three templates. It validates plumbing only. It cannot support a research claim. A claim requires untouched template-level test splits, at least 30 task instances per held-out template, and 5 declared seeds. Live model nondeterminism remains possible even when a replicate is labelled by a seed. Cached responses make replay exact.

The `challenge` suite contains all nine current tasks across eight templates. It broadens fact formats and distractors: conflicting community claims, archived values, official specifications or policies, safety-sensitive wording, and simulated irreversible actions. It is a development benchmark, not a held-out test set, because its tasks live in the repository and influence implementation.

## Final-v001 frozen evaluation

`final-v001` contains 90 instances across three templates absent from the development suites: `heldout-qualified-limit-v1`, `heldout-current-policy-v1`, and `heldout-security-threshold-v1`. Each template contributes exactly 30 instances. The registered seeds are `0,1,2,3,4`. OpenRouter receives each seed for direct and candidate-generation calls, although provider determinism is not assumed to be bit-exact.

A compliant final report requires the `final` suite, all five registered seeds, a fresh campaign-scoped cache, both base and fine-tuned Laya enabled, a live provider, and complete execution. Reports record `final_protocol.compliant`, `suite_version`, and `task_manifest_sha256`, plus standard deviations, deterministic 95% bootstrap intervals, per-template results, and paired success and step deltas.

Before the single-use final campaign, a fine-tuned checkpoint must pass both the live four-arm runtime smoke and visible challenge suites through the same `LayaPredictor` API used by final evaluation. Offline tensor-level development scores alone cannot promote a checkpoint. Runtime failure closes the gate without exposing final-v001.
The automated v2 pipeline uses stable campaign identifiers and atomically checkpoints each completed episode. Smoke and challenge require a complete four-arm report, fine-tuned success no worse than the best control, zero fine-tuned unsafe actions, and no increase in unnecessary actions relative to base Laya. Any failure stops before the next suite.

## Arms

- `direct_gpt`: the configured GPT model chooses one action directly.
- `candidates_heuristic`: the configured GPT model generates candidates. A transparent heuristic scores them.
- `candidates_base_laya`: the same cached candidates are scored by the unmodified local Laya checkpoint.
- `candidates_finetuned_laya`: the same cached candidates are scored by the promoted `laya-dynamics-v002` checkpoint.

The deterministic provider supplies offline stand-ins for wiring tests and must never be reported as GPT or Laya performance.

## Frozen metadata and latency measurements

Every episode records campaign/run/task/template identifiers, seed, provider, exact model, labeler version, prompt version, cache path, actions, outcomes, usage and reconstructable transitions in SQLite plus JSONL. The durable episode summary and `metrics.json` also contain the following latency contract:

- `wall_clock_ms`: observed time from immediately before environment reset through the completed episode.
- `planner_wall_ms`: complete wall time inside `planner.propose_actions`.
- `candidate_generation_ms`: provider generation time on fresh candidate calls and zero on replay.
- `candidate_cache_lookup_ms`: key lookup plus candidate deserialization, contained within planner wall time.
- `predictor_wall_ms`: wall time around the complete predictor call.
- `predictor_reported_ms`: sum of per-action timings reported inside the predictor.
- `policy_ms` and `environment_ms`: explicit selection and environment timings. `environment_ms` includes reset.
- `selector_wall_ms`: predictor plus policy wall time for candidate-based arms.
- `framework_overhead_ms`: non-negative remainder after subtracting planner, predictor, policy and environment wall components. It includes trajectory persistence and orchestration overhead.

Candidate generation and lookup are subdivisions of planner wall time and are never added to the component budget. Every candidate step records `candidate_source` as `fresh` or `cache`, its cache key, generation provider/model and original generation latency when available. Episodes report fresh and replay counts and classify the candidate path as cold, warm or mixed.

Selector performance asks how quickly already available candidates pass through the predictor and policy. End-to-end performance covers observation, any GPT candidate generation, prediction, selection, environment transitions and eventual success. A warm replay cannot by itself establish a cold-start system speedup. When every replayed cache entry contains its original generation latency, `effective_end_to_end_ms_reconstructed` equals observed episode wall time plus those replayed generation latencies. The observed measurement is retained, the reconstruction is labelled, and incomplete legacy provenance produces `null` with `effective_end_to_end_complete=false`.

Per-mode reports include mean, median/p50, p90 and p95 wall time plus a bootstrap 95% interval and mean component timings. Paired time-to-success deltas and left/right ratios use only pairs where both arms succeed and both durations are positive and finite. All terminal outcomes are summarized separately, so a quick failure is never presented as a speedup. Fixed ratio buckets are `<=0.25x`, `0.25–0.5x`, `0.5–0.8x`, `0.8–1.25x`, `1.25–2x`, `2–3x` and `>3x`. A ratio below one means the left arm is faster. Observed warm/cold ratios and reconstructed end-to-end ratios remain separate.

Report task success, mean/median steps, unnecessary actions, loops, tool failures, simulated unsafe actions, provider token/cost fields, all latency fields and prediction MAE. For a full held-out campaign add mean/median/std and bootstrap confidence intervals, peak RAM/VRAM, balanced accuracy, Brier score, log loss, ECE, and reliability plots where outputs are calibrated probabilities.

## Decision rule

The specialised model is interesting only if held-out task success is no worse than `direct_gpt`, unnecessary actions decrease, API tokens do not materially increase, and fine-tuned Laya beats base Laya on transition prediction and task outcomes. Evaluation compares base Laya, fine-tuned Laya, direct GPT and the heuristic on success, steps, safety, unnecessary actions, prediction quality, selector latency, observed wall latency, reconstructed end-to-end latency and paired latency ratios. Fine-tuning must be allowed to remove catastrophic slow or failing cases while preserving or improving states where base Laya is already very fast. The current hypothesis tests whether one specialised Laya can excel across the target domain. GPT/Laya fast-path routing remains future work only if post-training still reveals heterogeneous weaknesses. Raw results and confidence intervals accompany every conclusion. This rule is fixed before training.

Ablations cover history, beliefs, information gain, risk, base versus fine-tuned Laya, 1/3/5/8 candidates, raw versus structured formatting, and a cheaper GPT model versus the main `openai/gpt-5.6-sol` campaign.
