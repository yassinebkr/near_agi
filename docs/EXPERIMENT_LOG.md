# Experiment log

## Final-v001 pre-registration freeze

The final matrix was frozen before live execution: 90 tasks, three held-out templates, 30 instances per template, and seeds `0,1,2,3,4`. A compliant report must use a fresh cache, CUDA base Laya, a live provider, and complete all 1,350 episodes. Each report fingerprints the canonical task manifest with SHA-256. A one-seed pilot is operational validation only and is excluded from final inference.

## Historical latency limitation

The campaigns below predate durable episode wall-clock and cache-provenance telemetry. Candidate-based arms shared a `CachedPlanner`, so the first selector reaching a state could pay the GPT generation latency while another selector replayed the same candidates. Their recorded `predictor time` measures predictor-internal work only. it is neither selector-path wall time nor cold-start end-to-end time.

Terminal observations suggested substantial heterogeneity: some base-Laya episodes appeared two to three times slower or failed, while others appeared roughly three to six times faster than direct GPT. Those observations motivated the current instrumentation, but they cannot support a retrospective speedup distribution. Historical values remain unchanged, and no definitive end-to-end latency conclusion is drawn from them. New campaigns separately report observed wall time, fresh/replayed provenance and reconstructed effective end-to-end latency when original generation timing is complete.

## benchmark-9407b961 · latency instrumentation validation

Date: 2026-09-28. Provider: deterministic fixture. Suite: smoke. Seed: 0. Device: CUDA for base Laya. This nine-episode run validates telemetry only and carries no claim about GPT or Laya speed.

The campaign recorded seven fresh candidate generations and four cache replays. All nine SQLite run rows contain `summary_json`, every transition records candidate provenance and component timings, and every episode budget closes as `wall_clock_ms = planner_wall_ms + predictor_wall_ms + policy_ms + environment_ms + framework_overhead_ms` within floating-point precision. Reconstructed latency was complete for all six candidate-based episodes. Time-to-success correctly excluded the two failed Laya pairs.

The deterministic fixture makes provider generation nearly instantaneous, while synchronous SQLite/JSONL persistence appears as roughly 129–144 ms mean framework overhead per episode. This is expected and exposes the measurement boundary rather than a timing defect. The run also contains one mixed Laya path caused by a divergent state, which validates simultaneous fresh and replay accounting. No speedup conclusion is drawn.

## benchmark-795a36f9 · challenge · seed 0

Date: 2026-09-28. Provider: OpenRouter. Model: `openai/gpt-5.6-sol`. Laya checkpoint: `/mnt/fast-ssd/models/laya/base-english`. Device: CUDA. Candidate cache: fresh and campaign-scoped.

| arm | runs | success | mean steps | unsafe | predictor time |
|:--|--:|--:|--:|--:|--:|
| `direct_gpt` | 9 | 100% | 2.00 | 0 | 0.0 s |
| `candidates_heuristic` | 9 | 100% | 2.00 | 0 | 0.0 s |
| `candidates_base_laya` | 9 | 100% | 2.11 | 0 | 2.7 s |

Candidate generation used 10,326 tokens and cost $0.044804. Direct GPT used 7,054 tokens and cost $0.023908. Total provider usage was 17,380 tokens at $0.068712. Laya loaded on CUDA in 1.76 seconds. Cache accounting was 20 misses and 17 internal hits, with no legacy or replayed provenance.

The only behavioural difference occurred on `pressure-001`. Direct GPT and the heuristic selector opened `/pump-sheet` and answered in two steps. Base Laya first selected the no-op `inspect_search_page`, then opened `/pump-sheet` and answered in three steps. At the first decision, the fixed utility policy scored `inspect_search_page` at approximately 0.726 and `open_official_pump_sheet` at approximately 0.699, a margin of 0.028. The observed no-op had zero goal progress and zero information gain. This is a concrete base-model ranking error, not a planner failure, but one visible development case is not statistically conclusive.

The extra branch explains the cache counts: the heuristic arm generated candidates for 18 canonical states. Laya reused 17 of them and created two additional entries for its divergent history. All arms completed every task without simulated unsafe actions. Absolute Laya confidence remains uncalibrated because the checkpoint temperature warning was active.

The raw campaign artifacts remain local under `reports/`, `data/trajectories.sqlite3`, `data/candidate_cache/`, and `logs/runs/`. They are intentionally excluded from Git. this entry contains the stable aggregate and interpretation needed for repository auditability.

This challenge suite is a visible development benchmark. It does not satisfy the held-out sample-size requirements in the evaluation protocol and must not be presented as evidence that the architecture improves performance.

## benchmark-a314047d · final-v001 pilot · seed 0

Date: 2026-09-28. Provider: OpenRouter. Model: `openai/gpt-5.6-sol`. Laya checkpoint: `/mnt/fast-ssd/models/laya/base-english`. Device: CUDA. Candidate cache: fresh and campaign-scoped. This was the pre-registered one-seed pilot and is explicitly non-compliant for final inference because it did not use seeds `0,1,2,3,4`.

| arm | runs | success | success 95% CI | mean steps | steps 95% CI | unsafe | unnecessary |
|:--|--:|--:|:--|--:|:--|--:|--:|
| `direct_gpt` | 90 | 100.0% | [100.0%, 100.0%] | 2.00 | [2.00, 2.00] | 0 | 0 |
| `candidates_heuristic` | 90 | 100.0% | [100.0%, 100.0%] | 2.00 | [2.00, 2.00] | 0 | 0 |
| `candidates_base_laya` | 90 | 85.6% | [77.8%, 92.2%] | 3.06 | [2.79, 3.34] | 0 | 121 |

Against either control, base Laya's paired success delta was -14.4 percentage points with a 95% bootstrap interval of [-22.2, -7.8]. Its paired step delta was +1.06 with a 95% interval of [+0.79, +1.34]. The pilot therefore provides strong evidence that the zero-shot checkpoint is harmful in this environment, not merely slower on a few examples.

The failure is concentrated by task family. Base Laya scored 70.0% on `heldout-qualified-limit-v1`, 86.7% on `heldout-current-policy-v1`, and 100.0% on `heldout-security-threshold-v1`. Nine of its 13 failures came from qualified-limit, four from current-policy, and none from security-threshold. Ten failures exhausted the six-step limit and three selected a terminal failure action. The traces show repeated inspection, search and reopening of official sources after sufficient evidence was available. This supports training explicit counterfactual labels for progress, information gain, redundancy and terminal readiness, while keeping final-v001 examples excluded from the training data.

Candidate generation consumed 194,076 tokens at $0.819272. Direct planning consumed 76,541 tokens at $0.260818, for $1.08009 total recorded provider cost. Laya loaded on CUDA in 1.79 seconds and spent 45.8 seconds in predictor calls across its 90 episodes. The cache recorded 324 misses and 131 internal hits.

This result does not justify changing final-v001 after observing it. The suite stays frozen for the declared base-versus-fine-tuned comparison, and its individual states, labels and failures must not drive training or model selection. The 100% heuristic result also exposes a limit of this suite: it can test whether a fine-tuned Laya recovers reliable ranking, but cannot demonstrate superiority over the heuristic. A later external computer-use benchmark with new environments and failure-recovery cases is required for that broader claim.
