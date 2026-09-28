# Experiment log

## Final-v001 pre-registration freeze

The final matrix was frozen before live execution: 90 tasks, three held-out templates, 30 instances per template, and seeds `0,1,2,3,4`. A compliant report must use a fresh cache, CUDA base Laya, a live provider, and complete all 1,350 episodes. Each report fingerprints the canonical task manifest with SHA-256. A one-seed pilot is operational validation only and is excluded from final inference.

## Historical latency limitation

The campaigns below predate durable episode wall-clock and cache-provenance telemetry. Candidate-based arms shared a `CachedPlanner`, so the first selector reaching a state could pay the GPT generation latency while another selector replayed the same candidates. Their recorded `predictor time` measures predictor-internal work only. It is neither selector-path wall time nor cold-start end-to-end time.

Terminal observations suggested substantial heterogeneity: some base-Laya episodes appeared two to three times slower or failed, while others appeared roughly three to six times faster than direct GPT. Those observations motivated the current instrumentation, but they cannot support a retrospective speedup distribution. Historical values remain unchanged, and no definitive end-to-end latency conclusion is drawn from them. New campaigns separately report observed wall time, fresh/replayed provenance and reconstructed effective end-to-end latency when original generation timing is complete.

## benchmark-9407b961 · latency instrumentation validation

Date: 2026-09-28. Provider: deterministic fixture. Suite: smoke. Seed: 0. Device: CUDA for base Laya. This nine-episode run validates telemetry only and carries no claim about GPT or Laya speed.

The campaign recorded seven fresh candidate generations and four cache replays. All nine SQLite run rows contain `summary_json`, every transition records candidate provenance and component timings, and every episode budget closes as `wall_clock_ms = planner_wall_ms + predictor_wall_ms + policy_ms + environment_ms + framework_overhead_ms` within floating-point precision. Reconstructed latency was complete for all six candidate-based episodes. Time-to-success correctly excluded the two failed Laya pairs.

The deterministic fixture makes provider generation nearly instantaneous, while synchronous SQLite/JSONL persistence appears as roughly 129–144 ms mean framework overhead per episode. This is expected and exposes the measurement boundary rather than a timing defect. The run also contains one mixed Laya path caused by a divergent state, which validates simultaneous fresh and replay accounting. No speedup conclusion is drawn.

## laya-dynamics-v002 · interrupted corpus audit

Date: 2026-09-28. The paid v2 run was stopped during epoch one before export or evaluation. Its log had reached step 660, with frequent zero-cross-entropy batches. An audit then established that opaque current action identifiers still reached Laya through `candidate_action`, while earlier identifiers reached it through `recent_history`. Fixed phase and candidate grammar also remained exploitable. Continuing the run could not answer the intended transfer question, so the interruption was a methodological decision rather than a training-system failure.

The resumable v2 checkpoint and original logs remain on the persistent Nebius system disk as historical evidence. They must not be renamed, copied into the v2bis output directory or used for promotion. No v2 quality result is inferred because training did not complete.

The replacement v2bis pilot uses GPT-5.6 Sol candidate sets and independently cloned simulator execution for every candidate. Its accepted 20-call pilot produced 100 executed candidates and 600 labeled questions, represented all three tools, and included positive success, progress, information-gain and risk transitions in every split. Recorded provider usage was 20,194 tokens at $0.120364. This validates collection mechanics only; it is not a model-quality result.

## laya-dynamics-v001 · Nebius post-training

Date: 2026-09-28. Hardware: one NVIDIA H100 80 GB. Laya revision: `9d955671415fc19f069b9cc998928075c1f255ec`. The three-step smoke was resumed into the frozen four-epoch recipe. Training completed 7,352 optimizer updates and 41,809,925 tokens in 7,722 seconds, with 8.74 GiB peak reserved VRAM. Epoch-average objectives were `0.0110`, `0.0426`, `-0.0107` and `-0.0108`. The signed RLCD objective is not expected to decrease monotonically. Cross-entropy was exactly zero at 77.9% of logged points, indicating that most sampled batches became trivial; this makes independent task evaluation essential.

On the unseen development templates, base Laya reached 44.38% property argmax accuracy, 43.06% candidate-selection accuracy and 0.460819 mean absolute error. The fine-tuned checkpoint reached 100% property argmax accuracy, 100% selection accuracy and 0.00002344 mean absolute error across 10,800 items and 360 candidate groups. Both checkpoints made zero unsafe selections. All four pre-registered development promotion checks passed.

The result is a strong fit-and-transfer signal within the synthetic generator, but it is not yet the final scientific result. Development templates are disjoint from training, while sharing the same deterministic labeler and representation. Near-perfect scores may therefore reflect transferable rules, excessive task regularity or both. The frozen `final-v001` task benchmark remains necessary to measure end-to-end success, unnecessary actions and cache-aware latency without changing the training recipe.

The final model SHA-256 is `dca6ad280d90fdc57458328ca1939f14adf84cd5c741c074475ac7e05586b7a4`. A local evacuation verified 3,759 files against a manifest whose SHA-256 is `c13d7fde32ab115dd5fde265731eb53c94c6ad317e77301a65554dd965884df2`. The machine-specific backup location is intentionally untracked. Structured results are recorded in `configs/train/v001.result.json`. The checkpoint's `post_training` object and structured result are authoritative; its legacy `training` object was inherited from the base checkpoint and does not describe this run.

A subsequent four-arm deterministic runtime smoke, `benchmark-0bb29e0f`, invalidated promotion before `final-v001` was touched. Direct and heuristic controls passed all three tasks, base Laya passed one, and fine-tuned Laya passed none. Fine-tuned prediction MAE averaged 0.4594. A schema-aligned probe without explicit noul labels also failed all three tasks, ruling out that formatting difference as the sole cause.

The offline development split changed template names and entity values but reused the same synthetic action identifiers, paths and phase structure as training. The model could therefore obtain perfect offline scores from shortcuts that do not exist in runtime benchmark actions. The result is reclassified as `runtime_transfer_gate_failed`; the checkpoint is archived as a negative result and is forbidden from the single-use final campaign. Future training data must randomize action identifiers and paths, diversify state/action phrasing, and pass runtime smoke plus challenge before final promotion.

## benchmark-795a36f9 · challenge · seed 0

Date: 2026-09-28. Provider: OpenRouter. Model: `openai/gpt-5.6-sol`. Laya checkpoint: `<local-checkpoint>/base-english`. Device: CUDA. Candidate cache: fresh and campaign-scoped.

| arm | runs | success | mean steps | unsafe | predictor time |
|:--|--:|--:|--:|--:|--:|
| `direct_gpt` | 9 | 100% | 2.00 | 0 | 0.0 s |
| `candidates_heuristic` | 9 | 100% | 2.00 | 0 | 0.0 s |
| `candidates_base_laya` | 9 | 100% | 2.11 | 0 | 2.7 s |

Candidate generation used 10,326 tokens and cost $0.044804. Direct GPT used 7,054 tokens and cost $0.023908. Total provider usage was 17,380 tokens at $0.068712. Laya loaded on CUDA in 1.76 seconds. Cache accounting was 20 misses and 17 internal hits, with no legacy or replayed provenance.

The only behavioural difference occurred on `pressure-001`. Direct GPT and the heuristic selector opened `/pump-sheet` and answered in two steps. Base Laya first selected the no-op `inspect_search_page`, then opened `/pump-sheet` and answered in three steps. At the first decision, the fixed utility policy scored `inspect_search_page` at approximately 0.726 and `open_official_pump_sheet` at approximately 0.699, a margin of 0.028. The observed no-op had zero goal progress and zero information gain. This is a concrete base-model ranking error, not a planner failure, but one visible development case is not statistically conclusive.

The extra branch explains the cache counts: the heuristic arm generated candidates for 18 canonical states. Laya reused 17 of them and created two additional entries for its divergent history. All arms completed every task without simulated unsafe actions. Absolute Laya confidence remains uncalibrated because the checkpoint temperature warning was active.

The raw campaign artifacts remain local under `reports/`, `data/trajectories.sqlite3`, `data/candidate_cache/`, and `logs/runs/`. They are intentionally excluded from Git. This entry contains the stable aggregate and interpretation needed for repository auditability.

This challenge suite is a visible development benchmark. It does not satisfy the held-out sample-size requirements in the evaluation protocol and must not be presented as evidence that the architecture improves performance.

## benchmark-a314047d · final-v001 pilot · seed 0

Date: 2026-09-28. Provider: OpenRouter. Model: `openai/gpt-5.6-sol`. Laya checkpoint: `<local-checkpoint>/base-english`. Device: CUDA. Candidate cache: fresh and campaign-scoped. This was the pre-registered one-seed pilot and is explicitly non-compliant for final inference because it did not use seeds `0,1,2,3,4`.

| arm | runs | success | success 95% CI | mean steps | steps 95% CI | unsafe | unnecessary |
|:--|--:|--:|:--|--:|:--|--:|--:|
| `direct_gpt` | 90 | 100.0% | [100.0%, 100.0%] | 2.00 | [2.00, 2.00] | 0 | 0 |
| `candidates_heuristic` | 90 | 100.0% | [100.0%, 100.0%] | 2.00 | [2.00, 2.00] | 0 | 0 |
| `candidates_base_laya` | 90 | 85.6% | [77.8%, 92.2%] | 3.06 | [2.79, 3.34] | 0 | 121 |

Against either control, base Laya's paired success delta was -14.4 percentage points with a 95% bootstrap interval of [-22.2, -7.8]. Its paired step delta was +1.06 with a 95% interval of [+0.79, +1.34]. The pilot therefore provides strong evidence that the zero-shot checkpoint is harmful in this environment, not merely slower on a few examples.

The failure is concentrated by task family. Base Laya scored 70.0% on `heldout-qualified-limit-v1`, 86.7% on `heldout-current-policy-v1`, and 100.0% on `heldout-security-threshold-v1`. Nine of its 13 failures came from qualified-limit, four from current-policy, and none from security-threshold. Ten failures exhausted the six-step limit and three selected a terminal failure action. The traces show repeated inspection, search and reopening of official sources after sufficient evidence was available. This supports training explicit counterfactual labels for progress, information gain, redundancy and terminal readiness, while keeping final-v001 examples excluded from the training data.

Candidate generation consumed 194,076 tokens at $0.819272. Direct planning consumed 76,541 tokens at $0.260818, for $1.08009 total recorded provider cost. Laya loaded on CUDA in 1.79 seconds and spent 45.8 seconds in predictor calls across its 90 episodes. The cache recorded 324 misses and 131 internal hits.

This result does not justify changing final-v001 after observing it. The suite stays frozen for the declared base-versus-fine-tuned comparison, and its individual states, labels and failures must not drive training or model selection. The 100% heuristic result also exposes a limit of this suite: it can test whether a fine-tuned Laya recovers reliable ranking, but cannot demonstrate superiority over the heuristic. A later external computer-use benchmark with new environments and failure-recovery cases is required for that broader claim.
