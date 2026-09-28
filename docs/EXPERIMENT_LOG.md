# Experiment log

## benchmark-795a36f9 · challenge · seed 0

Date: 2026-09-28. Provider: OpenRouter. Model: `openai/gpt-5.6-sol`. Laya checkpoint: `/mnt/fast-ssd/models/laya/base-english`. Device: CUDA. Candidate cache: fresh and campaign-scoped.

| arm | runs | success | mean steps | unsafe | predictor time |
|:--|--:|--:|--:|--:|--:|
| `direct_gpt` | 9 | 100% | 2.00 | 0 | 0.0 s |
| `candidates_heuristic` | 9 | 100% | 2.00 | 0 | 0.0 s |
| `candidates_base_laya` | 9 | 100% | 2.11 | 0 | 2.7 s |

Candidate generation used 10,326 tokens and cost $0.044804. Direct GPT used 7,054 tokens and cost $0.023908. Total provider usage was 17,380 tokens at $0.068712. Laya loaded on CUDA in 1.76 seconds. Cache accounting was 20 misses and 17 internal hits, with no legacy or replayed provenance.

The only behavioural difference occurred on `pressure-001`. Direct GPT and the heuristic selector opened `/pump-sheet` and answered in two steps. Base Laya first selected the no-op `inspect_search_page`, then opened `/pump-sheet` and answered in three steps. At the first decision, the fixed utility policy scored `inspect_search_page` at approximately 0.726 and `open_official_pump_sheet` at approximately 0.699, a margin of 0.028. The observed no-op had zero goal progress and zero information gain. This is a concrete base-model ranking error, not a planner failure, but one visible development case is not statistically conclusive.

The extra branch explains the cache counts: the heuristic arm generated candidates for 18 canonical states; Laya reused 17 of them and created two additional entries for its divergent history. All arms completed every task without simulated unsafe actions. Absolute Laya confidence remains uncalibrated because the checkpoint temperature warning was active.

The raw campaign artifacts remain local under `reports/`, `data/trajectories.sqlite3`, `data/candidate_cache/`, and `logs/runs/`. They are intentionally excluded from Git; this entry contains the stable aggregate and interpretation needed for repository auditability.

This challenge suite is a visible development benchmark. It does not satisfy the held-out sample-size requirements in the evaluation protocol and must not be presented as evidence that the architecture improves performance.
