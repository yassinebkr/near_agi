# Laya post-training plan

## Research question

The base English checkpoint is heterogeneous. It can select an action quickly when its ranking is correct, but it also loops, adds unnecessary actions, and fails tasks that direct GPT and the heuristic solve. The registered experiment asks whether one domain-specialised Laya can remove those failures and long-tail slowdowns while preserving fast local selection.

The target is prediction of six transition properties followed by the frozen utility policy. It is not generic language-model training. GPT/Laya routing remains future work only if a completed held-out evaluation shows persistent regions of weakness.

## Claim boundary

The upstream browser-agent result and this project are not directly comparable. Upstream reports task success and element selection in a different environment, model, action space, and sample design. A strong `final-v001` result would establish performance only in this bounded deterministic sandbox.

The evidence ladder is:

- **fit:** transition prediction improves on disjoint development templates;
- **runtime transfer:** the same exported checkpoint works through the production predictor API;
- **held-out result:** the complete frozen final campaign meets the registered success and safety rules;
- **external generalisation:** a later benchmark covers unseen sites, perturbations, and recovery behavior.

Only the first two stages are complete. The held-out campaign is paused and external generalisation has not been tested.

## Frozen v2bis dataset

GPT-5.6 Sol generated contrastive candidate actions at reachable simulator states. Every candidate was executed on an independent cloned state, and all six labels came from the observed transition. GPT supplied actions and rationales, never labels.

The frozen corpus contains:

- 320 disjoint scenario groups and 1,600 anchor states;
- 7,963 semantically distinct candidate transitions;
- 47,778 typed property-question sequences;
- 29,856 training, 4,482 validation, 4,476 calibration, and 8,964 development questions;
- 1,600 candidate-generation calls, 1,624,442 tokens, and $9.775152 recorded provider cost.

Current and historical action identifiers are excluded from model input and retained separately for audit. Final-v001 templates and identifiers are rejected. The committed manifest `configs/train/v002bis.dataset-manifest.json` is authoritative.

## Training record

The v2bis recipe ran on one H100 80 GB at pinned Laya revision `9d955671415fc19f069b9cc998928075c1f255ec`:

- four epochs and 5,648 updates;
- 46,371,505 processed tokens;
- 5,687.8 seconds elapsed;
- 8.82 GiB peak reserved VRAM;
- exported model SHA-256 `19eb33a1a2ad62e019325b29f3796b2e507fb4723ca6bd4ac22b3de726458fbf`.

Calibration used its frozen disjoint split. The first evaluator incorrectly required exact action hashes and understated selection accuracy when several actions had equal target utility. The corrected evaluator is tie-aware and preserves exact match as a diagnostic. V2bis selected an optimal action in 299 of 300 development groups, made one suboptimal selection, produced zero unsafe selections, and recorded mean regret `0.006`.

No weights changed during the evaluator correction.

## Historical negative results

V1 achieved a perfect synthetic offline score but failed the runtime smoke because the generator retained exploitable action identifiers, paths, and phase structure. It is classified `runtime_transfer_gate_failed`.

V2 was interrupted after a representation audit found that current and historical action identifiers still reached model input. Its resume checkpoint is retained as historical evidence and cannot be promoted or renamed as v2bis.

These failures motivated executed counterfactuals, identifier removal, semantic deduplication, and mandatory runtime gates.

## Runtime promotion

V2bis passed both live gates through the production `LayaPredictor` API:

- smoke: 3/3 successes, two steps per task, zero unsafe and unnecessary actions;
- challenge: 9/9 successes, two steps per task, zero unsafe and unnecessary actions;
- base Laya on the same challenge: 7/9 successes with ten unnecessary actions.

Passing these gates unlocked the single registered `final-v001` campaign. It did not establish the final result.

## Paused final evaluation

The final matrix has 90 held-out tasks, five seeds, and four arms for 1,800 episodes. It uses one campaign-scoped cache so candidate selectors receive byte-equivalent actions for matching states.

The campaign is paused after 705 durable episodes because the OpenRouter key budget was exhausted. The failed episode is excluded from the completed set. The checkpoint remains resumable, and the complete workspace was evacuated before VM and disk deletion. No result is inferred from the incomplete prefix.

When API budget is available, restore the evacuated workspace and resume the identical campaign. Do not regenerate the cache, change prompts, alter task definitions, retrain v2bis, or use the observed prefix for model selection.

## Frozen decision rule

The specialised model is interesting only if the complete held-out report shows all of the following:

- task success is no worse than direct GPT;
- fine-tuned Laya improves over base Laya on transition prediction and task outcomes;
- unsafe actions do not increase;
- unnecessary actions decrease;
- API tokens do not materially increase;
- selector speed and end-to-end latency are reported separately with paired intervals and long-tail distributions.

The ideal result preserves the fastest Laya cases while removing loops, errors, and major slowdowns. A negative result is valid and ends further spending on the current representation until the research question or state/action design changes.

## Sources

- [Official Laya repository and fine-tuning guidance](https://github.com/NandhaKishorM/laya)
- [Official browser-agent fine-tuning example](https://github.com/NandhaKishorM/laya/blob/main/docs/finetune_browser_agent.md)
- [Reproducible laya-browser training artifacts](https://huggingface.co/cklxx/laya-browser)
- [BeyondVRAM cloud runbook](https://github.com/yassinebkr/beyondvram/blob/main/docs/planning/track5-cloud-runbook.md)
