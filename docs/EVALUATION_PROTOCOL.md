# Evaluation protocol (pre-registered draft)

## Experimental contract

The primary live comparison uses one exact model id for both GPT arms: direct GPT action selection versus GPT candidate generation followed by a local transition selector. Candidate lists are cached by provider/model, prompt version, seed, candidate limit, and canonical state hash. Cache schema v2 stores generation timestamp, latency, provider, model, state hash, and original token/cost usage. Legacy entries remain replayable but are explicitly reported as having unknown usage provenance. Therefore heuristic and Laya selectors receive byte-equivalent candidates whenever they reach the same state. A direct GPT action is generated independently and is not mislabeled as “the first candidate”.

Clean live campaigns must use `--fresh-candidate-cache`. This selects a new campaign-scoped cache without deleting previous evidence. Cache hits can still occur inside that campaign when the heuristic and Laya arms reach the same canonical state; these internal hits are required for a paired comparison and are not contamination from an earlier campaign.

The smoke suite currently contains three deterministic tasks from three templates. It validates plumbing only. It cannot support a research claim. A claim requires untouched template-level test splits, at least 30 task instances per held-out template, and 5 declared seeds. Live model nondeterminism remains possible even when a replicate is labelled by a seed; cached responses make replay exact.

## Arms

- `direct_gpt`: the configured GPT model chooses one action directly.
- `candidates_heuristic`: the configured GPT model generates candidates; a transparent heuristic scores them.
- `candidates_base_laya`: the same cached candidates are scored by the unmodified local Laya checkpoint.
- A future `candidates_finetuned_laya` arm is admitted only after Milestone 2.

The deterministic provider supplies offline stand-ins for wiring tests and must never be reported as GPT or Laya performance.

## Frozen metadata and measurements

Every episode records campaign/run/task/template identifiers, seed, provider, exact model, labeler version, direct or candidate prompt version, candidate count, policy weights, candidate-cache path, chosen actions, stop reason, planner usage, planner latency, predictor latency, unsafe actions, unnecessary actions, prediction error, and reconstructable transitions in SQLite plus JSONL.

Report task success, mean/median steps, unnecessary actions, loops, tool failures, simulated unsafe actions, token/cost fields returned by the provider, component latency, and prediction MAE. For a full held-out campaign add mean/median/std and bootstrap confidence intervals, peak RAM/VRAM, balanced accuracy, Brier score, log loss, ECE, and reliability plots where outputs are calibrated probabilities.

## Decision rule

The specialised model is interesting only if held-out task success is no worse than `direct_gpt`, unnecessary actions decrease, API tokens do not materially increase, added decision latency stays below 20% of GPT latency, and fine-tuned Laya beats base Laya on both transition prediction and task outcomes. Raw results and confidence intervals accompany every conclusion. This rule is fixed before training.

Ablations cover history, beliefs, information gain, risk, base versus fine-tuned Laya, 1/3/5/8 candidates, raw versus structured formatting, and a cheaper GPT model versus the main `openai/gpt-5.6-sol` campaign.
