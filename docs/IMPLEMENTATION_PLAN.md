# Implementation plan

## Scope and experimental boundary

Milestone 1 answers one narrow question: does adding typed, per-candidate transition predictions improve decisions over the same candidate generator? The sandbox is the source of truth. Laya is removable and is never allowed to execute tools. No Shopify, JEV, latent rollout, or fine-tuning is in this milestone.

## Architecture

`Planner -> candidate actions -> TransitionPredictor -> Policy -> Environment -> deterministic labels -> prediction errors -> SQLite + JSONL`.

All boundaries are protocols. `USE_LAYA=0` selects the GPT-only policy without changing the environment, planner, runner, or storage. The initial sandbox is an in-process deterministic web abstraction: it has pages, concise observations, typed navigation/answer actions, source quality, contradictions, missing information, and simulated irreversible actions. This is deliberately smaller and more reproducible than Playwright; a browser adapter can later implement the same `Environment` protocol.

## Interfaces

- `Planner.propose_actions(state, max_actions) -> list[CandidateAction]`
- `TransitionPredictor.predict(state, actions) -> dict[action_id, TransitionPrediction]`
- `Policy.select(state, actions, predictions) -> CandidateAction`
- `Environment.reset(task) -> AgentState`; `step(action) -> Transition`
- `TransitionLabeler.label(before, action, after, truth) -> ObservedProperties`
- `TrajectoryStore` persists a reconstructable run and mirrors events to JSONL.

## Data schemas

Pydantic v2 models use `schema_version="1.0"`, forbid unknown fields, and canonical JSON (`sorted keys`, compact separators) for SHA-256 state hashes. Actions contain a stable id, allowlisted tool, JSON arguments, and optional short rationale. Prediction values are bounded floats and preserve primitive/calibration metadata rather than claiming every raw score is a calibrated probability.

## Laya strategy

One batched `predict` call asks six typed questions for one candidate rendered with the compact state. Boolean properties use `noul` with explicit neutral model-facing labels; information gain can use an ordinal `score`, normalized only for policy utility and retained with its raw scale. Each candidate currently requires a call because the questions concern a different proposed transition; latency is recorded. Real Laya failures fail closed and are logged. A deterministic `OracleLikePredictor` exists only for offline wiring tests and is labelled `surrogate`, never `base_laya`.

## Labels and prediction error

Labels derive from sandbox state changes and ground truth: terminal correctness, relevant facts discovered, unknowns resolved, side-effect flags, reversibility, and whether uncertainty remains. Every field records deterministic provenance. Prediction errors are absolute error per property; aggregate probabilistic metrics are deferred until a sufficiently large held-out set exists.

## Benchmark

Paired task templates and seeds compare `gpt_only`, `heuristic`, and, only when a checkpoint loads, `base_laya`. Smoke runs validate plumbing and do not support scientific conclusions. Full evaluation will split by task template, use repeated seeds, retain raw results, and report mean/median/std/bootstrap intervals, success, steps, unsafe actions, tokens, latency, RAM, and VRAM.

## Risks

- Laya 0.3.20 base checkpoints are weak zero-shot and miscalibrated; reported confidence is not correctness.
- RTX 3070 Ti 8 GB is the local inference and training-smoke target, not a hard training limit. Full Milestone 2 fine-tuning/post-training may use explicitly authorized Nebius compute, reusing the proven cloud lifecycle from BeyondVRAM while preserving the same experimental contract.
- Candidate formatting can dominate checkpoint differences; formatting version is recorded.
- A deterministic planner proves the loop, not the GPT-vs-Laya hypothesis. Live experiments require OpenAI credentials and the local checkpoint.
- Python 3.13 is available locally, but Laya/PyTorch compatibility may require Python 3.11 or 3.12; the project permits 3.11–3.13 and `doctor` reports the actual stack.

## Milestones and exact order

1. Freeze schemas, hashes, protocols, allowlist, and compact formatting.
2. Implement deterministic sandbox tasks and oracle labels.
3. Implement SQLite/event log and exact reconstruction.
4. Add deterministic and OpenAI planners.
5. Add explicit policies and loop detection.
6. Add Laya adapter with typed mappings and telemetry.
7. Run end-to-end demo and paired smoke benchmark.
8. Harden tests and produce report artifacts.
9. Milestone 2: collect trajectories, template-level splits, reproduce official fine-tune, calibrate on a disjoint set, evaluate untouched test.
10. Milestone 3 only after a positive result: DAgger and broader templates.

