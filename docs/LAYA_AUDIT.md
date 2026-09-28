# Laya audit

Audit date: 2026-09-28. Upstream: `NandhaKishorM/laya`, `main`; package metadata on `main`: **0.3.21** (the README heading still says 0.3.20). This checkout does not vendor or pin an upstream commit yet, so the exact commit must be recorded before a scientific run.

## Verified upstream contract

Laya is Apache-2.0 and exposes a local Python decision engine. `laya.load(...)` returns an agent whose `predict(state, questions)` accepts typed `choice`, ordinal `score`, and Boolean `noul` questions in one forward pass. Results contain typed answers, distributions/confidence, action metadata, and input-token usage. The router can select English or multilingual checkpoints. The current release documents `max_len=8192`, with accuracy strongest below roughly 4,000 tokens; checkpoint-specific head limits still apply.

The official documentation says the English and multilingual base checkpoints are near chance on its typed-decisions benchmark (0.362/0.352), warns that both are over-confident, and reports calibration repair by fitting temperatures per question type and option-count bucket. Numeric temperatures are clamped to `[0.5, 5.0]`; a load fallback is not evidence of calibration. `action.act_probability` currently has no useful signal; it must not gate execution.

`noul` can follow option labels instead of state, particularly on English. Explicit `true`/`false` criteria plus neutral labels, or a neutral two-option `choice`, must be validated on this domain. The multilingual checkpoint has a documented first-level bias for `score`. All six properties therefore need held-out calibration and formatting ablations.

## Fine-tuning and browser example

The official notebook `notebooks/laya_finetune_typed_decisions_2xT4_kaggle.ipynb` covers RLCD/GRPO-style fine-tuning, gradient checkpointing, temperature fitting, evaluation, and export. Current guidance holds calibration samples out of training and removes inherited bucket temperatures before a new fit. The documented run uses two T4 GPUs for roughly 4–5 hours over about 30k questions; that is not evidence that training fits an 8 GB 3070 Ti.

The worked browser-agent guide reports single-16-GB-GPU specialisation, with element top-1 improving from 0.10 to 0.66 and live-task success from 0% to 62%. It also reports formatting/data effects: real executed DONE states, mid-task negatives, rare-operation reweighting, and retaining page text mattered; templated leakage and unbalanced Mind2Web data failed. These numbers are upstream claims, not reproduced here.

## Mapping used by this POC

`success`, `goal_progress`, `risk`, `reversible`, and `needs_more_observation` use one `noul` each with explicit criteria. `information_gain` uses a five-level `score`; its raw expected level and distribution are retained, while `/4` is only a policy feature. Six questions for the same `(state, candidate)` share one forward pass. Calls across candidates remain separate so each state describes one contemplated action.

## Runtime and hooks

Upstream documents prediction hooks for logging/redaction/cache/gating, batch/HTTP interfaces, CPU and NVIDIA CUDA containers, and an OOM fallback that disables the TileLang fast path before CPU retry. This POC wraps rather than patches Laya and independently records wall latency and process/GPU telemetry. Actual CUDA, PyTorch, checkpoint load, length, batching, and VRAM behavior are machine-dependent and reported by `lda doctor`; none were exercised during this source-only audit.

## Primary sources

- https://github.com/NandhaKishorM/laya
- https://github.com/NandhaKishorM/laya/blob/main/docs/index.md
- https://github.com/NandhaKishorM/laya/blob/main/docs/finetune_browser_agent.md
- https://github.com/NandhaKishorM/laya/blob/main/notebooks/laya_finetune_typed_decisions_2xT4_kaggle.ipynb

