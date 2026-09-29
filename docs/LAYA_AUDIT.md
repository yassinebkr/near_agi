# Laya audit

Initial audit date: 2026-09-28. Upstream: `NandhaKishorM/laya`. The training and evaluation environment is pinned to revision `9d955671415fc19f069b9cc998928075c1f255ec`; package metadata reports **0.3.21**.

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

Upstream documents prediction hooks for logging/redaction/cache/gating, batch/HTTP interfaces, CPU and NVIDIA CUDA containers, and an OOM fallback that disables the TileLang fast path before CPU retry. This POC wraps rather than patches Laya and independently records wall latency and process/GPU telemetry.

The original source audit has now been followed by measured execution. Base and fine-tuned checkpoints load through the same `LayaPredictor` API on local CUDA and a Nebius H100. The v2bis full training peaked at 8.82 GiB reserved VRAM. The exported model passed offline evaluation plus live smoke and challenge gates. Base-checkpoint confidence remains uncalibrated because one shipped temperature is outside `[0.5, 5]` and is clamped during load; the recalibrated v2bis checkpoint does not emit that warning.

## Primary sources

- https://github.com/NandhaKishorM/laya
- https://github.com/NandhaKishorM/laya/blob/main/docs/index.md
- https://github.com/NandhaKishorM/laya/blob/main/docs/finetune_browser_agent.md
- https://github.com/NandhaKishorM/laya/blob/main/notebooks/laya_finetune_typed_decisions_2xT4_kaggle.ipynb
