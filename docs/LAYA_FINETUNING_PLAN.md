# Laya post-training plan

## Decision to make

The base English checkpoint currently shows heterogeneous behavior: it fails or loops on some tasks, while qualitative timings suggest that it may be very fast on others. Earlier campaigns did not persist enough cache-aware wall telemetry to quantify an end-to-end speedup. The next milestone tests whether domain post-training changes that result. The target is not generic language modelling: it is accurate prediction of the six recorded transition properties for candidate actions, followed by the already frozen utility policy.

The final-v001 benchmark remains untouched. Its three templates, 90 tasks, five seeds and task-manifest fingerprint are evaluation-only. No final-v001 state, candidate, label or outcome may enter training, calibration, prompt tuning or model-selection decisions.

## Research claim and ambition

The published Laya browser-agent result is a useful reference point, but not a directly comparable baseline. Its reported end-to-end result moved from 0% zero-shot task success to 62% for the fine-tuned 322M model on 16 browser tasks over three runs. The nearby 63% figure is the 322M model's element top-1 score on 2,734 held-out decisions. Our benchmark measures a different agent, action space and environment, so we must not present a higher number as a direct reproduction or replacement of that result.

Our primary research objective is nevertheless deliberately stronger: determine whether a domain-specialized Laya selector can approach perfect task success while preserving the speed advantage of a small local decision model. "Approach perfect" means at least 95% success on unseen task families, with zero unsafe actions, fewer unnecessary actions than base Laya, and no statistically supported regression against direct GPT. A point estimate of 100% is a stretch result, not proof of universal reliability.

Post-training latency evaluation will measure the full distribution rather than a single total: paired observed and reconstructed end-to-end ratios, p50/p90/p95, catastrophic long-tail cases and the preservation of already-fast cases. The ideal outcome is to retain the best observed speedups while eliminating loops, errors and major slowdowns, without assuming in advance that this is achievable. Selector speed with candidates already available and system-level cold-start efficiency against direct GPT are separate claims.

The evidence ladder is:

- fit: held-out transition prediction and calibration improve on disjoint instances.
- transfer: the fine-tuned selector beats base Laya and the heuristic on unseen development templates.
- final: it reaches at least 95% success on frozen final-v001, with paired bootstrap intervals and per-template results reported.
- generalization: a later external computer-use suite with unseen sites, perturbations and failure recovery reproduces the gain.

Only the first three claims are in scope for the current experiment. Near-100% on final-v001 would establish strong performance in our bounded environment, not near-100% general computer use. To keep that result credible, final-v001 remains single-use for the declared comparison. Any training iteration after seeing it requires a new hidden final suite and manifest.

## Evidence behind the compute choice

The local checkpoint is the 421M English model with an 842,609,210-byte safetensors file. The official Laya typed-decisions recipe trains this model for four epochs over roughly 30,000 questions on two T4 GPUs in four to five hours, with encoder and head gradient checkpointing. The official browser-agent adaptation reports full fine-tuning on one 16 GB RTX 4070 Ti SUPER: four epochs take about two hours for the 421M model and one hour for the 322M model. It explicitly says that 16 GB fits micro-batch 4 at length 1024 without checkpointing.

Our RTX 3070 Ti has 8 GB. For the 421M checkpoint, full-parameter AdamW has a static memory floor from parameters, gradients and optimizer states before activations and CUDA workspaces. Therefore a full reference run on 8 GB is unlikely to be comfortable. This is an inference from model size and the published 16 GB recipe, not a measured local OOM. We will settle it with one optimizer step rather than assume.

BeyondVRAM remains the operations reference, not the Laya trainer. Its historical H200 run used a measured $2.58/h configuration and demonstrated checkpoint/resume, evacuation, fail-closed gates and teardown. Current Nebius list prices differ: L40S starts at $1.55/GPU-hour on demand and $0.74 preemptible. H100/H200 preemptible starts at $0.79, while H100 and H200 on demand are $3.85 and $4.50 before the announced October 2026 increase. Spot prices are dynamic and exclude tax. Since upstream Laya fits on 16 GB, an L40S is already ample. An H200 is unnecessary unless availability makes it cheaper in practice.

The 2026-09-28 console check found that L40S capacity was available only as a two-GPU allocation. A single H100 was quoted at $2.16 per hour, $0.10 less per hour than the two-L40S allocation, so H100 is the selected target. With a remaining $20 balance, the hard compute budget stays at $16. A six-hour execution cap costs at most $12.96 before tax and leaves $3.04 for bootstrap time and variance.

## Current post-training status

The frozen Nebius recipe completed on 2026-09-28 and passed its offline development gate, increasing candidate-selection accuracy from 43.06% to 100% on 360 synthetic groups. A later runtime smoke exposed that this gate was insufficient: the fine-tuned checkpoint failed all three ordinary benchmark tasks while the direct and heuristic controls passed all three. The development generator had changed template names and values but retained training action identifiers, paths and phase structure, allowing shortcut learning.

The checkpoint is therefore classified `runtime_transfer_gate_failed` and must not enter `final-v001`. The final four-arm benchmark implementation is ready, but the frozen suite remains untouched. A replacement training dataset must randomize action identifiers and paths, diversify state and action wording, and pass both runtime smoke and challenge before the single registered final comparison.
Training v2 was interrupted before completion after an audit found that opaque current and historical action identifiers still reached the model input. Frequent zero-cross-entropy batches reinforced the decision not to spend further compute on that corpus. The resumable v2 checkpoint remains historical evidence and is not eligible for promotion or reuse by v2bis.

Training v2bis removes action identifiers from both training and runtime model inputs while preserving them as external audit metadata. Scenarios are assigned to disjoint splits before generation. At each reachable anchor state, GPT-5.6 Sol generates the actual contrastive candidate set; every candidate is then executed on an independent clone of that exact simulator state, and all six labels come exclusively from the observed transition. Invalid semantic actions remain auditable negative examples. The generator fails closed when a split lacks tool diversity or positive success, progress, information-gain or risk examples.

The validated pilot produced 100 executed candidates from 20 GPT calls, with five candidates per state, all three tools and positive examples for every gated property. It cost $0.120364 for 20,194 tokens. This closes the synthetic-candidate domain gap, but it does not claim real-browser transitions or complete behavioral diversity. The automatic promotion path remains offline development, four-arm smoke, four-arm challenge, then the frozen regression suite; every runtime phase is resumable at episode granularity.

## Dataset contract

Build examples from candidate transitions, not from benchmark tasks. Each item freezes the compact pre-action state, one GPT-generated candidate action, the six executed transition labels, template identifier, task identifier, scenario-group identifier, state hash, semantic action hash, post-action state, planner provenance, labeler version and transition error.

For the deterministic sandbox, generate counterfactual labels for every candidate from a cloned state so the model learns from good, irrelevant, premature and unsafe actions rather than only the action selected by the current policy. For future real environments, only executed transitions are ground truth. On-policy mistakes become DAgger-style additions after independent verification.

Use template-level separation:

- training: new synthetic templates and instances that do not occur in final-v001.
- validation: disjoint instances from training templates for optimization and early stopping.
- calibration: disjoint instances used only to fit per-type temperatures after weights are frozen.
- development test: unseen non-final templates for model selection.
- final-v001: untouched until the single declared base-versus-fine-tuned evaluation.

The first target is 30,000 to 60,000 typed questions, balanced across the six properties and across positive, negative and unsafe transitions. Record class balance before training. Do not manufacture labels with an LLM when the simulator can supply exact outcomes.

## Training stages and gates

1. Reproduce the official item format on 32 examples and verify byte-level round trips, labels, masks and checkpoint reload on CPU.
2. Run one local CUDA optimizer step on the 421M checkpoint with sequence length 1024, micro-batch 1, gradient accumulation, encoder and decision-head checkpointing, autocast, and `torch.compile` disabled. Record peak allocated and reserved VRAM, wall time, tokens per second and checkpoint size.
3. Continue locally only if peak reserved VRAM is at most 7.5 GB, there is no CPU parameter offload, checkpoint reload is exact, and projected full-run wall time is at most 12 hours. Otherwise stop. An OOM is a routing result, not a failure to hide.
4. Run a small end-to-end training smoke on 256–1,024 items. Require decreasing held-out transition loss and improvement over the base checkpoint on the development split. Fit temperatures on the separate calibration split and remove inherited option-bucket temperatures before export.
5. If the local gate fails, launch one Nebius H100 preemptible instance with a persistent network disk. Reuse BeyondVRAM's ten-minute checkpoints, SIGTERM save, resume cursor, artifact hashes, evacuation and automatic teardown. Set a $16 hard budget and do not provision without explicit approval.
6. Train the frozen recipe, export an ordinary Laya checkpoint, reload it locally and evaluate base versus fine-tuned on the development suite. Only a clear development win unlocks final-v001.
7. Run final-v001 once with identical cached GPT candidates for `candidates_base_laya` and `candidates_finetuned_laya`. Direct GPT and heuristic remain controls. Report paired success, steps, unsafe actions, prediction error, calibration, selector latency, observed wall latency, reconstructed end-to-end latency and latency-ratio distributions.

## Promotion and kill rules

Promote to final evaluation only if the fine-tuned model beats base Laya on held-out transition prediction, does not increase unsafe selections, reduces unnecessary actions on the development test without materially increasing decision latency, and reaches at least 95% success on the unseen development templates. Reject the training recipe if it merely memorizes templates, improves calibration without improving ranking, or loses to the transparent heuristic.

The project hypothesis fails in its present form if a correctly trained and calibrated Laya still cannot match the heuristic selector or if its final-v001 task success is below direct GPT. A negative result is valid and should end further spend until the state/action representation or research question changes.

A GPT/Laya fast-path or slow-path router is outside the current experiment. It becomes future work only if the specialised checkpoint retains distinct regions of weakness after the registered training and held-out evaluation.

## Sources

- [Official Laya repository and fine-tuning guidance](https://github.com/NandhaKishorM/laya)
- [Official browser-agent fine-tuning example](https://github.com/NandhaKishorM/laya/blob/main/docs/finetune_browser_agent.md)
- [Reproducible laya-browser training artifacts](https://huggingface.co/cklxx/laya-browser)
- [Current Nebius GPU pricing](https://nebius.com/prices)
- [BeyondVRAM cloud runbook](https://github.com/yassinebkr/beyondvram/blob/main/docs/planning/track5-cloud-runbook.md)
