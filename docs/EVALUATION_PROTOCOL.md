# Evaluation protocol (pre-registered draft)

Compare paired runs using identical task instances, seeds, candidate limits, and planner versions. Primary comparison: GPT-only versus GPT candidates plus fine-tuned Laya. Controls: heuristic policy, base Laya zero-shot, and sandbox oracle. Splits are grouped by task/site/template into train, validation, calibration, and untouched test.

The specialised model is interesting only if test task success is no worse than GPT-only, unnecessary actions decrease, API tokens do not materially increase, added decision latency stays below 20% of GPT latency, and fine-tuned Laya beats base Laya on both transition prediction and task outcomes. This rule is fixed before training; confidence intervals and raw results accompany every conclusion. At least 30 task instances per held-out template and 5 seeds are required for claims.

Report task success, mean/median steps, unnecessary actions, loops, tool failures, simulated unsafe actions, token/cost, component latency, peak RAM/VRAM, plus accuracy/balanced accuracy/Brier/log loss/ECE and reliability plots where outputs are calibrated probabilities. Ablate history, beliefs, information gain, risk, Laya, zero-shot/fine-tuned, 1/3/5/8 candidates, and raw versus structured formatting.

