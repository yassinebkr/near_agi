# Experimental hypothesis

**H0:** a specialised Laya model does not improve the agent enough to justify its complexity, latency, and operational cost.

**H1:** a local decision model trained on `(state_before, action, state_after, outcome)` estimates useful transition properties well enough to improve a GPT planner's decisions.

The evidence unit is a held-out task-template and seed pair. Smoke and challenge are runtime-transfer gates. They do not establish the final claim.

The completed v2bis checkpoint passed its offline, smoke, and challenge gates. The registered `final-v001` campaign remains incomplete at 705 of 1,800 arm episodes, so neither hypothesis has been accepted or rejected. No inference is drawn from the incomplete prefix.
