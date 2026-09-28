# Experimental hypothesis

**H0:** a specialised Laya model does not improve the agent enough to justify its complexity, latency, and operational cost.

**H1:** a local decision model trained on `(state_before, action, state_after, outcome)` estimates useful transition properties well enough to improve a GPT planner's decisions.

The unit of evidence is a held-out sandbox task template, not an individual row. A smoke run is engineering validation only.

