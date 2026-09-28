# Toward an Agent with an Abstract Dynamics Model — LLM + Post-Trained Laya

## 1. Starting Point

The initial question was:

> Why do companies keep pushing LLMs toward AGI when an autoregressive LLM seems insufficient, by itself, to produce general intelligence?

The important conclusion is that there is currently neither proof that scaling LLMs will necessarily lead to AGI nor proof that it cannot.

The current paradigm is already moving beyond:

```text
larger LLM -> better performance
```

toward systems composed of several building blocks:

```text
foundation model
+ reinforcement learning
+ tools
+ search
+ memory
+ planning
+ agents
+ predictive models of the environment
+ interaction with the world
```

The interesting hypothesis is therefore not necessarily:

```text
one enormous LLM = AGI
```

but rather:

```text
a cognitive system = several specialized subsystems
```


## 2. Why the Concept of a “World Model” Is Often Misunderstood

The term *world model* can easily suggest an internal visual simulator:

```text
virtual agent
+ virtual computer
+ 3D environment
+ detailed simulation of the future
```

That would be unnecessarily expensive for many abstract tasks.

A world model useful to a general agent can be much simpler. Its minimal formulation is:

\[
(s_t, a_t) \rightarrow \hat{s}_{t+1}
\]

where:

- \(s_t\) is the current state;
- \(a_t\) is the contemplated action;
- \(\hat{s}_{t+1}\) is the predicted future state.

The state does not need to be an image or a 3D scene. It can be an abstract representation of:

- what the agent knows;
- what it does not know;
- its objective;
- the hypotheses it is considering;
- the available tools;
- actions already taken;
- confidence in specific information;
- the state of a software environment;
- the likely consequences of an action.


## 3. Example: Internet Research

Suppose the agent must find the maximum voltage of a component.

The state could be:

```yaml
goal:
  find_absolute_max_voltage: true

knowledge:
  manufacturer: "Texas Instruments"
  max_voltage: null

evidence:
  forum_result_found: true
  primary_datasheet_found: false

confidence:
  voltage_value: 0.25
```

Possible actions:

```text
A1 = open a forum result
A2 = search the manufacturer's website
A3 = open the official datasheet
A4 = answer immediately
```

An abstract world model could estimate:

```text
A1 -> possible information, but from a secondary source
A2 -> high probability of reaching a primary source
A3 -> high probability of reducing uncertainty
A4 -> high risk of answering with insufficiently verified information
```

It therefore does not simulate:

```text
hand -> mouse -> keyboard -> browser -> pixels
```

It simulates:

```text
knowledge state + action
-> probable knowledge state
```


## 4. “Simulating Every Possibility” Is Not Necessary

A general intelligence would probably not benefit from exhaustively simulating every possibility.

A more realistic approach is:

```text
generate a few plausible actions
        |
        v
predict their consequences
        |
        v
evaluate promising branches
        |
        v
expand only selected branches
```

That is:

\[
s_t \rightarrow \{a_1,a_2,a_3,\ldots\}
\]

then:

\[
(s_t,a_i)\rightarrow \hat{s}_{t+1}^{(i)}
\]

This produces a form of counterfactual planning:

> “If I do X, what will probably happen?”


## 5. Several Levels of “World Model”

For a general agent, it may be more useful to speak of **dynamics models** than of a single universal world model.

### 5.1 The Physical World

```text
I push the object
-> it falls
-> it may break
```

This is especially useful in robotics.

### 5.2 The Software World

```text
git reset --hard
-> local changes are lost
```

```text
POST /purchase
-> a transaction may be created
```

### 5.3 The Informational World

```text
open official documentation
-> obtain new information
-> probably reduce uncertainty
```

### 5.4 The Causal World

```text
change this parameter
-> temperature increases
-> stability may decrease
```

### 5.5 The Social World

```text
send this message
-> probable response
-> expected new information
```

### 5.6 Epistemic State

The agent must also be able to model:

```text
what I know
what I believe
what I assume
what I do not know
what I must verify
```

This is particularly important for a hypothetical AGI.


## 6. The Truly Interesting Loop: Prediction -> Action -> Observation

An architecture richer than a standalone LLM could operate as follows:

```text
observation
    |
    v
state encoder
    |
    v
s_t
    |
    +----> candidate action generation
    |
    +----> consequence prediction
    |
    +----> value / risk / information-gain estimation
    |
    v
action selection
    |
    v
real environment
    |
    v
s_{t+1}
```

It would then compare:

\[
\hat{s}_{t+1}
\]

with:

\[
s_{t+1}
\]

The fundamental signal becomes:

\[
e_t = s_{t+1} - \hat{s}_{t+1}
\]

or, more generally, a structured prediction error. This loop enables the system to correct its model progressively.


## 7. Why Laya Is Interesting in This Context

Laya is a non-autoregressive, decision-oriented model.

At the time of this design, the main English checkpoint uses a ModernBERT-large backbone and a decision head, for approximately 421 million parameters.

It supports typed decisions including:

- `choice`;
- `score`;
- `noul`, or probabilistic yes/no.

It is designed to evaluate several decisions in one pass instead of generating an answer token by token.

The goal is therefore not to replace the generative LLM. The proposed separation is:

```text
LLM
-> invents / proposes / reasons / formulates

Laya
-> classifies / scores / filters / estimates quickly
```

This distinction is fundamental.


## 8. Laya Is Not Naturally a World Model

Laya performs something closer to:

\[
f(s,q)\rightarrow P(y)
\]

whereas a true transition model would seek:

\[
f(s,a)\rightarrow \hat{s}_{t+1}
\]

We should therefore not claim that Laya is already a world model.

It can, however, be post-trained to learn **properties of transitions**. For example:

```text
Will this action reveal useful information?
Will this action reduce uncertainty?
Will this action advance the goal?
Is this action reversible?
Can this action modify external state?
Is this action risky?
Will another observation probably be required?
Is the current hypothesis likely to survive this action?
```

We can then approximate:

\[
P(\Delta s \mid s,a)
\]

through a collection of typed decisions.


## 9. Proposed Architecture

```text
                         +------------------+
                         |   Observation    |
                         +---------+--------+
                                   |
                                   v
                         +------------------+
                         |  State Builder   |
                         +---------+--------+
                                   |
                                  s_t
                                   |
               +-------------------+-------------------+
               |                                       |
               v                                       v
        +--------------+                       +----------------+
        | GPT / LLM    |                       | episodic memory|
        | proposes     |                       | trajectories   |
        | actions      |                       +----------------+
        +------+-------+
               |
        {a1,a2,a3,...}
               |
               v
        +------------------------------------------+
        |            post-trained Laya            |
        |                                          |
        | P(goal_progress)                         |
        | P(success)                               |
        | P(information_gain)                      |
        | P(risk)                                  |
        | P(reversible)                            |
        | P(needs_observation)                     |
        +------------------+-----------------------+
                           |
                           v
                    planner / policy
                           |
                           v
                       action a*
                           |
                           v
                     environment
                           |
                           v
                         s_t+1
                           |
              +------------+-------------+
              |                          |
              v                          v
       prediction error            trajectory log
              |                          |
              +------------+-------------+
                           |
                           v
                     training data
```


## 10. A More Ambitious Extension

In a second phase, a latent transition representation could be added:

\[
f(s_t,a_t)\rightarrow\Delta z
\]

then:

\[
\hat z_{t+1} = z_t + \Delta z
\]

This could theoretically enable several steps to be rolled out without acting in the real environment:

\[
z_t \xrightarrow{a_1} \hat z_{t+1} \xrightarrow{a_2} \hat z_{t+2}
\]

This part is substantially more experimental. The first POC must not depend on it.


## 11. Why This Architecture May Be Better Than a Generic “Judge”

A previous experiment with JEV in a Shopify agent produced mixed results.

The general problem with inserting a *judge* into an agent is that it may:

- approximately repeat what the LLM can already do;
- introduce latency;
- add a second failure point;
- block good decisions;
- correlate poorly with real task success;
- never learn from the agent's trajectories;
- increase API cost when it is remote.

Our use of Laya must therefore be different.

### Bad Objective

```text
GPT proposes an action
-> Laya arbitrarily says yes or no
```

### Proposed Objective

```text
GPT proposes several actions
-> Laya predicts measurable properties
   of their consequences
-> the planner chooses
-> the action is executed
-> the real result is observed
-> prediction error is measured
-> the dataset is enriched
-> the model is retrained
```

Laya then becomes a **learned local dynamics/value function** rather than a simple judge.


## 12. Why Local Inference Is Particularly Interesting

In this architecture, Laya may be queried very frequently:

```text
10 candidate actions
x
6 properties
x
several steps
```

A paid API service quickly becomes expensive and adds network latency.

Local inference provides:

- no API cost per decision;
- low latency;
- privacy;
- the ability to batch many decisions;
- full checkpoint control;
- post-training capability;
- reproducibility;
- offline operation.

Based on practical experience with the target machine, the full Laya model already fits in VRAM with a small inference footprint. The POC can therefore start with the full model rather than being designed around a reduced checkpoint.


## 13. The Dataset That Actually Matters

The dataset should not consist solely of human preferences.

Its fundamental unit should be:

```text
(state_before, action, state_after, outcome)
```

Example:

```json
{
  "state_before": {
    "goal": "find official maximum voltage",
    "primary_source_found": false,
    "confidence": 0.31
  },
  "action": {
    "tool": "open_url",
    "target": "manufacturer_datasheet"
  },
  "state_after": {
    "primary_source_found": true,
    "confidence": 0.94
  },
  "outcome": {
    "goal_progress": 0.92,
    "information_gain": 0.88,
    "external_side_effect": false,
    "success": true
  }
}
```

Some labels can be generated automatically from state deltas.


## 14. Preventing Label Leakage

The model must not learn trivial shortcuts.

Examples of bad correlations:

```text
if history != empty -> DONE
```

or:

```text
if text contains "success" -> action succeeded
```

We will need to:

- separate sites and tasks between training and testing;
- introduce counterexamples;
- avoid overly repetitive templates;
- test on unseen trajectories;
- perform ablations;
- measure calibration, not only accuracy.

Laya's official browser-agent fine-tuning documentation likewise emphasizes input formatting, genuine `DONE` states, intermediate negatives, and on-policy corrections.


## 15. Recommended POC

The first prototype must remain controllable.

### Environment

A small local or sandboxed web environment with tasks such as:

```text
find information
navigate to a page
fill out a form
compare two pieces of information
identify a primary source
correct a hypothesis after an observation
```

### Baselines

#### A. GPT Alone

```text
state -> GPT -> action
```

#### B. GPT + Zero-Shot Laya

```text
state -> GPT candidates -> base Laya -> action
```

#### C. GPT + Post-Trained Laya

```text
state -> GPT candidates -> specialized Laya -> action
```

#### D. Optional: GPT + JEV

Only if the previous integration is easy enough to reproduce cleanly.

This baseline could compare:

- cloud versus local;
- generic versus specialized;
- cost;
- latency;
- decision quality.


## 16. Metrics

The POC must produce quantitative measurements.

### Success

\[
TaskSuccessRate
\]

### Efficiency

```text
required actions
unnecessary actions
retries
loops
tool errors
```

### Latency

```text
LLM latency
Laya latency
planner latency
total task latency
```

### Cost

```text
input tokens
output tokens
API cost
local GPU time
```

### Predictive Quality

```text
Brier score
log loss
accuracy
top-k accuracy
ECE / calibration error
```

### Transition Model

Compare predictions:

```text
predicted:
goal_progress = 0.84

observed:
goal_progress = 0.72
```

and track the error across successive retraining cycles.


## 17. Project Success Criterion

The experiment must be allowed to fail. This is essential.

The interesting result is not:

> “We built an embryonic AGI.”

It is:

> “We tested whether a small local decision model specialized on an agent's real transitions enables an LLM planner to make better decisions with fewer actions and lower cost.”

A convincing initial success could be:

```text
TaskSuccess >= GPT-only baseline
AND
unnecessary actions decrease
AND
API cost decreases or remains stable
AND
latency added by Laya remains low
AND
Laya scores are genuinely calibrated
AND
fine-tuning clearly outperforms zero-shot
```


## 18. Scientific Hypothesis

### H0 — Null Hypothesis

> Adding a specialized Laya model does not improve agent behavior enough to justify its complexity.

### H1 — Experimental Hypothesis

> A small local, non-generative model trained on the agent's real transitions can learn useful properties of task dynamics and improve a generative LLM's planning.


## 19. Proposed Phases

### Phase 0 — Audit

- reproduce Laya locally;
- measure VRAM, RAM, and latency;
- reproduce its typed outputs;
- inspect its fine-tuning format;
- document hardware and exact versions.

### Phase 1 — Deterministic Environment

- local web environment;
- explicit state;
- explicit actions;
- reproducible logs;
- GPT as planner.

### Phase 2 — Collection

Record:

```text
s_t
candidate_actions
selected_action
predictions
actual_s_t+1
task_outcome
```

### Phase 3 — Zero-Shot Laya

Measure objectively whether it adds value.

### Phase 4 — Post-Training

Build a dataset from real trajectories.

### Phase 5 — Calibration

Calibrate probabilities on a dedicated split.

### Phase 6 — Blind Benchmark

Test on:

- new tasks;
- new sites or environments;
- new phrasings.

### Phase 7 — DAgger / On-Policy

Collect mistakes made by the current policy itself, add them to the dataset, and retrain.

### Phase 8 — Multi-Step Planning

Only after validating the one-step predictor.


## 20. What Not to Do Initially

Do not begin with:

- a completely unrestricted Internet browser;
- a complex multi-agent system;
- elaborate vector memory;
- a video world model;
- end-to-end reinforcement learning;
- an uninterpretable latent transition model;
- production Shopify;
- irreversible actions;
- dozens of tools.

That would make it impossible to determine which component improves or degrades the system.


## 21. Future Connection to the Shopify Agent

Once the hypothesis has been validated in the sandbox, the system could be applied to the existing Shopify agent.

Example state:

```yaml
goal:
  improve_homepage_conversion: true

store:
  theme: "..."
  unsaved_changes: true

observations:
  mobile_layout_issue: true
  broken_asset: false

history:
  - inspect_theme
  - read_homepage
```

Candidate actions:

```text
edit_section
inspect_css
preview_mobile
publish_theme
ask_user
```

A specialized Laya model could estimate:

```text
goal_progress
risk
reversibility
need_for_observation
probability_of_success
```

The crucial point is that:

```text
publish_theme
```

and:

```text
inspect_css
```

must clearly not receive the same risk profile.


## 22. Final Architectural Principle

The goal is not:

```text
LLM + a second model that votes
```

The goal is:

```text
GENERATIVE MODEL
    |
    | proposes
    v
candidate actions
    |
    v
LEARNED DYNAMICS / VALUE MODEL
    |
    | predicts consequences
    v
planner
    |
    v
real action
    |
    v
observation
    |
    v
prediction error
    |
    v
learning
```

This loop turns the experiment into something much more interesting than a simple classification system.


## 23. Useful Technical References

- NandhaKishorM/laya — the main Laya repository.
- Laya documentation — typed `choice`, `score`, and `noul` decisions, fine-tuning, and calibration.
- `docs/finetune_browser_agent.md` — a particularly relevant example of specializing Laya for browser-agent action selection.
- Mind2Web — a dataset of web trajectories that may be useful as a complementary source.
- DAgger — a useful approach for adding corrections on states actually encountered by the current policy.


## Conclusion

The project is not an attempt to claim that we are building AGI.

It isolates a much more serious question:

> **Can a fast, local, non-generative model such as Laya, post-trained on the real consequences of an agent's actions, become a useful enough abstract-dynamics predictor to improve an LLM's planning?**

If the answer is yes, it provides an interesting building block:

```text
generative LLM
+
local decision/dynamics model
+
trajectory memory
+
prediction -> action -> observation -> correction loop
```

That would already be a substantial result.
