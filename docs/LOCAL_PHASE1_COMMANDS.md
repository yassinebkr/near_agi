# Portable local phase 1 commands

Choose a local storage directory before running these commands. Machine-specific paths belong in the shell or ignored `.env` file and must never be committed.

```bash
cd near_agi
export LDA_STORAGE="${LDA_STORAGE:-${XDG_DATA_HOME:-$HOME/.local/share}/laya-dynamics-agent}"

mkdir -p "$LDA_STORAGE"
mkdir -p "$LDA_STORAGE/uv-cache"

UV_PROJECT_ENVIRONMENT="$LDA_STORAGE/.venv" \
UV_CACHE_DIR="$LDA_STORAGE/uv-cache" \
uv sync --extra test

UV_CACHE_DIR="$LDA_STORAGE/uv-cache" \
uv pip install \
  --python "$LDA_STORAGE/.venv/bin/python" \
  "laya @ git+https://github.com/NandhaKishorM/laya.git@9d955671415fc19f069b9cc998928075c1f255ec"

export LAYA_CHECKPOINT=/path/to/laya/base-english
export HF_HOME=/path/to/huggingface-cache

./scripts/verify_laya_checkpoint.sh
"$LDA_STORAGE/.venv/bin/lda" doctor
"$LDA_STORAGE/.venv/bin/pytest"
"$LDA_STORAGE/.venv/bin/lda" demo
"$LDA_STORAGE/.venv/bin/lda" benchmark --suite smoke --with-laya
```

The last command is a deterministic engineering smoke test. It validates the local Laya load and the complete prediction/policy/environment/storage loop; it is not yet the scientific live-GPT comparison. Reports are written under `reports/`, trajectories to `data/trajectories.sqlite3`, and events to `logs/runs/`.



## Configure `.env`

Copy the template once and edit only the local `.env` file; it is ignored by Git:

```bash
cd near_agi
cp .env.example .env
nano .env
```

Use these exact values (replace only the key):

```dotenv
OPENROUTER_API_KEY=sk-or-v1-REPLACE_ME
OPENROUTER_MODEL=openai/gpt-5.6-sol
LAYA_CHECKPOINT=/path/to/laya/base-english
HF_HOME=/path/to/huggingface-cache
```

## Phase 1 execution order

First validate the benchmark without network or API cost:

```bash
cd near_agi
"$LDA_STORAGE/.venv/bin/pytest"
"$LDA_STORAGE/.venv/bin/lda" benchmark \
  --suite smoke \
  --provider deterministic \
  --seeds 0,1
```

Then validate the local Laya checkpoint, still without an API call:

```bash
cd near_agi
"$LDA_STORAGE/.venv/bin/lda" benchmark \
  --suite smoke \
  --provider deterministic \
  --seeds 0 \
  --with-laya
```

Finally run the small live OpenRouter comparison using the exact same GPT-5.6 Sol model for direct actions and candidate generation:

```bash
cd near_agi
"$LDA_STORAGE/.venv/bin/lda" benchmark \
  --suite smoke \
  --provider openrouter \
  --model openai/gpt-5.6-sol \
  --seeds 0 \
  --with-laya
```

The live campaign contains `direct_gpt`, `candidates_heuristic`, and `candidates_base_laya` across three tasks. Candidate responses are cached under `data/candidate_cache/`; trajectories are stored in `data/trajectories.sqlite3`, events under `logs/runs/`, and the comparative report under `reports/latest/`. Re-running an identical candidate state replays its cached response rather than spending another candidate-generation call. Direct GPT calls remain independent because they are the control arm.

For a clean CUDA-backed campaign that does not replay candidates from an earlier cache:

```bash
cd near_agi

LAYA_DEVICE=cuda \
"$LDA_STORAGE/.venv-cu124/bin/lda" benchmark \
  --suite smoke \
  --provider openrouter \
  --model openai/gpt-5.6-sol \
  --seeds 0 \
  --with-laya \
  --fresh-candidate-cache
```

The fresh cache is named with the generated campaign identifier. Existing caches are preserved. Within the new campaign, both candidate-selection arms continue to share byte-identical generated candidates.

The next development campaign uses the nine-task challenge suite:

```bash
cd near_agi

LAYA_DEVICE=cuda \
"$LDA_STORAGE/.venv-cu124/bin/lda" benchmark \
  --suite challenge \
  --provider openrouter \
  --model openai/gpt-5.6-sol \
  --seeds 0 \
  --with-laya \
  --fresh-candidate-cache
```

This produces 27 episodes: nine direct-GPT episodes, nine heuristic-selector episodes, and nine base-Laya episodes. Start with one seed. Multi-seed execution is deferred until the challenge results have been inspected and the held-out dataset exists.

Press `Ctrl+C` once to request a graceful stop. The current atomic operation finishes, then the partial run is committed with status `interrupted`. OAuth is not part of this milestone. The smoke suite validates the experimental wiring; it is not the full statistically powered campaign.

## Frozen final benchmark

Run a one-seed pilot first to validate provider compatibility and estimate current cost. The report will intentionally say `NON-COMPLIANT PILOT`:

```bash
cd near_agi

LAYA_DEVICE=cuda \
"$LDA_STORAGE/.venv-cu124/bin/lda" benchmark \
  --suite final \
  --provider openrouter \
  --model openai/gpt-5.6-sol \
  --seeds 0 \
  --with-laya \
  --fresh-candidate-cache
```

After inspecting the pilot, run the pre-registered campaign:

```bash
cd near_agi

LAYA_DEVICE=cuda \
"$LDA_STORAGE/.venv-cu124/bin/lda" benchmark \
  --suite final \
  --provider openrouter \
  --model openai/gpt-5.6-sol \
  --seeds 0,1,2,3,4 \
  --with-laya \
  --fresh-candidate-cache
```

The compliant campaign contains 1,350 episodes. Extrapolating from `benchmark-795a36f9`, budget approximately 75–120 minutes and about $3.4, with provider-dependent variance. The one-seed pilot contains 270 episodes and is expected to cost roughly $0.7. Do not combine pilot results with the compliant campaign.

## Terminal output

The default benchmark output shows one progress line per episode and a compact final table. Complete metrics are always written to `reports/latest/metrics.json`. Use `--json` only when machine-readable terminal output is needed, and `--debug` when diagnosing an exception.
