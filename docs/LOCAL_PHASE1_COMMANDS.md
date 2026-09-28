# Exact commands for the reference machine

These commands use `/home/kwestog/Documents/code/near_agi` and the downloaded checkpoint at `/mnt/fast-ssd/models/laya/base-english`. They place the heavy Python environment and download cache on the fast SSD.

```bash
cd /home/kwestog/Documents/code/near_agi

mkdir -p /mnt/fast-ssd/laya-dynamics-agent
mkdir -p /mnt/fast-ssd/uv-cache

UV_PROJECT_ENVIRONMENT=/mnt/fast-ssd/laya-dynamics-agent/.venv \
UV_CACHE_DIR=/mnt/fast-ssd/uv-cache \
uv sync --extra test

UV_CACHE_DIR=/mnt/fast-ssd/uv-cache \
uv pip install \
  --python /mnt/fast-ssd/laya-dynamics-agent/.venv/bin/python \
  "laya @ git+https://github.com/NandhaKishorM/laya.git@9d955671415fc19f069b9cc998928075c1f255ec"

export LAYA_CHECKPOINT=/mnt/fast-ssd/models/laya/base-english
export HF_HOME=/mnt/fast-ssd/huggingface

./scripts/verify_laya_checkpoint.sh
/mnt/fast-ssd/laya-dynamics-agent/.venv/bin/lda doctor
/mnt/fast-ssd/laya-dynamics-agent/.venv/bin/pytest
/mnt/fast-ssd/laya-dynamics-agent/.venv/bin/lda demo
/mnt/fast-ssd/laya-dynamics-agent/.venv/bin/lda benchmark --suite smoke --with-laya
```

The last command is a deterministic engineering smoke test. It validates the local Laya load and the complete prediction/policy/environment/storage loop; it is not yet the scientific live-GPT comparison. Reports are written under `reports/`, trajectories to `data/trajectories.sqlite3`, and events to `logs/runs/`.



## Configure `.env`

Copy the template once and edit only the local `.env` file; it is ignored by Git:

```bash
cd /home/kwestog/Documents/code/near_agi
cp .env.example .env
nano .env
```

Use these exact values (replace only the key):

```dotenv
OPENROUTER_API_KEY=sk-or-v1-REPLACE_ME
OPENROUTER_MODEL=openai/gpt-5.6-sol
LAYA_CHECKPOINT=/mnt/fast-ssd/models/laya/base-english
HF_HOME=/mnt/fast-ssd/huggingface
```

## Phase 1 execution order

First validate the benchmark without network or API cost:

```bash
cd /home/kwestog/Documents/code/near_agi
/mnt/fast-ssd/laya-dynamics-agent/.venv/bin/pytest
/mnt/fast-ssd/laya-dynamics-agent/.venv/bin/lda benchmark \
  --suite smoke \
  --provider deterministic \
  --seeds 0,1
```

Then validate the local Laya checkpoint, still without an API call:

```bash
cd /home/kwestog/Documents/code/near_agi
/mnt/fast-ssd/laya-dynamics-agent/.venv/bin/lda benchmark \
  --suite smoke \
  --provider deterministic \
  --seeds 0 \
  --with-laya
```

Finally run the small live OpenRouter comparison using the exact same GPT-5.6 Sol model for direct actions and candidate generation:

```bash
cd /home/kwestog/Documents/code/near_agi
/mnt/fast-ssd/laya-dynamics-agent/.venv/bin/lda benchmark \
  --suite smoke \
  --provider openrouter \
  --model openai/gpt-5.6-sol \
  --seeds 0 \
  --with-laya
```

The live campaign contains `direct_gpt`, `candidates_heuristic`, and `candidates_base_laya` across three tasks. Candidate responses are cached under `data/candidate_cache/`; trajectories are stored in `data/trajectories.sqlite3`, events under `logs/runs/`, and the comparative report under `reports/latest/`. Re-running an identical candidate state replays its cached response rather than spending another candidate-generation call. Direct GPT calls remain independent because they are the control arm.

For a clean CUDA-backed campaign that does not replay candidates from an earlier cache:

```bash
cd /home/kwestog/Documents/code/near_agi

LAYA_DEVICE=cuda \
/mnt/fast-ssd/laya-dynamics-agent/.venv-cu124/bin/lda benchmark \
  --suite smoke \
  --provider openrouter \
  --model openai/gpt-5.6-sol \
  --seeds 0 \
  --with-laya \
  --fresh-candidate-cache
```

The fresh cache is named with the generated campaign identifier. Existing caches are preserved. Within the new campaign, both candidate-selection arms continue to share byte-identical generated candidates.

Press `Ctrl+C` once to request a graceful stop. The current atomic operation finishes, then the partial run is committed with status `interrupted`. OAuth is not part of this milestone. The smoke suite validates the experimental wiring; it is not the full statistically powered campaign.

## Terminal output

The default benchmark output shows one progress line per episode and a compact final table. Complete metrics are always written to `reports/latest/metrics.json`. Use `--json` only when machine-readable terminal output is needed, and `--debug` when diagnosing an exception.
