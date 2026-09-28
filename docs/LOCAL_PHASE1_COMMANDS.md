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



## OpenRouter with GPT-5.6 Sol

```bash
export OPENROUTER_API_KEY=sk-or-v1-...
export OPENROUTER_MODEL=openai/gpt-5.6-sol
export LAYA_CHECKPOINT=/mnt/fast-ssd/models/laya/base-english

/mnt/fast-ssd/laya-dynamics-agent/.venv/bin/lda \
  benchmark --suite smoke \
  --provider openrouter \
  --model openai/gpt-5.6-sol \
  --with-laya
```

Press `Ctrl+C` once to request a graceful stop. The current atomic operation finishes, then the partial run is committed with status `interrupted`. OAuth is not part of this milestone.
