# Portable local commands

Machine-specific paths belong in the shell or ignored `.env` file. They must not be committed.

## Environment setup

```bash
cd near_agi
export LDA_STORAGE="${LDA_STORAGE:-${XDG_DATA_HOME:-$HOME/.local/share}/laya-dynamics-agent}"

mkdir -p "$LDA_STORAGE/uv-cache"

UV_PROJECT_ENVIRONMENT="$LDA_STORAGE/.venv" \
UV_CACHE_DIR="$LDA_STORAGE/uv-cache" \
uv sync --extra test --extra laya

cp .env.example .env
uv run lda doctor
uv run pytest -q
```

For the validated local CUDA environment, follow [CUDA setup](CUDA_SETUP.md). Do not rely on an implicit CPU fallback for a timed campaign.

## Deterministic engineering checks

These checks require no API key:

```bash
uv run lda demo

uv run lda benchmark \
  --suite smoke \
  --provider deterministic \
  --seeds 0

LAYA_DEVICE=cuda \
LAYA_CHECKPOINT=/path/to/base-english \
uv run lda benchmark \
  --suite smoke \
  --provider deterministic \
  --seeds 0 \
  --with-laya
```

They validate plumbing, persistence, and checkpoint loading. They do not measure live GPT or establish Laya performance.

## Live four-arm development check

Set secrets in `.env` or the shell:

```dotenv
OPENROUTER_API_KEY=replace_me
OPENROUTER_MODEL=openai/gpt-5.6-sol
OPENROUTER_MAX_TOKENS=4096
LAYA_DEVICE=cuda
```

Run a four-arm smoke or challenge with explicit checkpoints:

```bash
LAYA_DEVICE=cuda \
uv run lda benchmark \
  --suite challenge \
  --provider openrouter \
  --model openai/gpt-5.6-sol \
  --seeds 0 \
  --base-laya-checkpoint /path/to/base-english \
  --finetuned-laya-checkpoint /path/to/laya-dynamics-v002bis \
  --fresh-candidate-cache
```

This produces 36 episodes: nine tasks across direct GPT, heuristic, base Laya, and fine-tuned Laya. The three candidate selectors share a campaign cache. Direct GPT remains independent.

`--fresh-candidate-cache` creates a new campaign-scoped cache without deleting prior evidence. Add `--campaign-id ID` for a stable long run and use the identical command with `--resume` after interruption.

The first `Ctrl+C` requests a graceful stop after the current atomic planner, predictor, or environment operation. Completed episodes remain durable in SQLite, JSONL, and the campaign checkpoint.

## Frozen final campaign

The registered v2bis final campaign has already started. It contains 1,800 episodes: 90 tasks, five seeds, and four arms. It is paused after 705 durable episodes because the OpenRouter budget was exhausted.

Do not start a new final campaign or create a fresh cache. Restore and resume the archived campaign by following [Nebius post-training runbook](NEBIUS_POSTTRAIN_RUNBOOK.md). A new one-seed pilot would consume API budget and cannot be combined with the registered result.

## Outputs

- `data/trajectories.sqlite3`: durable transitions.
- `data/candidate_cache/`: campaign candidate provenance and original generation timing.
- `logs/runs/<run_id>/events.jsonl`: structured episode events.
- `reports/<campaign>/checkpoint.json`: atomic resume state.
- `reports/latest/metrics.json`: complete machine-readable report.
- `reports/latest/summary.md`: readable report.

Terminal output is human-readable by default. Use `--json` for complete terminal JSON and `--debug` for exception tracebacks.
