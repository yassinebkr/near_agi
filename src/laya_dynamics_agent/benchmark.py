from __future__ import annotations

import hashlib
import json
import random
import re
import uuid
import time
import warnings
from pathlib import Path
from statistics import mean, median, pstdev
from typing import Any

from .planners import CachedPlanner, DeterministicDirectPlanner, DeterministicPlanner, OpenAIDirectPlanner, OpenAIPlanner, OpenRouterDirectPlanner, OpenRouterPlanner
from .policies import GPTOnlyPolicy, GreedyUtilityPolicy, UtilityWeights
from .predictors import HeuristicPredictor, LayaPredictor
from .runner_impl import run_episode
from .sandbox import LABELER_VERSION, SUITES, SandboxWebEnvironment, TASKS
from .storage_v2 import TrajectoryStore

PROMPT_VERSION = "planner-v001"
DIRECT_PROMPT_VERSION = "direct-v001"
DEFAULT_SEEDS = (0,)


def _planner_pair(provider: str, model: str | None) -> tuple[Any, Any]:
    if provider == "deterministic":
        return DeterministicPlanner(), DeterministicDirectPlanner()
    if provider == "openai":
        return OpenAIPlanner(model=model), OpenAIDirectPlanner(model=model)
    if provider == "openrouter":
        return OpenRouterPlanner(model=model), OpenRouterDirectPlanner(model=model)
    raise ValueError(f"unknown planner provider: {provider}")


def _bootstrap_mean_ci(values: list[float], *, samples: int = 2000, seed: int = 0) -> list[float]:
    if not values:
        return [0.0, 0.0]
    rng = random.Random(seed)
    size = len(values)
    estimates = sorted(mean(values[rng.randrange(size)] for _ in range(size)) for _ in range(samples))
    return [estimates[int(samples * 0.025)], estimates[int(samples * 0.975)]]


def _metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    successes = [float(row["success"]) for row in rows]
    steps = [float(row["steps"]) for row in rows]
    return {
        "episodes": len(rows), "success_rate": mean(successes), "success_std": pstdev(successes),
        "success_ci95": _bootstrap_mean_ci(successes), "mean_steps": mean(steps),
        "median_steps": median(steps), "steps_std": pstdev(steps), "steps_ci95": _bootstrap_mean_ci(steps),
        "unsafe_actions": sum(row["unsafe_actions"] for row in rows),
        "unnecessary_actions": sum(row["unnecessary_actions"] for row in rows),
        "planner_latency_ms": sum(row["planner_latency_ms"] for row in rows),
        "predictor_latency_ms": sum(row["predictor_latency_ms"] for row in rows), "usage": _sum_usage(rows),
    }


def _paired_comparisons(results: list[dict[str, Any]]) -> dict[str, Any]:
    indexed = {(row["task_id"], row["seed"], row["mode"]): row for row in results}
    comparisons: dict[str, Any] = {}
    pairs = (("candidates_heuristic", "direct_gpt"), ("candidates_base_laya", "direct_gpt"), ("candidates_base_laya", "candidates_heuristic"))
    for left, right in pairs:
        keys = sorted((task_id, seed) for task_id, seed, mode in indexed if mode == left and (task_id, seed, right) in indexed)
        success_delta = [float(indexed[task_id, seed, left]["success"]) - float(indexed[task_id, seed, right]["success"]) for task_id, seed in keys]
        steps_delta = [float(indexed[task_id, seed, left]["steps"]) - float(indexed[task_id, seed, right]["steps"]) for task_id, seed in keys]
        comparisons[f"{left}_vs_{right}"] = {
            "pairs": len(keys), "success_delta": mean(success_delta) if success_delta else 0.0,
            "success_delta_ci95": _bootstrap_mean_ci(success_delta), "steps_delta": mean(steps_delta) if steps_delta else 0.0,
            "steps_delta_ci95": _bootstrap_mean_ci(steps_delta),
        }
    return comparisons


def _aggregate(results: list[dict[str, Any]]) -> dict[str, Any]:
    by_mode: dict[str, list[dict[str, Any]]] = {}
    by_template: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for row in results:
        by_mode.setdefault(row["mode"], []).append(row)
        by_template.setdefault(row["template_id"], {}).setdefault(row["mode"], []).append(row)
    return {
        "episodes": len(results), "modes": {mode: _metrics(rows) for mode, rows in by_mode.items()},
        "templates": {template: {mode: _metrics(rows) for mode, rows in modes.items()} for template, modes in by_template.items()},
        "paired_comparisons": _paired_comparisons(results),
    }


def _sum_usage(rows: list[dict[str, Any]]) -> dict[str, float]:
    total: dict[str, float] = {}
    for row in rows:
        for key, value in row.get("usage", {}).items():
            if isinstance(value, (int, float)):
                total[key] = total.get(key, 0) + value
    return total


async def run_benchmark_suite(*, include_laya: bool, provider: str, model: str | None, shutdown: Any, seeds: tuple[int, ...] = DEFAULT_SEEDS, progress: Any = print, fresh_candidate_cache: bool = False, suite: str = "smoke") -> dict[str, Any]:
    campaign_id = f"benchmark-{uuid.uuid4().hex[:8]}"
    generator, direct = _planner_pair(provider, model)
    resolved_model = getattr(generator, "model", "deterministic")
    safe_model = re.sub(r"[^a-zA-Z0-9_.-]+", "_", str(resolved_model))
    cache_suffix = f"-{campaign_id}" if fresh_candidate_cache else ""
    cache_path = Path("data/candidate_cache") / f"{provider}-{safe_model}{cache_suffix}.json"
    cached = CachedPlanner(generator, cache_path, prompt_version=PROMPT_VERSION)
    task_ids = SUITES[suite]
    task_manifest = {task_id: TASKS[task_id] for task_id in task_ids}
    task_manifest_hash = hashlib.sha256(json.dumps(task_manifest, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    progress(f"Benchmark {campaign_id} | provider={provider} | model={resolved_model}")
    progress(f"Suite: {suite} | tasks: {len(task_ids)} | seeds: {','.join(map(str, seeds))} | Laya: {'yes' if include_laya else 'no'}")
    laya_runtime: dict[str, Any] | None = None
    configs: list[tuple[str, Any, Any | None, Any]] = [
        ("direct_gpt", direct, None, GPTOnlyPolicy()),
        ("candidates_heuristic", cached, HeuristicPredictor(), GreedyUtilityPolicy()),
    ]
    if include_laya:
        checkpoint = __import__("os").getenv("LAYA_CHECKPOINT", "laya")
        progress(f"Loading Laya checkpoint: {checkpoint}")
        with warnings.catch_warnings(record=True) as caught_warnings:
            warnings.simplefilter("always")
            laya_predictor = LayaPredictor(checkpoint)
        for warning in caught_warnings:
            first_line = str(warning.message).splitlines()[0]
            progress(f"WARNING: {first_line}")
        laya_runtime = {"checkpoint": checkpoint, "device": laya_predictor.device, "load_time_ms": laya_predictor.load_time_ms}
        progress(f"Laya device: {laya_predictor.device} | load={laya_predictor.load_time_ms / 1000:.1f}s")
        configs.append(("candidates_base_laya", cached, laya_predictor, GreedyUtilityPolicy()))
    store = TrajectoryStore(Path("data/trajectories.sqlite3"), Path("logs/runs"))
    results: list[dict[str, Any]] = []
    total_episodes = len(seeds) * len(task_ids) * len(configs)
    episode_number = 0
    try:
        for seed in seeds:
            cached.seed = seed
            if hasattr(generator, "seed"):
                generator.seed = seed
            if hasattr(direct, "seed"):
                direct.seed = seed
            for task_id in task_ids:
                for mode, planner, predictor, policy in configs:
                    if shutdown.event.is_set():
                        break
                    episode_number += 1
                    progress(f"[{episode_number}/{total_episodes}] {task_id} | {mode} | seed={seed} ...")
                    episode_started = time.perf_counter()
                    run_id = f"{campaign_id}-{mode}-{task_id}-s{seed}"
                    config = {"campaign_id": campaign_id, "mode": mode, "task_id": task_id, "template_id": TASKS[task_id]["template_id"], "seed": seed, "provider": provider, "model": getattr(planner, "model", resolved_model), "prompt_version": DIRECT_PROMPT_VERSION if mode == "direct_gpt" else PROMPT_VERSION, "labeler_version": LABELER_VERSION, "laya_runtime": laya_runtime, "candidate_count": 1 if mode == "direct_gpt" else 5, "weights": UtilityWeights().__dict__, "candidate_cache": str(cache_path)}
                    store.start_run(run_id, config)
                    try:
                        result = await run_episode(run_id=run_id, task_id=task_id, environment=SandboxWebEnvironment(), planner=planner, predictor=predictor, policy=policy, store=store, max_actions=1 if mode == "direct_gpt" else 5, stop_requested=shutdown.event.is_set)
                        result.update(run_id=run_id, campaign_id=campaign_id, mode=mode, seed=seed, template_id=config["template_id"], provider=provider, model=config["model"], prompt_version=config["prompt_version"])
                        status = "interrupted" if result["stop_reason"] == "interrupted" else "completed"
                        store.finish_run(run_id, status, result)
                        results.append(result)
                        marker = "OK" if result["success"] else "FAIL"
                        progress(f"    {marker} | steps={result['steps']} | stop={result['stop_reason']} | elapsed={time.perf_counter() - episode_started:.1f}s")
                    except BaseException as exc:
                        failed = {"run_id": run_id, "campaign_id": campaign_id, "mode": mode, "task_id": task_id, "seed": seed, "steps": 0, "stop_reason": "error", "error": type(exc).__name__}
                        store.finish_run(run_id, "error", failed)
                        progress(f"    FAILED | elapsed={time.perf_counter() - episode_started:.1f}s")
                        raise
                if shutdown.event.is_set():
                    break
            if shutdown.event.is_set():
                break
    finally:
        store.close()
    expected_episodes = len(task_ids) * len(seeds) * len(configs)
    final_protocol = {
        "heldout_suite": suite == "final", "declared_seeds": list(seeds) == [0, 1, 2, 3, 4],
        "fresh_candidate_cache": fresh_candidate_cache, "laya_enabled": include_laya,
        "live_provider": provider in {"openai", "openrouter"},
        "complete": not shutdown.event.is_set() and len(results) == expected_episodes,
    }
    report = {"campaign_id": campaign_id, "suite": suite, "suite_version": {"smoke": "smoke-v001", "challenge": "challenge-v001", "final": "final-v001"}[suite], "task_manifest_sha256": task_manifest_hash, "provider": provider, "model": resolved_model, "prompt_versions": {"direct": DIRECT_PROMPT_VERSION, "candidates": PROMPT_VERSION}, "labeler_version": LABELER_VERSION, "seeds": list(seeds), "tasks": list(task_ids), "laya_runtime": laya_runtime, "candidate_cache": str(cache_path), "fresh_candidate_cache": fresh_candidate_cache, "cache_hits": cached.cache_hits, "cache_misses": cached.cache_misses, "candidate_generation_usage": cached.total_usage, "candidate_generation_usage_current_run": cached.fresh_usage, "candidate_generation_usage_replayed": cached.replayed_usage, "candidate_generation_usage_unknown_entries": cached.unknown_usage_entries, "direct_usage": _sum_usage([r for r in results if r["mode"] == "direct_gpt"]), "interrupted": shutdown.event.is_set(), "aggregate": _aggregate(results), "results": results}
    report["final_protocol"] = {**final_protocol, "compliant": all(final_protocol.values())}
    root = Path("reports") / campaign_id
    root.mkdir(parents=True, exist_ok=True)
    (root / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")
    latest = Path("reports/latest")
    latest.mkdir(parents=True, exist_ok=True)
    (latest / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")
    lines = ["# Benchmark", "", f"Campaign: `{campaign_id}` · suite: `{suite}` · provider: `{provider}` · model: `{resolved_model}` · prompts: `{DIRECT_PROMPT_VERSION}` / `{PROMPT_VERSION}`", "", "| mode | episodes | success | success 95% CI | mean steps | steps 95% CI | unsafe | unnecessary |", "|:--|--:|--:|:--|--:|:--|--:|--:|"]
    for mode, metrics in report["aggregate"]["modes"].items():
        success_ci = metrics["success_ci95"]
        steps_ci = metrics["steps_ci95"]
        lines.append(f"| {mode} | {metrics['episodes']} | {metrics['success_rate']:.3f} | [{success_ci[0]:.3f}, {success_ci[1]:.3f}] | {metrics['mean_steps']:.2f} | [{steps_ci[0]:.2f}, {steps_ci[1]:.2f}] | {metrics['unsafe_actions']} | {metrics['unnecessary_actions']} |")
    lines.extend(["", "## Per-template results"])
    for template, modes in report["aggregate"]["templates"].items():
        lines.extend(["", f"### `{template}`", "", "| mode | episodes | success | mean steps |", "|:--|--:|--:|--:|"])
        for mode, metrics in modes.items():
            lines.append(f"| {mode} | {metrics['episodes']} | {metrics['success_rate']:.3f} | {metrics['mean_steps']:.2f} |")
    lines.extend(["", "## Paired comparisons", "", "Positive deltas mean the left-hand arm is higher.", "", "| comparison | pairs | success delta | success delta 95% CI | steps delta | steps delta 95% CI |", "|:--|--:|--:|:--|--:|:--|"])
    for name, comparison in report["aggregate"]["paired_comparisons"].items():
        success_ci = comparison["success_delta_ci95"]
        steps_ci = comparison["steps_delta_ci95"]
        lines.append(f"| {name} | {comparison['pairs']} | {comparison['success_delta']:.3f} | [{success_ci[0]:.3f}, {success_ci[1]:.3f}] | {comparison['steps_delta']:.3f} | [{steps_ci[0]:.3f}, {steps_ci[1]:.3f}] |")
    summary = "\n".join(lines) + "\n"
    (root / "summary.md").write_text(summary)
    (latest / "summary.md").write_text(summary)
    progress(f"Report: {latest / 'summary.md'}")
    return report
