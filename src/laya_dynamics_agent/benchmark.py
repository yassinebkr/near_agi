from __future__ import annotations

import hashlib
import json
import math
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


def _bootstrap_mean_ci(values: list[float], *, samples: int = 2000, seed: int = 0) -> list[float | None]:
    if not values:
        return [None, None]
    rng = random.Random(seed)
    size = len(values)
    estimates = sorted(mean(values[rng.randrange(size)] for _ in range(size)) for _ in range(samples))
    return [estimates[int(samples * 0.025)], estimates[int(samples * 0.975)]]


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def _numbers(rows: list[dict[str, Any]], field: str) -> list[float]:
    return [float(row[field]) for row in rows
            if isinstance(row.get(field), (int, float)) and math.isfinite(float(row[field]))]


def _ratio_distribution(ratios: list[float]) -> dict[str, int]:
    counts = {"<=0.25x": 0, "0.25-0.5x": 0, "0.5-0.8x": 0, "0.8-1.25x": 0,
              "1.25-2x": 0, "2-3x": 0, ">3x": 0}
    for value in ratios:
        if value <= .25: counts["<=0.25x"] += 1
        elif value <= .5: counts["0.25-0.5x"] += 1
        elif value <= .8: counts["0.5-0.8x"] += 1
        elif value <= 1.25: counts["0.8-1.25x"] += 1
        elif value <= 2: counts["1.25-2x"] += 1
        elif value <= 3: counts["2-3x"] += 1
        else: counts[">3x"] += 1
    return counts


def _metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    successes = [float(row["success"]) for row in rows]
    steps = [float(row["steps"]) for row in rows]
    wall = _numbers(rows, "wall_clock_ms")
    effective = _numbers([row for row in rows if row.get("effective_end_to_end_complete")],
                         "effective_end_to_end_ms_reconstructed")
    result = {
        "episodes": len(rows), "success_rate": mean(successes), "success_std": pstdev(successes),
        "success_ci95": _bootstrap_mean_ci(successes), "mean_steps": mean(steps),
        "median_steps": median(steps), "steps_std": pstdev(steps), "steps_ci95": _bootstrap_mean_ci(steps),
        "unsafe_actions": sum(row["unsafe_actions"] for row in rows),
        "unnecessary_actions": sum(row["unnecessary_actions"] for row in rows),
        "planner_latency_ms": sum(_numbers(rows, "planner_wall_ms")),
        "predictor_latency_ms": sum(_numbers(rows, "predictor_reported_ms")),
        "mean_wall_clock_ms": mean(wall) if wall else None,
        "median_wall_clock_ms": median(wall) if wall else None,
        "p50_wall_clock_ms": _percentile(wall, .50),
        "p90_wall_clock_ms": _percentile(wall, .90),
        "p95_wall_clock_ms": _percentile(wall, .95),
        "wall_clock_ci95": _bootstrap_mean_ci(wall),
        "candidate_fresh_count": sum(int(row.get("candidate_fresh_count", 0)) for row in rows),
        "candidate_cache_hit_count": sum(int(row.get("candidate_cache_hit_count", 0)) for row in rows),
        "effective_end_to_end_complete_episodes": len(effective),
        "mean_effective_end_to_end_ms_reconstructed": mean(effective) if effective else None,
        "usage": _sum_usage(rows),
    }
    for field in ("planner_wall_ms", "predictor_wall_ms", "predictor_reported_ms",
                  "candidate_generation_ms", "candidate_cache_lookup_ms", "policy_ms",
                  "selector_wall_ms", "environment_ms", "framework_overhead_ms"):
        values = _numbers(rows, field)
        result[f"mean_{field}"] = mean(values) if values else None
    return result


def _effective_latency(row: dict[str, Any]) -> float | None:
    if row.get("mode") == "direct_gpt":
        value = row.get("wall_clock_ms")
    elif row.get("effective_end_to_end_complete"):
        value = row.get("effective_end_to_end_ms_reconstructed")
    else:
        return None
    if not isinstance(value, (int, float)) or not math.isfinite(float(value)) or value <= 0:
        return None
    return float(value)


def _paired_comparisons(results: list[dict[str, Any]]) -> dict[str, Any]:
    indexed = {(row["task_id"], row["seed"], row["mode"]): row for row in results}
    comparisons: dict[str, Any] = {}
    pairs = (("candidates_heuristic", "direct_gpt"), ("candidates_base_laya", "direct_gpt"),
             ("candidates_base_laya", "candidates_heuristic"))
    for left, right in pairs:
        keys = sorted((task_id, seed) for task_id, seed, mode in indexed
                      if mode == left and (task_id, seed, right) in indexed)
        rows = [(indexed[task_id, seed, left], indexed[task_id, seed, right]) for task_id, seed in keys]
        success_delta = [float(a["success"]) - float(b["success"]) for a, b in rows]
        steps_delta = [float(a["steps"]) - float(b["steps"]) for a, b in rows]
        terminal_wall_deltas = [float(a["wall_clock_ms"]) - float(b["wall_clock_ms"]) for a, b in rows
                                if isinstance(a.get("wall_clock_ms"), (int, float))
                                and isinstance(b.get("wall_clock_ms"), (int, float))]
        successful = [(a, b) for a, b in rows if a.get("success") and b.get("success")]
        valid_success = [(a, b) for a, b in successful
                         if isinstance(a.get("wall_clock_ms"), (int, float))
                         and isinstance(b.get("wall_clock_ms"), (int, float))
                         and float(a["wall_clock_ms"]) > 0 and float(b["wall_clock_ms"]) > 0]
        wall_deltas = [float(a["wall_clock_ms"]) - float(b["wall_clock_ms"]) for a, b in valid_success]
        ratios = [float(a["wall_clock_ms"]) / float(b["wall_clock_ms"]) for a, b in valid_success]
        effective_pairs = [(_effective_latency(a), _effective_latency(b)) for a, b in successful]
        effective_ratios = [a / b for a, b in effective_pairs if a is not None and b is not None and b > 0]
        comparisons[f"{left}_vs_{right}"] = {
            "pairs": len(keys), "pairs_both_success": len(successful),
            "pairs_valid_time_to_success": len(valid_success),
            "pairs_excluded_time_to_success": len(keys) - len(valid_success),
            "success_delta": mean(success_delta) if success_delta else 0.0,
            "success_delta_ci95": _bootstrap_mean_ci(success_delta),
            "steps_delta": mean(steps_delta) if steps_delta else 0.0,
            "steps_delta_ci95": _bootstrap_mean_ci(steps_delta),
            "wall_clock_delta_ms": mean(wall_deltas) if wall_deltas else None,
            "wall_clock_delta_ci95": _bootstrap_mean_ci(wall_deltas),
            "mean_latency_ratio": mean(ratios) if ratios else None,
            "median_latency_ratio": median(ratios) if ratios else None,
            "latency_ratio_ci95": _bootstrap_mean_ci(ratios),
            "latency_ratio_distribution": _ratio_distribution(ratios),
            "all_terminal_outcomes": {
                "pairs": len(terminal_wall_deltas),
                "mean_wall_clock_delta_ms": mean(terminal_wall_deltas) if terminal_wall_deltas else None,
                "wall_clock_delta_ci95": _bootstrap_mean_ci(terminal_wall_deltas),
            },
            "effective_end_to_end": {
                "pairs_complete": len(effective_ratios),
                "pairs_incomplete": len(successful) - len(effective_ratios),
                "mean_latency_ratio": mean(effective_ratios) if effective_ratios else None,
                "median_latency_ratio": median(effective_ratios) if effective_ratios else None,
                "latency_ratio_ci95": _bootstrap_mean_ci(effective_ratios),
                "latency_ratio_distribution": _ratio_distribution(effective_ratios),
            },
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
                        progress(f"    {marker} | steps={result['steps']} | stop={result['stop_reason']} | elapsed={result['wall_clock_ms'] / 1000:.1f}s")
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
    lines = ["# Benchmark", "", f"Campaign: `{campaign_id}` · suite: `{suite}` · provider: `{provider}` · model: `{resolved_model}` · prompts: `{DIRECT_PROMPT_VERSION}` / `{PROMPT_VERSION}`", "", "Observed wall time is the measured runtime. Reconstructed end-to-end time adds original GPT candidate-generation latency only for cache replays with complete provenance.", "", "| mode | episodes | success | mean steps | wall mean | wall p50 | wall p90 | wall p95 | fresh | replay |", "|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|"]
    for mode, metrics in report["aggregate"]["modes"].items():
        values = [metrics[key] for key in ("mean_wall_clock_ms", "p50_wall_clock_ms", "p90_wall_clock_ms", "p95_wall_clock_ms")]
        formatted = [(f"{value / 1000:.3f}s" if value is not None else "n/a") for value in values]
        lines.append(f"| {mode} | {metrics['episodes']} | {metrics['success_rate']:.3f} | {metrics['mean_steps']:.2f} | {formatted[0]} | {formatted[1]} | {formatted[2]} | {formatted[3]} | {metrics['candidate_fresh_count']} | {metrics['candidate_cache_hit_count']} |")
    lines.extend(["", "## Component latency", "", "Candidate generation and cache lookup are subdivisions of planner wall time and must not be added to it.", "", "| mode | planner mean | generation mean | cache lookup mean | predictor wall mean | predictor reported mean | policy mean | selector mean | environment mean | overhead mean |", "|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|"])
    component_keys = ("mean_planner_wall_ms", "mean_candidate_generation_ms", "mean_candidate_cache_lookup_ms", "mean_predictor_wall_ms", "mean_predictor_reported_ms", "mean_policy_ms", "mean_selector_wall_ms", "mean_environment_ms", "mean_framework_overhead_ms")
    for mode, metrics in report["aggregate"]["modes"].items():
        components = [(f"{metrics[key]:.2f}ms" if metrics[key] is not None else "n/a") for key in component_keys]
        lines.append(f"| {mode} | " + " | ".join(components) + " |")
    lines.extend(["", "## Per-template results"])
    for template, modes in report["aggregate"]["templates"].items():
        lines.extend(["", f"### `{template}`", "", "| mode | episodes | success | mean steps | mean wall |", "|:--|--:|--:|--:|--:|"])
        for mode, metrics in modes.items():
            wall = f"{metrics['mean_wall_clock_ms'] / 1000:.3f}s" if metrics["mean_wall_clock_ms"] is not None else "n/a"
            lines.append(f"| {mode} | {metrics['episodes']} | {metrics['success_rate']:.3f} | {metrics['mean_steps']:.2f} | {wall} |")
    lines.extend(["", "## Paired comparisons", "", "Time-to-success contains only pairs where both arms succeed and both durations are valid. Ratio is left/right; below 1 means the left arm is faster. All-terminal timing is reported separately.", "", "| comparison | pairs | both succeed | wall delta | median ratio | ratio 95% CI | effective pairs | effective median ratio |", "|:--|--:|--:|--:|--:|:--|--:|--:|"])
    for name, comparison in report["aggregate"]["paired_comparisons"].items():
        delta = comparison["wall_clock_delta_ms"]
        ratio = comparison["median_latency_ratio"]
        ratio_ci = comparison["latency_ratio_ci95"]
        effective = comparison["effective_end_to_end"]
        effective_ratio = effective["median_latency_ratio"]
        delta_text = f"{delta:.2f}ms" if delta is not None else "n/a"
        ratio_text = f"{ratio:.3f}" if ratio is not None else "n/a"
        ratio_ci_text = f"[{ratio_ci[0]:.3f}, {ratio_ci[1]:.3f}]" if ratio_ci[0] is not None else "n/a"
        effective_ratio_text = f"{effective_ratio:.3f}" if effective_ratio is not None else "n/a"
        lines.append(f"| {name} | {comparison['pairs']} | {comparison['pairs_both_success']} | "
                     f"{delta_text} | {ratio_text} | {ratio_ci_text} | "
                     f"{effective['pairs_complete']} | {effective_ratio_text} |")
        if name == "candidates_base_laya_vs_direct_gpt":
            lines.extend(["", "### Laya versus direct latency-ratio distribution", "", "Observed warm/cold mix: `" + json.dumps(comparison["latency_ratio_distribution"], sort_keys=True) + "`", "", "Reconstructed end-to-end (complete provenance only): `" + json.dumps(effective["latency_ratio_distribution"], sort_keys=True) + "`"])
    summary = "\n".join(lines) + "\n"
    (root / "summary.md").write_text(summary)
    (latest / "summary.md").write_text(summary)
    progress(f"Report: {latest / 'summary.md'}")
    return report
