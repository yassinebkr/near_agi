from __future__ import annotations

import argparse
import asyncio
import importlib.util
import json
import os
import platform
import shutil
import signal
import sqlite3
import subprocess
import sys
import uuid
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from .planners import DeterministicPlanner, OpenAIPlanner, OpenRouterPlanner
from .policies import GPTOnlyPolicy, GreedyUtilityPolicy, UtilityWeights
from .predictors import HeuristicPredictor, LayaPredictor
from .runner_impl import run_episode
from .sandbox import SandboxWebEnvironment
from .storage_v2 import TrajectoryStore


# Load project-local secrets without overriding explicitly exported variables.
load_dotenv(dotenv_path=Path.cwd() / ".env", override=False)


class GracefulShutdown:
    def __init__(self) -> None:
        self.event = asyncio.Event()
        self.signal_count = 0

    def request(self) -> None:
        self.signal_count += 1
        if self.signal_count == 1:
            print("\nCtrl+C received: finishing the current operation and saving the partial run...", file=sys.stderr)
            self.event.set()
        else:
            print("\nShutdown already requested; waiting for the current atomic operation.", file=sys.stderr)


def doctor() -> int:
    checks = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "sqlite": sqlite3.sqlite_version,
        "disk_free_gb": round(shutil.disk_usage(".").free / 2**30, 2),
        "openai_configured": bool(os.getenv("OPENAI_API_KEY")),
        "openrouter_configured": bool(os.getenv("OPENROUTER_API_KEY")),
        "openrouter_model": os.getenv("OPENROUTER_MODEL", "openai/gpt-5.6-sol"),
        "laya_import": importlib.util.find_spec("laya") is not None,
        "playwright_import": importlib.util.find_spec("playwright") is not None,
    }
    try:
        checks["nvidia_smi"] = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5,
        ).stdout.strip() or "unavailable"
    except (FileNotFoundError, subprocess.SubprocessError):
        checks["nvidia_smi"] = "unavailable"
    try:
        import torch
        checks.update({"torch": torch.__version__, "torch_cuda": torch.cuda.is_available(), "cuda_version": torch.version.cuda})
    except ImportError:
        checks["torch"] = "not installed"
    print(json.dumps(checks, indent=2))
    return 0


def make_planner(provider: str, model: str | None) -> Any:
    if provider == "deterministic":
        return DeterministicPlanner()
    if provider == "openai":
        return OpenAIPlanner(model=model)
    if provider == "openrouter":
        return OpenRouterPlanner(model=model)
    raise ValueError(f"unknown planner provider: {provider}")


async def execute(mode: str, *, planner_provider: str = "deterministic", model: str | None = None, report: bool = False, shutdown: GracefulShutdown | None = None) -> dict[str, Any]:
    run_id = f"{mode}-{uuid.uuid4().hex[:8]}"
    store = TrajectoryStore(Path("data/trajectories.sqlite3"), Path("logs/runs"))
    planner = make_planner(planner_provider, model)
    resolved_model = getattr(planner, "model", None)
    store.start_run(run_id, {"mode": mode, "planner_provider": planner_provider, "planner_model": resolved_model, "candidate_count": 5, "weights": UtilityWeights().__dict__})
    predictor = None
    policy = GPTOnlyPolicy()
    result: dict[str, Any]
    try:
        if mode == "heuristic":
            predictor, policy = HeuristicPredictor(), GreedyUtilityPolicy()
        elif mode == "base_laya":
            predictor, policy = LayaPredictor(os.getenv("LAYA_CHECKPOINT", "laya")), GreedyUtilityPolicy()
        result = await run_episode(
            run_id=run_id, task_id="voltage-001", environment=SandboxWebEnvironment(),
            planner=planner, predictor=predictor, policy=policy, store=store,
            stop_requested=(shutdown.event.is_set if shutdown else None),
        )
        result.update(run_id=run_id, mode=mode, planner_provider=planner_provider, planner_model=resolved_model)
        status = "interrupted" if result["stop_reason"] == "interrupted" else "completed"
        store.finish_run(run_id, status, result)
    except BaseException as exc:
        result = {"run_id": run_id, "mode": mode, "task_id": "voltage-001", "steps": 0, "stop_reason": "interrupted" if isinstance(exc, (KeyboardInterrupt, asyncio.CancelledError)) else "error", "error": type(exc).__name__}
        store.finish_run(run_id, "interrupted" if result["stop_reason"] == "interrupted" else "error", result)
        raise
    finally:
        store.close()
    if report:
        root = Path("reports") / run_id
        root.mkdir(parents=True, exist_ok=True)
        (root / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")
        (root / "summary.md").write_text("# Benchmark run\n\n```json\n" + json.dumps(result, indent=2) + "\n```\n")
    return result


async def benchmark(include_laya: bool, *, planner_provider: str, model: str | None, shutdown: GracefulShutdown) -> list[dict[str, Any]]:
    results = []
    for mode in ["gpt_only", "heuristic"] + (["base_laya"] if include_laya else []):
        if shutdown.event.is_set():
            break
        results.append(await execute(mode, planner_provider=planner_provider, model=model, report=True, shutdown=shutdown))
    root = Path("reports/latest")
    root.mkdir(parents=True, exist_ok=True)
    (root / "metrics.json").write_text(json.dumps(results, indent=2) + "\n")
    interrupted = shutdown.event.is_set()
    lines = ["# Benchmark", "", f"Planner provider: `{planner_provider}`. Interrupted: `{interrupted}`.", "", "| mode | success | steps | stop |", "|---|---:|---:|---|"] + [f"| {r['mode']} | {r['success']} | {r['steps']} | {r['stop_reason']} |" for r in results]
    (root / "summary.md").write_text("\n".join(lines) + "\n")
    return results


async def dispatch(args: argparse.Namespace) -> Any:
    shutdown = GracefulShutdown()
    loop = asyncio.get_running_loop()
    try:
        loop.add_signal_handler(signal.SIGINT, shutdown.request)
        loop.add_signal_handler(signal.SIGTERM, shutdown.request)
    except NotImplementedError:
        pass
    if args.command == "demo":
        return await execute("heuristic", planner_provider=args.provider, model=args.model, shutdown=shutdown)
    return await benchmark(args.with_laya, planner_provider=args.provider, model=args.model, shutdown=shutdown)


def add_planner_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--provider", choices=["deterministic", "openai", "openrouter"], default="deterministic")
    parser.add_argument("--model", default=None, help="Provider-specific model id; defaults come from the environment")


def main() -> None:
    parser = argparse.ArgumentParser(prog="lda")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("doctor")
    demo = sub.add_parser("demo")
    add_planner_arguments(demo)
    bench = sub.add_parser("benchmark")
    bench.add_argument("--suite", default="smoke", choices=["smoke"])
    bench.add_argument("--with-laya", action="store_true")
    add_planner_arguments(bench)
    args = parser.parse_args()
    if args.command == "doctor":
        raise SystemExit(doctor())
    try:
        result = asyncio.run(dispatch(args))
    except KeyboardInterrupt:
        raise SystemExit(130)
    print(json.dumps(result, indent=2))
