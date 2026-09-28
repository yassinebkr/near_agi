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

from .benchmark import run_benchmark_suite
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
    seeds = tuple(int(value.strip()) for value in args.seeds.split(",") if value.strip())
    return await run_benchmark_suite(include_laya=args.with_laya, provider=args.provider, model=args.model, shutdown=shutdown, seeds=seeds)


def add_planner_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--provider", choices=["deterministic", "openai", "openrouter"], default="deterministic")
    parser.add_argument("--model", default=None, help="Provider-specific model id; defaults come from the environment")


def _nested_provider_message(value: Any) -> str | None:
    if isinstance(value, str):
        try:
            return _nested_provider_message(json.loads(value))
        except (json.JSONDecodeError, TypeError):
            return None
    if isinstance(value, dict):
        raw = value.get("raw")
        nested = _nested_provider_message(raw) if raw else None
        if nested:
            return nested
        for child in value.values():
            nested = _nested_provider_message(child)
            if nested:
                return nested
        message = value.get("message")
        if isinstance(message, str) and message != "Provider returned error":
            return message
    if isinstance(value, list):
        for child in value:
            nested = _nested_provider_message(child)
            if nested:
                return nested
    return None


def concise_error(exc: Exception) -> str:
    message = _nested_provider_message(getattr(exc, "body", None))
    if not message:
        message = str(exc).splitlines()[0]
    return f"{type(exc).__name__}: {message}"


def print_benchmark_summary(report: dict[str, Any]) -> None:
    print("\nBenchmark complete")
    print(f"Campaign: {report['campaign_id']}")
    print(f"Provider: {report['provider']} | model: {report['model']}")
    print("Mode                         Runs  Success  Steps  Unsafe  Predictor")
    print("---------------------------  ----  -------  -----  ------  ---------")
    for mode, metrics in report["aggregate"]["modes"].items():
        predictor_seconds = metrics["predictor_latency_ms"] / 1000
        print(f"{mode:<27}  {metrics['episodes']:>4}  {metrics['success_rate']:>7.1%}  {metrics['mean_steps']:>5.2f}  {metrics['unsafe_actions']:>6}  {predictor_seconds:>7.1f}s")
    print(f"Cache: {report['cache_hits']} hits / {report['cache_misses']} misses")
    print("Full report: reports/latest/summary.md")
    print("Raw metrics: reports/latest/metrics.json")


def main() -> None:
    parser = argparse.ArgumentParser(prog="lda")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("doctor")
    demo = sub.add_parser("demo")
    demo.add_argument("--debug", action="store_true", help="Show full tracebacks")
    add_planner_arguments(demo)
    bench = sub.add_parser("benchmark")
    bench.add_argument("--suite", default="smoke", choices=["smoke"])
    bench.add_argument("--with-laya", action="store_true")
    bench.add_argument("--seeds", default="0", help="Comma-separated deterministic seeds, e.g. 0,1,2")
    bench.add_argument("--json", action="store_true", help="Print the complete JSON result to the terminal")
    bench.add_argument("--debug", action="store_true", help="Show full tracebacks")
    add_planner_arguments(bench)
    args = parser.parse_args()
    if args.command == "doctor":
        raise SystemExit(doctor())
    try:
        result = asyncio.run(dispatch(args))
    except KeyboardInterrupt:
        raise SystemExit(130)
    except Exception as exc:
        if getattr(args, "debug", False):
            raise
        print(f"\nERROR: {concise_error(exc)}", file=sys.stderr)
        print("The failed run was finalized in SQLite/JSONL. Re-run with --debug for the full traceback.", file=sys.stderr)
        raise SystemExit(1)
    if args.command == "benchmark" and not args.json:
        print_benchmark_summary(result)
    else:
        print(json.dumps(result, indent=2))
