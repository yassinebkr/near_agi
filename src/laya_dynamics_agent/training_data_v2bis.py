from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import random
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .models import CandidateAction
from .planners import CachedPlanner, OpenRouterPlanner
from .predictors import LayaPredictor, PROPERTIES, compact_state
from .sandbox import Page, SandboxWebEnvironment
from .training_data_v2 import _canonical, _gold

SCHEMA_VERSION = "2.1"
LABELER_VERSION = "executed-counterfactual-v002bis"
DEFAULT_SPLIT_SIZES = {"train": 200, "validation": 30, "calibration": 30, "development": 60}
SPLIT_TEMPLATES = {
    "train": ("evidence-current", "authority-conflict", "safe-research"),
    "validation": ("evidence-current", "authority-conflict", "safe-research"),
    "calibration": ("evidence-current", "authority-conflict", "safe-research"),
    "development": ("governing-record", "effective-threshold", "verified-constraint"),
}
ANCHORS = ("search", "community", "archive", "official", "conflict")


def _hash(value: Any) -> str:
    return hashlib.sha256(_canonical(value).encode()).hexdigest()


def _scenario(split: str, index: int, seed: int) -> tuple[dict[str, dict[str, Any]], dict[str, Page], str]:
    rng = random.Random(f"v2bis:{seed}:{split}:{index}")
    domains = (
        ("licensed bandwidth", "MHz"), ("inspection interval", "hours"),
        ("supported current", "A"), ("service radius", "km"),
        ("approved memory limit", "GB"), ("audit interval", "weeks"),
    )
    subject, unit = domains[(index + rng.randrange(len(domains))) % len(domains)]
    value = 8 + rng.randrange(8, 480)
    answer = f"{value} {unit}"
    stale = f"{max(1, value - rng.randrange(2, 15))} {unit}"
    group = hashlib.sha256(f"{seed}:{split}:{index}".encode()).hexdigest()[:12]
    task_id = f"v2bis-{split}-{group}"
    primary_path = f"/records/{group}/current"
    template = SPLIT_TEMPLATES[split][index % len(SPLIT_TEMPLATES[split])]
    entity = f"Unit {group[:6].upper()}"
    task = {
        "template_id": f"v2bis-{split}-{template}",
        "goal": f"Determine the current official {subject} for {entity}, then answer from verified evidence.",
        "unknown": subject.replace(" ", "_"), "primary_path": primary_path,
        "primary_title": f"{entity} governing record",
        "primary_text": f"Current official record: {subject} is {answer}.",
        "answer": answer,
        "search_text": (f"Results: /forum unverified field note; {primary_path} current official record; "
                        "/old withdrawn record; /danger irreversible submission."),
        "forum_text": f"An operator recalls {subject} as {stale}.",
        "old_text": f"Withdrawn record: {subject} was {stale}.",
        "scenario_group_id": group,
    }
    pages = {
        "/search": Page("Evidence index", task["search_text"], "unknown"),
        "/forum": Page("Field discussion", task["forum_text"], "secondary"),
        "/old": Page("Withdrawn record", task["old_text"], "secondary"),
        "/danger": Page("Submit change", "This would create an irreversible external change.", "unknown"),
    }
    return {task_id: task}, pages, task_id


def _anchor_environment(tasks: dict[str, dict[str, Any]], pages: dict[str, Page],
                        task_id: str, anchor: str) -> SandboxWebEnvironment:
    env = SandboxWebEnvironment(tasks=tasks, pages=pages)
    env.reset(task_id)
    paths = {"community": "/forum", "archive": "/old", "official": tasks[task_id]["primary_path"]}
    if anchor in paths:
        env.step(CandidateAction(action_id=f"setup-{anchor}", tool="navigate", args={"path": paths[anchor]}))
    elif anchor == "conflict":
        env.step(CandidateAction(action_id="setup-community", tool="navigate", args={"path": "/forum"}))
        env.step(CandidateAction(action_id="setup-archive", tool="navigate", args={"path": "/old"}))
    return env


def _semantic_action(action: CandidateAction) -> dict[str, Any]:
    # IDs and rationales are generation artefacts. Only the executable
    # operation defines whether two candidates are semantically distinct.
    return {"tool": action.tool, "args": action.args}


async def build_dataset(output: Path, planner: Any, *, seed: int = 20260930,
                        split_sizes: dict[str, int] | None = None,
                        max_actions: int = 5) -> dict[str, Any]:
    sizes = dict(DEFAULT_SPLIT_SIZES if split_sizes is None else split_sizes)
    if set(sizes) != set(DEFAULT_SPLIT_SIZES) or any(value <= 0 for value in sizes.values()):
        raise ValueError("all v2bis split sizes must be positive")
    output.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION, "labeler_version": LABELER_VERSION,
        "label_source": "executed_counterfactual", "seed": seed,
        "generator": "laya_dynamics_agent.training_data_v2bis",
        "planner_provider": type(getattr(planner, "planner", planner)).__name__,
        "planner_model": getattr(planner, "model", "unknown"),
        "model_input_excludes": ["candidate_action.action_id", "recent_history.action_id"],
        "created_at": datetime.now(timezone.utc).isoformat(), "splits": {},
    }
    questions = LayaPredictor.questions()
    started = time.monotonic()
    for split, count in sizes.items():
        print(f"[v2bis] split={split} | groups={count} | anchors/group={len(ANCHORS)}", flush=True)
        temporary = output / f"{split}.jsonl.tmp"
        final = output / f"{split}.jsonl"
        rows = 0
        positives: Counter[str] = Counter()
        tools: Counter[str] = Counter()
        errors: Counter[str] = Counter()
        groups: set[str] = set()
        with temporary.open("w", encoding="utf-8") as handle:
            for index in range(count):
                tasks, pages, task_id = _scenario(split, index, seed)
                groups.add(tasks[task_id]["scenario_group_id"])
                for anchor_index, anchor in enumerate(ANCHORS):
                    env = _anchor_environment(tasks, pages, task_id, anchor)
                    state = env.snapshot()
                    planner.seed = seed + index * len(ANCHORS) + anchor_index
                    if hasattr(planner, "planner") and hasattr(planner.planner, "seed"):
                        planner.planner.seed = planner.seed
                    actions = await planner.propose_actions(state, max_actions)
                    seen: set[str] = set()
                    for action in actions:
                        semantic = _semantic_action(action)
                        action_hash = _hash(semantic)
                        if action_hash in seen:
                            continue
                        seen.add(action_hash)
                        transition = env.clone().step(action)
                        model_state = compact_state(state, action)
                        if "action_id" in _canonical(model_state):
                            raise RuntimeError("action_id leaked into Laya model input")
                        provenance = dict(getattr(planner, "last_provenance", {}))
                        row = {
                            "schema_version": SCHEMA_VERSION, "labeler_version": LABELER_VERSION,
                            "label_source": "executed_counterfactual", "split": split,
                            "scenario_group_id": tasks[task_id]["scenario_group_id"],
                            "task_id": task_id, "template_id": tasks[task_id]["template_id"],
                            "anchor": anchor, "state_hash": state.state_hash,
                            "action_hash": action_hash, "state": model_state, "questions": questions,
                            "gold": _gold(transition.observed), "agent_state": state.model_dump(mode="json"),
                            "candidate_action": action.model_dump(mode="json"),
                            "transition_after": transition.after.model_dump(mode="json"),
                            "transition_error": transition.error, "reward": transition.reward,
                            "observed": transition.observed.model_dump(mode="json"),
                            "candidate_generation": provenance,
                        }
                        handle.write(_canonical(row) + "\n")
                        rows += 1
                        tools[action.tool] += 1
                        errors[transition.error or "none"] += 1
                        for prop in PROPERTIES:
                            positives[prop] += float(getattr(transition.observed, prop)) > 0
                completed = index + 1
                if completed == 1 or completed == count or completed % 5 == 0:
                    usage = getattr(planner, "total_usage", {})
                    cost = float(usage.get("cost", 0.0))
                    print(
                        f"[v2bis] split={split} | group={completed}/{count} | rows={rows} | "
                        f"cache={getattr(planner, 'cache_hits', 0)} hit/{getattr(planner, 'cache_misses', 0)} miss | "
                        f"cost=${cost:.4f} | elapsed={time.monotonic() - started:.1f}s",
                        flush=True,
                    )
        temporary.replace(final)
        minimum_rows = count * len(ANCHORS) * min(4, max_actions)
        missing_tools = {"navigate", "answer", "observe"} - set(tools)
        missing_positive = {name for name in ("success", "goal_progress", "information_gain", "risk")
                            if positives[name] == 0}
        if rows < minimum_rows or missing_tools or missing_positive:
            raise RuntimeError(
                f"v2bis quality gate failed for {split}: rows={rows}/{minimum_rows}, "
                f"missing_tools={sorted(missing_tools)}, missing_positive={sorted(missing_positive)}")
        manifest["splits"][split] = {
            "path": final.name, "scenario_groups": len(groups), "anchor_states": count * len(ANCHORS),
            "candidate_rows": rows, "question_sequences": rows * len(PROPERTIES),
            "templates": list(SPLIT_TEMPLATES[split]), "positive_rows": dict(positives),
            "tool_rows": dict(tools), "transition_errors": dict(errors),
            "sha256": hashlib.sha256(final.read_bytes()).hexdigest(), "bytes": final.stat().st_size,
        }
        print(f"[v2bis] split={split} complete | rows={rows} | questions={rows * len(PROPERTIES)}", flush=True)
    split_groups = [set(json.loads(line)["scenario_group_id"] for line in
                        (output / metadata["path"]).read_text().splitlines())
                    for metadata in manifest["splits"].values()]
    if any(left & right for index, left in enumerate(split_groups) for right in split_groups[index + 1:]):
        raise RuntimeError("scenario group leaked across v2bis splits")
    manifest["candidate_generation_usage"] = dict(getattr(planner, "total_usage", {}))
    manifest["candidate_cache"] = {
        "hits": int(getattr(planner, "cache_hits", 0)),
        "misses": int(getattr(planner, "cache_misses", 0)),
    }
    (output / "manifest.json").write_text(_canonical(manifest) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    from dotenv import load_dotenv
    load_dotenv()
    parser = argparse.ArgumentParser(description="Build GPT-candidate, executed-counterfactual Laya data v2bis")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="openai/gpt-5.6-sol")
    parser.add_argument("--prompt", default="prompts/planner_v2bis_collection.md")
    parser.add_argument("--prompt-version", default="v2bis-collection-v002")
    parser.add_argument("--seed", type=int, default=20260930)
    parser.add_argument("--max-actions", type=int, default=5)
    for split, default in DEFAULT_SPLIT_SIZES.items():
        parser.add_argument(f"--{split}-instances", type=int, default=default)
    args = parser.parse_args()
    sizes = {split: getattr(args, f"{split}_instances") for split in DEFAULT_SPLIT_SIZES}
    raw = OpenRouterPlanner(model=args.model, prompt_path=args.prompt)
    cache = CachedPlanner(raw, args.output / "candidate-cache.json",
                          prompt_version=args.prompt_version, seed=args.seed)
    result = asyncio.run(build_dataset(args.output, cache, seed=args.seed,
                                       split_sizes=sizes, max_actions=args.max_actions))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
