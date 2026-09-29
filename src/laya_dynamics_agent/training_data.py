"""Historical v1 synthetic corpus builder, retained for reproducibility."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from .models import AgentState, Belief, CandidateAction, Observation, ObservedProperties
from .predictors import LayaPredictor, PROPERTIES, compact_state


DATASET_SCHEMA_VERSION = "1.0"
LABELER_VERSION = "synthetic-counterfactual-v001"
DEFAULT_SPLIT_SIZES = {"train": 300, "validation": 45, "calibration": 45, "development": 90}
TRAIN_TEMPLATES = ("train-authoritative-spec-v1", "train-current-record-v1", "train-security-release-v1")
DEVELOPMENT_TEMPLATES = ("dev-certified-capacity-v1", "dev-active-contract-v1")
RESERVED_TEMPLATE_PREFIXES = ("heldout-", "final-")


def _canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _hash(value: Any) -> str:
    return hashlib.sha256(_canonical(value).encode()).hexdigest()


def _score_distribution(value: float, levels: int = 5) -> list[float]:
    position = min(1.0, max(0.0, value)) * (levels - 1)
    low = int(position)
    high = min(levels - 1, low + 1)
    target = [0.0] * levels
    target[low] += high - position if high != low else 1.0
    if high != low:
        target[high] += position - low
    return target


def _gold(observed: ObservedProperties) -> dict[str, dict[str, dict[str, float]]]:
    result: dict[str, dict[str, dict[str, float]]] = {}
    for name in PROPERTIES:
        value = float(getattr(observed, name))
        if name == "information_gain":
            probabilities = {str(i): p for i, p in enumerate(_score_distribution(value))}
        else:
            probabilities = {"false": 1.0 - value, "true": value}
        result[name] = {"probabilities": probabilities}
    return result


def _observed(*, success: float = 0.0, progress: float = 0.0, information: float = 0.0,
              risk: float = 0.0, reversible: float = 1.0, needs_more: float = 1.0) -> ObservedProperties:
    return ObservedProperties(
        success=success, goal_progress=progress, information_gain=information,
        risk=risk, reversible=reversible, needs_more_observation=needs_more,
        provenance={name: "deterministic" for name in PROPERTIES},
    )


def _state(task: dict[str, str], phase: str) -> AgentState:
    common = dict(
        task_id=task["task_id"], template_id=task["template_id"], goal=task["goal"],
        available_tools=["answer", "navigate", "observe"], terminal=False,
    )
    if phase == "search":
        return AgentState(step_index=0, page_title="Search results", path="/search",
                          page_text=task["search_text"], unknowns=[task["unknown"]], **common)
    source_kind = "primary" if phase == "primary" else "secondary"
    path = "/official" if phase == "primary" else ("/community" if phase == "secondary" else "/archive")
    value = task["answer"] if phase == "primary" else task["distractor"]
    state = AgentState(
        step_index=1, page_title=f"{phase.title()} source", path=path,
        page_text=(task["primary_text"] if phase == "primary" else f"A non-authoritative source reports {value}."),
        observations=[Observation(source_id=path, title=f"{phase.title()} source", path=path,
                                  text=(task["primary_text"] if phase == "primary" else value), source_kind=source_kind)],
        unknowns=[] if phase == "primary" else [task["unknown"]], **common,
    )
    if phase == "primary":
        state.beliefs = [Belief(key=task["unknown"], value=task["answer"], confidence=1.0, evidence=[path])]
    return state


def _candidates(task: dict[str, str], phase: str) -> list[tuple[CandidateAction, ObservedProperties]]:
    official = CandidateAction(action_id="open_current_official_source", tool="navigate", args={"path": "/official"})
    danger = CandidateAction(action_id="perform_irreversible_action", tool="navigate", args={"path": "/danger"})
    observe = CandidateAction(action_id="observe_same_page_again", tool="observe", args={})
    if phase == "search":
        return [
            (official, _observed(progress=.7, information=1.0, needs_more=1.0)),
            (CandidateAction(action_id="open_community_claim", tool="navigate", args={"path": "/community"}), _observed()),
            (CandidateAction(action_id="open_archived_source", tool="navigate", args={"path": "/archive"}), _observed()),
            (observe, _observed()),
            (danger, _observed(risk=1.0, reversible=0.0)),
        ]
    if phase in {"secondary", "archived"}:
        return [
            (official, _observed(progress=.7, information=1.0, needs_more=1.0)),
            (CandidateAction(action_id="answer_from_unverified_source", tool="answer", args={"value": task["distractor"]}), _observed(needs_more=0.0)),
            (CandidateAction(action_id="reinspect_unverified_source", tool="observe", args={}), _observed()),
            (CandidateAction(action_id="switch_to_other_unverified_source", tool="navigate", args={"path": "/archive" if phase == "secondary" else "/community"}), _observed()),
            (danger, _observed(risk=1.0, reversible=0.0)),
        ]
    return [
        (CandidateAction(action_id="answer_with_verified_value", tool="answer", args={"value": task["answer"]}), _observed(success=1.0, progress=1.0, needs_more=0.0)),
        (CandidateAction(action_id="answer_with_stale_value", tool="answer", args={"value": task["distractor"]}), _observed(needs_more=0.0)),
        (CandidateAction(action_id="reopen_same_official_source", tool="navigate", args={"path": "/official"}), _observed(needs_more=0.0)),
        (observe, _observed(needs_more=0.0)),
        (danger, _observed(risk=1.0, reversible=0.0, needs_more=0.0)),
    ]


def _task(split: str, index: int, template_id: str, rng: random.Random) -> dict[str, str]:
    units = (("continuous pressure", "kPa"), ("retention period", "days"),
             ("safe firmware revision", "version"), ("rated capacity", "kg"), ("support window", "months"))
    subject, unit = units[(index + rng.randrange(len(units))) % len(units)]
    value = 20 + ((index * 17 + rng.randrange(13)) % 470)
    answer = f"{value}" if unit == "version" else f"{value} {unit}"
    distractor = f"{max(1, value - 7)}" if unit == "version" else f"{max(1, value - 7)} {unit}"
    entity = f"Unit-{split[:3]}-{index:04d}"
    return {
        "task_id": f"pt-{split}-{index:04d}", "template_id": template_id,
        "goal": f"Find the current official {subject} for {entity} and answer using verified evidence.",
        "unknown": subject.replace(" ", "_"), "answer": answer, "distractor": distractor,
        "search_text": f"Results include an official record for {entity}, a community recollection, an archived record, and an irreversible transaction.",
        "primary_text": f"Current official record for {entity}: {subject} is {answer}.",
    }


def iter_rows(split: str, count: int, seed: int) -> Iterable[dict[str, Any]]:
    templates = DEVELOPMENT_TEMPLATES if split == "development" else TRAIN_TEMPLATES
    rng = random.Random(f"{seed}:{split}")
    questions = LayaPredictor.questions()
    for index in range(count):
        task = _task(split, index, templates[index % len(templates)], rng)
        if task["task_id"].startswith("final-") or task["template_id"].startswith(RESERVED_TEMPLATE_PREFIXES):
            raise RuntimeError("reserved final benchmark identifier entered post-training data")
        for phase in ("search", "secondary", "archived", "primary"):
            state = _state(task, phase)
            for action, observed in _candidates(task, phase):
                action_payload = action.model_dump(mode="json")
                yield {
                    "schema_version": DATASET_SCHEMA_VERSION, "labeler_version": LABELER_VERSION,
                    "split": split, "task_id": task["task_id"], "template_id": task["template_id"],
                    "phase": phase, "state_hash": state.state_hash, "action_hash": _hash(action_payload),
                    "state": compact_state(state, action), "questions": questions, "gold": _gold(observed),
                    "agent_state": state.model_dump(mode="json"), "candidate_action": action_payload,
                    "observed": observed.model_dump(mode="json"), "provenance": "deterministic-synthetic-simulator",
                }


def build_dataset(output: Path, *, seed: int = 20260928,
                  split_sizes: dict[str, int] | None = None) -> dict[str, Any]:
    sizes = dict(DEFAULT_SPLIT_SIZES if split_sizes is None else split_sizes)
    if set(sizes) != set(DEFAULT_SPLIT_SIZES) or any(value <= 0 for value in sizes.values()):
        raise ValueError(f"split sizes must be positive for {sorted(DEFAULT_SPLIT_SIZES)}")
    output.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {
        "schema_version": DATASET_SCHEMA_VERSION, "labeler_version": LABELER_VERSION,
        "seed": seed, "generator": "laya_dynamics_agent.training_data", "splits": {},
        "reserved_final_prefixes": list(RESERVED_TEMPLATE_PREFIXES),
    }
    split_templates: dict[str, set[str]] = {}
    for split, count in sizes.items():
        path = output / f"{split}.jsonl"
        rows = 0
        templates: set[str] = set()
        positives = Counter()
        digest = hashlib.sha256()
        with path.open("w", encoding="utf-8") as handle:
            for row in iter_rows(split, count, seed):
                encoded = (_canonical(row) + "\n").encode()
                handle.write(encoded.decode())
                digest.update(encoded)
                rows += 1
                templates.add(row["template_id"])
                for prop in PROPERTIES:
                    positives[prop] += float(row["observed"][prop]) > 0
        split_templates[split] = templates
        manifest["splits"][split] = {
            "path": path.name, "instances": count, "candidate_rows": rows,
            "question_sequences": rows * len(PROPERTIES), "templates": sorted(templates),
            "positive_rows": dict(positives), "sha256": digest.hexdigest(), "bytes": path.stat().st_size,
        }
    if split_templates["train"] & split_templates["development"]:
        raise RuntimeError("development templates overlap training templates")
    (output / "manifest.json").write_text(_canonical(manifest) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Build deterministic counterfactual Laya post-training data")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260928)
    for split, default in DEFAULT_SPLIT_SIZES.items():
        parser.add_argument(f"--{split}-instances", type=int, default=default)
    args = parser.parse_args()
    sizes = {split: getattr(args, f"{split}_instances") for split in DEFAULT_SPLIT_SIZES}
    print(json.dumps(build_dataset(args.output, seed=args.seed, split_sizes=sizes), indent=2))


if __name__ == "__main__":
    main()
