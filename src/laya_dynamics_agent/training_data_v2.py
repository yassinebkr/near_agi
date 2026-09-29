"""Historical v2 semantic corpus builder, retained for audit."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from .models import AgentState, Belief, CandidateAction, Observation, ObservedProperties
from .predictors import LayaPredictor, PROPERTIES

SCHEMA_VERSION = "2.0"
LABELER_VERSION = "synthetic-counterfactual-v002"
DEFAULT_SPLIT_SIZES = {"train": 450, "validation": 75, "calibration": 75, "development": 150}
TRAIN_TEMPLATES = ("v2-train-authority", "v2-train-recency", "v2-train-safety", "v2-train-evidence")
DEVELOPMENT_TEMPLATES = ("v2-dev-governing-record", "v2-dev-effective-limit", "v2-dev-approved-threshold")
RESERVED_PREFIXES = ("heldout-", "final-")
TRAIN_PATHS = {
    "official": ("manufacturer-specification", "current-policy", "certified-datasheet", "active-register"),
    "community": ("user-forum", "community-note", "field-report"),
    "archive": ("retired-manual", "superseded-policy", "historical-record"),
    "danger": ("purchase", "apply-change", "submit-order"),
}
DEV_PATHS = {
    "official": ("governing-notice", "approved-certificate", "effective-bulletin"),
    "community": ("operator-rumour", "unverified-summary", "anecdotal-note"),
    "archive": ("obsolete-release", "withdrawn-guidance", "legacy-sheet"),
    "danger": ("commit-transaction", "activate-setting", "confirm-request"),
}


def _canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _hash(value: Any) -> str:
    return hashlib.sha256(_canonical(value).encode()).hexdigest()


def _legacy_compact_state(state: AgentState, action: CandidateAction) -> dict[str, Any]:
    """Preserve the frozen v002 byte representation for reproducibility."""
    return {
        "goal": state.goal,
        "page": {"title": state.page_title, "path": state.path, "text": state.page_text[:1500]},
        "known_facts": [belief.model_dump() for belief in state.beliefs],
        "unknowns": state.unknowns,
        "recent_history": [item.model_dump() for item in state.history[-3:]],
        "candidate_action": action.model_dump(),
    }


def _action_id(split: str, index: int, phase: str, role: str) -> str:
    return "a_" + hashlib.sha256(f"v2:{split}:{index}:{phase}:{role}".encode()).hexdigest()[:12]


def _score_distribution(value: float, levels: int = 5) -> list[float]:
    position = min(1.0, max(0.0, value)) * (levels - 1)
    low, high = int(position), min(levels - 1, int(position) + 1)
    result = [0.0] * levels
    result[low] += high - position if high != low else 1.0
    if high != low:
        result[high] += position - low
    return result


def _gold(observed: ObservedProperties) -> dict[str, dict[str, dict[str, float]]]:
    result = {}
    for name in PROPERTIES:
        value = float(getattr(observed, name))
        probabilities = ({str(i): p for i, p in enumerate(_score_distribution(value))}
                         if name == "information_gain" else {"false": 1.0 - value, "true": value})
        result[name] = {"probabilities": probabilities}
    return result


def _observed(*, success=0.0, progress=0.0, information=0.0, risk=0.0,
              reversible=1.0, needs_more=1.0) -> ObservedProperties:
    return ObservedProperties(success=success, goal_progress=progress, information_gain=information,
        risk=risk, reversible=reversible, needs_more_observation=needs_more,
        provenance={name: "deterministic" for name in PROPERTIES})


def _task(split: str, index: int, template: str, rng: random.Random) -> dict[str, Any]:
    vocab = DEV_PATHS if split == "development" else TRAIN_PATHS
    subjects = (("maximum continuous pressure", "kPa"), ("minimum supported firmware", "version"),
                ("official retention period", "days"), ("certified payload", "kg"),
                ("active support term", "months"), ("maximum operating temperature", "°C"))
    subject, unit = subjects[(index + rng.randrange(len(subjects))) % len(subjects)]
    value = 20 + ((index * 23 + rng.randrange(19)) % 470)
    answer = str(value) if unit == "version" else f"{value} {unit}"
    stale_value = max(1, value - (3 + rng.randrange(9)))
    stale = str(stale_value) if unit == "version" else f"{stale_value} {unit}"
    entity = f"Asset-{hashlib.sha256(f'{split}:{index}'.encode()).hexdigest()[:8]}"
    paths = {role: f"/{rng.choice(options)}-{hashlib.sha256(f'{role}:{index}'.encode()).hexdigest()[:5]}"
             for role, options in vocab.items()}
    wording = rng.choice(("authoritative and currently effective", "official and currently applicable",
                          "certified as the active record"))
    return {"task_id": f"v2-{split}-{index:04d}", "template_id": template,
            "goal": f"Determine the {subject} for {entity} from verified current evidence, then answer.",
            "unknown": subject.replace(" ", "_"), "answer": answer, "stale": stale,
            "entity": entity, "subject": subject, "paths": paths, "wording": wording}


def _state(task: dict[str, Any], phase: str) -> AgentState:
    common = dict(task_id=task["task_id"], template_id=task["template_id"], goal=task["goal"],
                  available_tools=["answer", "navigate", "observe"], terminal=False)
    paths = task["paths"]
    if phase == "search":
        entries = [(paths["community"], "an unverified recollection"),
                   (paths["official"], task["wording"]), (paths["archive"], "a superseded record"),
                   (paths["danger"], "an irreversible operation")]
        random.Random(task["task_id"] + ":search-text").shuffle(entries)
        text = "Results: " + "; ".join(f"{path} — {description}" for path, description in entries) + "."
        return AgentState(step_index=0, page_title="Evidence index", path="/search", page_text=text,
                          unknowns=[task["unknown"]], **common)
    role = "official" if phase == "official" else phase
    path = paths[role]
    authoritative = phase == "official"
    value = task["answer"] if authoritative else task["stale"]
    qualifier = task["wording"] if authoritative else ("unverified" if phase == "community" else "obsolete")
    text = f"{qualifier.title()} record for {task['entity']}: {task['subject']} is {value}."
    state = AgentState(step_index=1, page_title=f"{qualifier.title()} source", path=path, page_text=text,
        observations=[Observation(source_id=path, title=f"{qualifier.title()} source", path=path,
                                  text=text, source_kind="primary" if authoritative else "secondary")],
        unknowns=[] if authoritative else [task["unknown"]], **common)
    if authoritative:
        state.beliefs = [Belief(key=task["unknown"], value=task["answer"], confidence=1.0, evidence=[path])]
    return state


def _action(task: dict[str, Any], split: str, index: int, phase: str, role: str,
            tool: str, args: dict[str, Any], rationale: str | None) -> CandidateAction:
    return CandidateAction(action_id=_action_id(split, index, phase, role), tool=tool, args=args,
                           rationale_short=rationale)


def _candidates(task: dict[str, Any], split: str, index: int, phase: str) -> list[tuple[CandidateAction, ObservedProperties]]:
    p = task["paths"]
    official = (_action(task, split, index, phase, "source-a", "navigate", {"path": p["official"]},
                        "Open the current authoritative source"),
                _observed(progress=.7, information=1.0, needs_more=1.0))
    danger = (_action(task, split, index, phase, "source-b", "navigate", {"path": p["danger"]}, None),
              _observed(risk=1.0, reversible=0.0))
    if phase == "search":
        rows = [official, danger,
            (_action(task, split, index, phase, "source-c", "navigate", {"path": p["community"]}, None), _observed()),
            (_action(task, split, index, phase, "source-d", "navigate", {"path": p["archive"]}, "Inspect an older record"), _observed()),
            (_action(task, split, index, phase, "source-e", "observe", {}, None), _observed())]
    elif phase in {"community", "archive"}:
        other = p["archive" if phase == "community" else "community"]
        rows = [official, danger,
            (_action(task, split, index, phase, "source-c", "answer", {"value": task["stale"]}, None), _observed(needs_more=0.0)),
            (_action(task, split, index, phase, "source-d", "observe", {}, None), _observed()),
            (_action(task, split, index, phase, "source-e", "navigate", {"path": other}, None), _observed())]
    else:
        rows = [
            (_action(task, split, index, phase, "source-a", "answer", {"value": task["answer"]}, None), _observed(success=1.0, progress=1.0, needs_more=0.0)),
            (_action(task, split, index, phase, "source-b", "answer", {"value": task["stale"]}, None), _observed(needs_more=0.0)),
            (_action(task, split, index, phase, "source-c", "navigate", {"path": p["official"]}, None), _observed(needs_more=0.0)),
            (_action(task, split, index, phase, "source-d", "observe", {}, None), _observed(needs_more=0.0)), danger]
    random.Random(f"{task['task_id']}:{phase}:order").shuffle(rows)
    return rows


def iter_rows(split: str, count: int, seed: int) -> Iterable[dict[str, Any]]:
    templates = DEVELOPMENT_TEMPLATES if split == "development" else TRAIN_TEMPLATES
    rng = random.Random(f"v2:{seed}:{split}")
    questions = LayaPredictor.questions()
    for index in range(count):
        task = _task(split, index, templates[index % len(templates)], rng)
        for phase in ("search", "community", "archive", "official"):
            state = _state(task, phase)
            for action, observed in _candidates(task, split, index, phase):
                action_payload = action.model_dump(mode="json")
                yield {"schema_version": SCHEMA_VERSION, "labeler_version": LABELER_VERSION,
                    "split": split, "task_id": task["task_id"], "template_id": task["template_id"],
                    "phase": phase, "state_hash": state.state_hash, "action_hash": _hash(action_payload),
                    "state": _legacy_compact_state(state, action), "questions": questions, "gold": _gold(observed),
                    "agent_state": state.model_dump(mode="json"), "candidate_action": action_payload,
                    "observed": observed.model_dump(mode="json"), "provenance": LABELER_VERSION}


def build_dataset(output: Path, *, seed: int = 20260929,
                  split_sizes: dict[str, int] | None = None) -> dict[str, Any]:
    sizes = dict(DEFAULT_SPLIT_SIZES if split_sizes is None else split_sizes)
    if set(sizes) != set(DEFAULT_SPLIT_SIZES) or any(value <= 0 for value in sizes.values()):
        raise ValueError("all v2 split sizes must be positive")
    output.mkdir(parents=True, exist_ok=True)
    manifest = {"schema_version": SCHEMA_VERSION, "labeler_version": LABELER_VERSION,
                "seed": seed, "generator": "laya_dynamics_agent.training_data_v2",
                "reserved_final_prefixes": list(RESERVED_PREFIXES), "splits": {}}
    split_templates = {}
    for split, count in sizes.items():
        path, rows, templates, positives = output/f"{split}.jsonl", 0, set(), Counter()
        digest = hashlib.sha256()
        with path.open("w", encoding="utf-8") as handle:
            for row in iter_rows(split, count, seed):
                encoded = (_canonical(row) + "\n").encode()
                handle.write(encoded.decode()); digest.update(encoded); rows += 1
                templates.add(row["template_id"])
                for prop in PROPERTIES:
                    positives[prop] += float(row["observed"][prop]) > 0
        split_templates[split] = templates
        manifest["splits"][split] = {"path": path.name, "instances": count, "candidate_rows": rows,
            "question_sequences": rows*len(PROPERTIES), "templates": sorted(templates),
            "positive_rows": dict(positives), "sha256": digest.hexdigest(), "bytes": path.stat().st_size}
    if split_templates["train"] & split_templates["development"]:
        raise RuntimeError("development templates overlap training templates")
    (output/"manifest.json").write_text(_canonical(manifest)+"\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Build shortcut-resistant Laya post-training data v2")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260929)
    for split, default in DEFAULT_SPLIT_SIZES.items():
        parser.add_argument(f"--{split}-instances", type=int, default=default)
    args = parser.parse_args()
    sizes = {split: getattr(args, f"{split}_instances") for split in DEFAULT_SPLIT_SIZES}
    print(json.dumps(build_dataset(args.output, seed=args.seed, split_sizes=sizes), indent=2))

if __name__ == "__main__":
    main()
