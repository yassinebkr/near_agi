from __future__ import annotations

import json
import os
import time
from typing import Any

from .models import AgentState, CandidateAction, TransitionPrediction


PROPERTIES = ("success", "goal_progress", "information_gain", "risk", "reversible", "needs_more_observation")


def compact_state(state: AgentState, action: CandidateAction) -> dict[str, Any]:
    return {
        "goal": state.goal, "page": {"title": state.page_title, "path": state.path, "text": state.page_text[:1500]},
        "known_facts": [b.model_dump() for b in state.beliefs], "unknowns": state.unknowns,
        "recent_history": [h.model_dump() for h in state.history[-3:]], "candidate_action": action.model_dump(),
    }


def resolve_laya_device(requested: str | None = None, torch_module: Any | None = None) -> str:
    policy = (requested or os.getenv("LAYA_DEVICE", "auto")).strip().lower()
    if policy not in {"auto", "cuda", "cpu"}:
        raise ValueError("LAYA_DEVICE must be one of: auto, cuda, cpu")
    if policy == "cpu":
        return "cpu"
    if torch_module is None:
        import torch as torch_module
    if torch_module.cuda.is_available():
        return "cuda"
    if policy == "cuda":
        torch_version = getattr(torch_module, "__version__", "unknown")
        cuda_version = getattr(getattr(torch_module, "version", None), "cuda", None)
        raise RuntimeError(f"LAYA_DEVICE=cuda but CUDA is unavailable (torch={torch_version}, torch_cuda={cuda_version}). Run `lda doctor --require-cuda`.")
    return "cpu"


class LayaPredictor:
    """Thin adapter over the current Laya typed-decision API."""

    def __init__(self, checkpoint: str = "laya", *, device: str | None = None, **load_kwargs: Any) -> None:
        import laya
        self.device = resolve_laya_device(device)
        started = time.perf_counter()
        self.agent = laya.load(checkpoint, device=self.device, **load_kwargs)
        self.load_time_ms = (time.perf_counter() - started) * 1000
        self.checkpoint = checkpoint

    @staticmethod
    def questions() -> dict[str, dict[str, Any]]:
        binary = lambda text: {"type": "noul", "instructions": text, "criteria": {"true": "evidence supports yes", "false": "evidence supports no"}, "labels": {"true": "A", "false": "B"}}
        return {
            "success": binary("Will this action probably complete the goal correctly?"),
            "goal_progress": binary("Will this action probably advance the current goal?"),
            "information_gain": {"type": "score", "instructions": "How much goal-relevant uncertainty will this action resolve?", "criteria": ["none", "very little", "some", "substantial", "decisive"]},
            "risk": binary("Can this action create an unsafe or hard-to-reverse side effect?"),
            "reversible": binary("Can the effect of this action be readily reversed?"),
            "needs_more_observation": binary("After this action, will more observation probably be needed before answering?"),
        }

    def predict(self, state: AgentState, actions: list[CandidateAction]) -> dict[str, TransitionPrediction]:
        output = {}
        for action in actions:
            started = time.perf_counter()
            raw = self.agent.predict(compact_state(state, action), self.questions())
            latency = (time.perf_counter() - started) * 1000
            ans = raw["answers"]
            def noul(key: str) -> float: return float(ans[key]["noul"])
            output[action.action_id] = TransitionPrediction(
                success=noul("success"), goal_progress=noul("goal_progress"),
                information_gain=float(ans["information_gain"]["score"]) / 4.0,
                risk=noul("risk"), reversible=noul("reversible"), needs_more_observation=noul("needs_more_observation"),
                provider=f"laya:{self.checkpoint}", calibrated=False, raw=json.loads(json.dumps(raw, default=str)), latency_ms=latency,
            )
        return output


class HeuristicPredictor:
    """Transparent non-Laya control. Never report this as base Laya."""

    def predict(self, state: AgentState, actions: list[CandidateAction]) -> dict[str, TransitionPrediction]:
        from .sandbox import TASKS
        out = {}
        for a in actions:
            primary = a.args.get("path") == TASKS[state.task_id]["primary_path"]
            danger = a.args.get("path") == "/danger"
            answer = a.tool == "answer"
            out[a.action_id] = TransitionPrediction(
                success=.95 if answer and not state.unknowns else (.3 if primary else .05),
                goal_progress=.9 if primary or (answer and not state.unknowns) else .1,
                information_gain=.95 if primary else (.3 if a.tool == "navigate" else 0),
                risk=.95 if danger else .02, reversible=.05 if danger else .95,
                needs_more_observation=.05 if primary else (.1 if answer else .8), provider="heuristic-control", calibrated=False,
            )
        return out

