from __future__ import annotations

from typing import Protocol

from .models import AgentState, CandidateAction, Transition, TransitionPrediction


class Planner(Protocol):
    async def propose_actions(self, state: AgentState, max_actions: int) -> list[CandidateAction]: ...


class TransitionPredictor(Protocol):
    def predict(self, state: AgentState, actions: list[CandidateAction]) -> dict[str, TransitionPrediction]: ...


class Policy(Protocol):
    def select(self, state: AgentState, actions: list[CandidateAction], predictions: dict[str, TransitionPrediction]) -> CandidateAction: ...


class Environment(Protocol):
    def reset(self, task_id: str) -> AgentState: ...
    def step(self, action: CandidateAction) -> Transition: ...

