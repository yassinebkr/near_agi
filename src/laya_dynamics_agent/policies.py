from __future__ import annotations

import random
from dataclasses import dataclass

from .models import AgentState, CandidateAction, TransitionPrediction


@dataclass(frozen=True)
class UtilityWeights:
    success: float = 1.0
    goal_progress: float = 0.8
    information_gain: float = 0.5
    risk: float = 1.2
    reversible: float = 0.2
    needs_more_observation: float = 0.1


class GreedyUtilityPolicy:
    def __init__(self, weights: UtilityWeights | None = None) -> None: self.weights = weights or UtilityWeights()
    def utility(self, p: TransitionPrediction) -> float:
        w = self.weights
        return w.success*p.success + w.goal_progress*p.goal_progress + w.information_gain*p.information_gain - w.risk*p.risk + w.reversible*p.reversible - w.needs_more_observation*p.needs_more_observation
    def select(self, state: AgentState, actions: list[CandidateAction], predictions: dict[str, TransitionPrediction]) -> CandidateAction:
        return max(actions, key=lambda a: (self.utility(predictions[a.action_id]), -actions.index(a)))


class GPTOnlyPolicy:
    """Legacy ordered-candidate control; not a direct GPT baseline."""
    def select(self, state: AgentState, actions: list[CandidateAction], predictions: dict[str, TransitionPrediction]) -> CandidateAction:
        return actions[0]


class RandomPolicy:
    def __init__(self, seed: int = 0) -> None: self.rng = random.Random(seed)
    def select(self, state: AgentState, actions: list[CandidateAction], predictions: dict[str, TransitionPrediction]) -> CandidateAction: return self.rng.choice(actions)


class OraclePolicy:
    def select(self, state: AgentState, actions: list[CandidateAction], predictions: dict[str, TransitionPrediction]) -> CandidateAction:
        desired = "answer-primary" if not state.unknowns else "primary"
        return next((a for a in actions if a.action_id == desired), actions[0])

