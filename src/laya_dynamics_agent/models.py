"""Strict, versioned data contracts shared by the agent pipeline."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class StrictModel(BaseModel):
    """Reject unknown fields so schema drift fails loudly."""
    model_config = ConfigDict(extra="forbid")


class Observation(StrictModel):
    source_id: str
    title: str
    path: str
    text: str
    source_kind: Literal["primary", "secondary", "unknown"] = "unknown"


class Belief(StrictModel):
    key: str
    value: str
    confidence: float = Field(ge=0, le=1)
    evidence: list[str] = Field(default_factory=list)


class ActionRecord(StrictModel):
    step: int
    action_id: str
    tool: str
    args: dict[str, Any]
    state_hash_after: str


class AgentState(StrictModel):
    """Canonical observable state used for planning, caching, and replay."""
    schema_version: Literal["1.0"] = "1.0"
    task_id: str
    template_id: str
    step_index: int = Field(ge=0)
    goal: str
    page_title: str
    path: str
    page_text: str
    observations: list[Observation] = Field(default_factory=list)
    beliefs: list[Belief] = Field(default_factory=list)
    unknowns: list[str] = Field(default_factory=list)
    history: list[ActionRecord] = Field(default_factory=list)
    available_tools: list[str] = Field(default_factory=list)
    terminal: bool = False
    success: bool | None = None
    side_effects: list[str] = Field(default_factory=list)

    def canonical_json(self) -> str:
        """Serialize deterministically; this exact representation is hashed."""
        return json.dumps(self.model_dump(mode="json"), sort_keys=True, separators=(",", ":"), ensure_ascii=False)

    @property
    def state_hash(self) -> str:
        """Content address used for loop detection and cache keys."""
        return hashlib.sha256(self.canonical_json().encode()).hexdigest()


class CandidateAction(StrictModel):
    """Typed action proposed by GPT or a deterministic fixture planner."""
    schema_version: Literal["1.0"] = "1.0"
    action_id: str
    tool: Literal["navigate", "answer", "observe"]
    args: dict[str, Any]
    rationale_short: str | None = None


class TransitionPrediction(StrictModel):
    """Predicted normalized properties for one state/action transition."""
    success: float = Field(ge=0, le=1)
    goal_progress: float = Field(ge=0, le=1)
    information_gain: float = Field(ge=0, le=1)
    risk: float = Field(ge=0, le=1)
    reversible: float = Field(ge=0, le=1)
    needs_more_observation: float = Field(ge=0, le=1)
    provider: str
    calibrated: bool = False
    raw: dict[str, Any] = Field(default_factory=dict)
    latency_ms: float = Field(default=0, ge=0)


class ObservedProperties(StrictModel):
    """Ground-truth transition labels emitted by the environment."""
    success: float = Field(ge=0, le=1)
    goal_progress: float = Field(ge=0, le=1)
    information_gain: float = Field(ge=0, le=1)
    risk: float = Field(ge=0, le=1)
    reversible: float = Field(ge=0, le=1)
    needs_more_observation: float = Field(ge=0, le=1)
    provenance: dict[str, Literal["deterministic", "heuristic", "llm"]]


class Transition(StrictModel):
    """Auditable record of applying one action to one state."""
    before: AgentState
    action: CandidateAction
    after: AgentState
    observed: ObservedProperties
    reward: float
    error: str | None = None
