from __future__ import annotations

import json
import os
import time
from pathlib import Path

from .models import AgentState, CandidateAction


class DeterministicPlanner:
    """Fixture planner for reproducible smoke tests; it is not a GPT result."""

    async def propose_actions(self, state: AgentState, max_actions: int = 5) -> list[CandidateAction]:
        if state.path == "/manufacturer" and not state.unknowns:
            items = [CandidateAction(action_id="answer-15", tool="answer", args={"value": "15 V"}, rationale_short="Answer from primary source")]
        else:
            items = [
                CandidateAction(action_id="forum", tool="navigate", args={"path": "/forum"}),
                CandidateAction(action_id="manufacturer", tool="navigate", args={"path": "/manufacturer"}),
                CandidateAction(action_id="old", tool="navigate", args={"path": "/old"}),
                CandidateAction(action_id="danger", tool="navigate", args={"path": "/danger"}),
                CandidateAction(action_id="premature", tool="answer", args={"value": "12 V"}),
            ]
        return items[:max_actions]


class OpenAIPlanner:
    def __init__(self, model: str | None = None, prompt_path: str = "prompts/planner_v001.md") -> None:
        from openai import AsyncOpenAI
        self.client = AsyncOpenAI()
        self.model = model or os.getenv("OPENAI_MODEL", "gpt-5-mini")
        self.prompt = Path(prompt_path).read_text()
        self.last_usage: dict = {}
        self.last_latency_ms = 0.0

    async def propose_actions(self, state: AgentState, max_actions: int = 5) -> list[CandidateAction]:
        started = time.perf_counter()
        response = await self.client.responses.create(
            model=self.model,
            input=self.prompt + "\nSTATE:\n" + state.canonical_json() + f"\nReturn at most {max_actions} actions.",
            text={"format": {"type": "json_schema", "name": "candidates", "strict": True, "schema": {
                "type": "object", "properties": {"actions": {"type": "array", "maxItems": max_actions, "items": {
                    "type": "object", "properties": {"action_id": {"type": "string"}, "tool": {"type": "string", "enum": ["navigate", "answer", "observe"]}, "args": {"type": "object", "additionalProperties": True}, "rationale_short": {"type": ["string", "null"]}},
                    "required": ["action_id", "tool", "args", "rationale_short"], "additionalProperties": False}},}, "required": ["actions"], "additionalProperties": False}}},
        )
        self.last_latency_ms = (time.perf_counter() - started) * 1000
        self.last_usage = response.usage.model_dump() if response.usage else {}
        return [CandidateAction.model_validate(x) for x in json.loads(response.output_text)["actions"]]

