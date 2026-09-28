from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

from .models import AgentState, CandidateAction


class DeterministicPlanner:
    """Fixture planner for reproducible smoke tests; it is not a GPT result."""

    async def propose_actions(self, state: AgentState, max_actions: int = 5) -> list[CandidateAction]:
        task_answers = {"voltage-001": ("/manufacturer", "15 V"), "warranty-001": ("/warranty", "3 years"), "temperature-001": ("/relay", "85 C")}
        primary_path, answer = task_answers[state.task_id]
        if state.path == primary_path and not state.unknowns:
            items = [CandidateAction(action_id="answer-primary", tool="answer", args={"value": answer}, rationale_short="Answer from primary source")]
        else:
            items = [
                CandidateAction(action_id="forum", tool="navigate", args={"path": "/forum"}),
                CandidateAction(action_id="primary", tool="navigate", args={"path": primary_path}),
                CandidateAction(action_id="old", tool="navigate", args={"path": "/old"}),
                CandidateAction(action_id="danger", tool="navigate", args={"path": "/danger"}),
                CandidateAction(action_id="premature", tool="answer", args={"value": "12 V"}),
            ]
        return items[:max_actions]


class CachedPlanner:
    """Persist planner outputs so selector variants reuse identical candidates."""

    def __init__(self, planner: object, cache_path: Path, *, prompt_version: str = "planner-v001", seed: int = 0) -> None:
        self.planner = planner
        self.cache_path = cache_path
        self.prompt_version = prompt_version
        self.seed = seed
        self.model = getattr(planner, "model", "deterministic")
        self.cache_hits = 0
        self.cache_misses = 0
        self._last_usage: dict = {}
        self.total_usage: dict[str, float] = {}
        self._items: dict[str, list[dict]] = json.loads(cache_path.read_text()) if cache_path.exists() else {}

    def _key(self, state: AgentState, max_actions: int) -> str:
        raw = json.dumps({"state_hash": state.state_hash, "model": self.model, "prompt_version": self.prompt_version, "seed": self.seed, "max_actions": max_actions}, sort_keys=True)
        return hashlib.sha256(raw.encode()).hexdigest()

    async def propose_actions(self, state: AgentState, max_actions: int = 5) -> list[CandidateAction]:
        key = self._key(state, max_actions)
        if key in self._items:
            self.cache_hits += 1
            self._last_usage = {}
            return [CandidateAction.model_validate(item) for item in self._items[key]]
        self.cache_misses += 1
        actions = await self.planner.propose_actions(state, max_actions)
        self._last_usage = getattr(self.planner, "last_usage", {})
        for name, value in self._last_usage.items():
            if isinstance(value, (int, float)):
                self.total_usage[name] = self.total_usage.get(name, 0) + value
        self._items[key] = [a.model_dump(mode="json") for a in actions]
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.cache_path.write_text(json.dumps(self._items, indent=2, sort_keys=True) + "\n")
        return actions

    @property
    def last_usage(self) -> dict:
        return self._last_usage


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


class OpenRouterPlanner:
    """OpenRouter planner using its OpenAI-compatible chat endpoint."""

    def __init__(self, model: str | None = None, prompt_path: str = "prompts/planner_v001.md") -> None:
        from openai import AsyncOpenAI
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise RuntimeError("OPENROUTER_API_KEY is required for the OpenRouter provider")
        headers = {"X-OpenRouter-Title": os.getenv("OPENROUTER_APP_NAME", "Laya Dynamics Agent")}
        if referer := os.getenv("OPENROUTER_HTTP_REFERER"):
            headers["HTTP-Referer"] = referer
        self.client = AsyncOpenAI(api_key=api_key, base_url="https://openrouter.ai/api/v1", default_headers=headers)
        self.model = model or os.getenv("OPENROUTER_MODEL", "openai/gpt-5.6-sol")
        self.prompt = Path(prompt_path).read_text()
        self.last_usage: dict = {}
        self.last_latency_ms = 0.0

    async def propose_actions(self, state: AgentState, max_actions: int = 5) -> list[CandidateAction]:
        started = time.perf_counter()
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "system", "content": self.prompt}, {"role": "user", "content": "STATE:\n" + state.canonical_json() + f"\nReturn JSON with at most {max_actions} actions."}],
            response_format={"type": "json_schema", "json_schema": {"name": "candidate_actions", "strict": True, "schema": {
                "type": "object", "properties": {"actions": {"type": "array", "minItems": 1, "maxItems": max_actions, "items": _ACTION_SCHEMA}},
                "required": ["actions"], "additionalProperties": False,
            }}},
            extra_body={"provider": {"require_parameters": True}},
        )
        self.last_latency_ms = (time.perf_counter() - started) * 1000
        self.last_usage = response.usage.model_dump() if response.usage else {}
        content = response.choices[0].message.content
        if not content:
            raise RuntimeError("OpenRouter returned an empty planner response")
        actions = [_parse_structured_action(item) for item in json.loads(content)["actions"]]
        if not 1 <= len(actions) <= max_actions:
            raise ValueError(f"OpenRouter returned {len(actions)} actions; expected 1..{max_actions}")
        return actions



class DeterministicDirectPlanner:
    """Offline stand-in for the direct-agent control, without candidate generation."""

    model = "deterministic-direct"
    last_usage: dict = {}

    async def propose_actions(self, state: AgentState, max_actions: int = 1) -> list[CandidateAction]:
        primary, answer = {"voltage-001": ("/manufacturer", "15 V"), "warranty-001": ("/warranty", "3 years"), "temperature-001": ("/relay", "85 C")}[state.task_id]
        if state.unknowns:
            return [CandidateAction(action_id="direct-primary", tool="navigate", args={"path": primary}, rationale_short="Gather primary evidence")]
        return [CandidateAction(action_id="direct-answer", tool="answer", args={"value": answer}, rationale_short="Answer from gathered evidence")]


_ACTION_SCHEMA = {
    "type": "object",
    "properties": {
        "action_id": {"type": "string"},
        "tool": {"type": "string", "enum": ["navigate", "answer", "observe"]},
        "args": {
            "type": "object",
            "properties": {"path": {"type": ["string", "null"]}, "value": {"type": ["string", "null"]}},
            "required": ["path", "value"],
            "additionalProperties": False,
        },
        "rationale_short": {"type": ["string", "null"]},
    },
    "required": ["action_id", "tool", "args", "rationale_short"],
    "additionalProperties": False,
}
_DIRECT_SCHEMA = {"type": "object", "properties": {"action": _ACTION_SCHEMA}, "required": ["action"], "additionalProperties": False}


def _parse_structured_action(item: dict) -> CandidateAction:
    normalized = dict(item)
    normalized["args"] = {key: value for key, value in item.get("args", {}).items() if value is not None}
    return CandidateAction.model_validate(normalized)

_DIRECT_PROMPT = "Choose exactly one next action for the controlled research sandbox. Return one JSON object with an action field. Prefer primary evidence before answering. Use only tools and paths present in the state."


class OpenAIDirectPlanner:
    def __init__(self, model: str | None = None) -> None:
        from openai import AsyncOpenAI
        self.client = AsyncOpenAI()
        self.model = model or os.getenv("OPENAI_MODEL", "gpt-5-mini")
        self.last_usage: dict = {}
        self.last_latency_ms = 0.0

    async def propose_actions(self, state: AgentState, max_actions: int = 1) -> list[CandidateAction]:
        started = time.perf_counter()
        response = await self.client.responses.create(model=self.model, input=_DIRECT_PROMPT + "\nSTATE:\n" + state.canonical_json(), text={"format": {"type": "json_schema", "name": "direct_action", "strict": True, "schema": _DIRECT_SCHEMA}})
        self.last_latency_ms = (time.perf_counter() - started) * 1000
        self.last_usage = response.usage.model_dump() if response.usage else {}
        return [_parse_structured_action(json.loads(response.output_text)["action"])]


class OpenRouterDirectPlanner:
    def __init__(self, model: str | None = None) -> None:
        from openai import AsyncOpenAI
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise RuntimeError("OPENROUTER_API_KEY is required for the OpenRouter provider")
        headers = {"X-OpenRouter-Title": os.getenv("OPENROUTER_APP_NAME", "Laya Dynamics Agent")}
        if referer := os.getenv("OPENROUTER_HTTP_REFERER"):
            headers["HTTP-Referer"] = referer
        self.client = AsyncOpenAI(api_key=api_key, base_url="https://openrouter.ai/api/v1", default_headers=headers)
        self.model = model or os.getenv("OPENROUTER_MODEL", "openai/gpt-5.6-sol")
        self.last_usage: dict = {}
        self.last_latency_ms = 0.0

    async def propose_actions(self, state: AgentState, max_actions: int = 1) -> list[CandidateAction]:
        started = time.perf_counter()
        response = await self.client.chat.completions.create(model=self.model, messages=[{"role": "system", "content": _DIRECT_PROMPT}, {"role": "user", "content": "STATE:\n" + state.canonical_json()}], response_format={"type": "json_schema", "json_schema": {"name": "direct_action", "strict": True, "schema": _DIRECT_SCHEMA}}, extra_body={"provider": {"require_parameters": True}})
        self.last_latency_ms = (time.perf_counter() - started) * 1000
        self.last_usage = response.usage.model_dump() if response.usage else {}
        content = response.choices[0].message.content
        if not content:
            raise RuntimeError("OpenRouter returned an empty direct-action response")
        return [_parse_structured_action(json.loads(content)["action"])]
