"""Direct and candidate planners, including auditable candidate replay."""

from __future__ import annotations

import hashlib
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

from .models import AgentState, CandidateAction


def _openrouter_max_tokens() -> int:
    """Read the bounded response ceiling used to avoid cost-envelope errors."""
    raw = os.getenv("OPENROUTER_MAX_TOKENS", "4096")
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError("OPENROUTER_MAX_TOKENS must be an integer") from exc
    if not 256 <= value <= 16384:
        raise ValueError("OPENROUTER_MAX_TOKENS must be between 256 and 16384")
    return value


class DeterministicPlanner:
    """Fixture planner for reproducible smoke tests; it is not a GPT result."""

    async def propose_actions(self, state: AgentState, max_actions: int = 5) -> list[CandidateAction]:
        from .sandbox import TASKS
        primary_path, answer = TASKS[state.task_id]["primary_path"], TASKS[state.task_id]["answer"]
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
    """Persist actions and generation provenance for exact, auditable replay."""

    SCHEMA_VERSION = "2.0"

    def __init__(self, planner: object, cache_path: Path, *, prompt_version: str = "planner-v001", seed: int = 0) -> None:
        self.planner = planner
        self.cache_path = cache_path
        self.prompt_version = prompt_version
        self.seed = seed
        self.model = getattr(planner, "model", "deterministic")
        self.provider = type(planner).__name__
        self.cache_hits = 0
        self.cache_misses = 0
        self._last_usage: dict = {}
        self.last_provenance: dict = {}
        self.fresh_usage: dict[str, float] = {}
        self.replayed_usage: dict[str, float] = {}
        self.unknown_usage_entries = 0
        self._accounted_keys: set[str] = set()
        self._loaded_legacy = False
        self._entries = self._load()
        if self._loaded_legacy:
            self._write()

    def _load(self) -> dict[str, dict]:
        if not self.cache_path.exists():
            return {}
        raw = json.loads(self.cache_path.read_text())
        if raw.get("schema_version") == self.SCHEMA_VERSION and isinstance(raw.get("entries"), dict):
            return raw["entries"]
        # V1 stored key -> actions only. Preserve replayability while marking provenance unknown.
        self._loaded_legacy = True
        return {key: {"actions": actions, "metadata": {"legacy": True, "usage": {}}} for key, actions in raw.items()}

    def _write(self) -> None:
        payload = {"schema_version": self.SCHEMA_VERSION, "entries": self._entries}
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.cache_path.with_suffix(self.cache_path.suffix + ".tmp")
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        temporary.replace(self.cache_path)

    @staticmethod
    def _add_usage(target: dict[str, float], usage: dict) -> None:
        for name, value in usage.items():
            if isinstance(value, (int, float)):
                target[name] = target.get(name, 0) + value

    def _key(self, state: AgentState, max_actions: int) -> str:
        """Bind replay to state, model, prompt, seed, and candidate count."""
        raw = json.dumps({"state_hash": state.state_hash, "model": self.model, "prompt_version": self.prompt_version, "seed": self.seed, "max_actions": max_actions}, sort_keys=True)
        return hashlib.sha256(raw.encode()).hexdigest()

    async def propose_actions(self, state: AgentState, max_actions: int = 5) -> list[CandidateAction]:
        """Replay exact candidates or generate and atomically persist a miss."""
        lookup_started = time.perf_counter()
        key = self._key(state, max_actions)
        if key in self._entries:
            self.cache_hits += 1
            self._last_usage = {}
            entry = self._entries[key]
            metadata = entry.get("metadata", {})
            actions = [CandidateAction.model_validate(item) for item in entry["actions"]]
            lookup_ms = (time.perf_counter() - lookup_started) * 1000
            # Missing legacy latency stays unknown; it is never reconstructed.
            original_latency = metadata.get("latency_ms")
            valid_original_latency = isinstance(original_latency, (int, float)) and original_latency >= 0
            self.last_provenance = {
                "candidate_source": "cache", "candidate_cache_key": key,
                "candidate_generation_ms": 0.0, "candidate_cache_lookup_ms": lookup_ms,
                "replayed_candidate_generation_ms": float(original_latency) if valid_original_latency else None,
                "effective_latency_complete": valid_original_latency,
                "candidate_generation_provider": metadata.get("provider"),
                "candidate_generation_model": metadata.get("model"),
            }
            if key not in self._accounted_keys:
                usage = metadata.get("usage", {})
                if metadata.get("legacy") or not usage:
                    self.unknown_usage_entries += 1
                else:
                    self._add_usage(self.replayed_usage, usage)
                self._accounted_keys.add(key)
            return actions
        self.cache_misses += 1
        lookup_ms = (time.perf_counter() - lookup_started) * 1000
        started = time.perf_counter()
        actions = await self.planner.propose_actions(state, max_actions)
        latency_ms = (time.perf_counter() - started) * 1000
        self._last_usage = getattr(self.planner, "last_usage", {})
        self.last_provenance = {
            "candidate_source": "fresh", "candidate_cache_key": key,
            "candidate_generation_ms": latency_ms, "candidate_cache_lookup_ms": lookup_ms,
            "replayed_candidate_generation_ms": 0.0, "effective_latency_complete": True,
            "candidate_generation_provider": self.provider,
            "candidate_generation_model": self.model,
        }
        self._add_usage(self.fresh_usage, self._last_usage)
        self._accounted_keys.add(key)
        self._entries[key] = {
            "actions": [a.model_dump(mode="json") for a in actions],
            "metadata": {
                "created_at": datetime.now(timezone.utc).isoformat(),
                "provider": self.provider,
                "model": self.model,
                "prompt_version": self.prompt_version,
                "seed": self.seed,
                "max_actions": max_actions,
                "state_hash": state.state_hash,
                "latency_ms": latency_ms,
                "usage": self._last_usage,
            },
        }
        self._write()
        return actions

    @property
    def total_usage(self) -> dict[str, float]:
        total = dict(self.fresh_usage)
        self._add_usage(total, self.replayed_usage)
        return total

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
        self.max_tokens = _openrouter_max_tokens()
        self.seed = 0
        self.prompt = Path(prompt_path).read_text()
        self.last_usage: dict = {}
        self.last_latency_ms = 0.0

    async def propose_actions(self, state: AgentState, max_actions: int = 5) -> list[CandidateAction]:
        started = time.perf_counter()
        response = await self.client.chat.completions.create(
            model=self.model,
            seed=self.seed,
            messages=[{"role": "system", "content": self.prompt}, {"role": "user", "content": "STATE:\n" + state.canonical_json() + f"\nReturn JSON with at most {max_actions} actions."}],
            response_format={"type": "json_schema", "json_schema": {"name": "candidate_actions", "strict": True, "schema": {
                "type": "object", "properties": {"actions": {"type": "array", "minItems": 1, "maxItems": max_actions, "items": _ACTION_SCHEMA}},
                "required": ["actions"], "additionalProperties": False,
            }}},
            max_tokens=self.max_tokens,
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
        from .sandbox import TASKS
        primary, answer = TASKS[state.task_id]["primary_path"], TASKS[state.task_id]["answer"]
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
        self.max_tokens = _openrouter_max_tokens()
        self.seed = 0
        self.last_usage: dict = {}
        self.last_latency_ms = 0.0

    async def propose_actions(self, state: AgentState, max_actions: int = 1) -> list[CandidateAction]:
        started = time.perf_counter()
        response = await self.client.chat.completions.create(model=self.model, seed=self.seed, messages=[{"role": "system", "content": _DIRECT_PROMPT}, {"role": "user", "content": "STATE:\n" + state.canonical_json()}], response_format={"type": "json_schema", "json_schema": {"name": "direct_action", "strict": True, "schema": _DIRECT_SCHEMA}}, max_tokens=self.max_tokens, extra_body={"provider": {"require_parameters": True}})
        self.last_latency_ms = (time.perf_counter() - started) * 1000
        self.last_usage = response.usage.model_dump() if response.usage else {}
        content = response.choices[0].message.content
        if not content:
            raise RuntimeError("OpenRouter returned an empty direct-action response")
        return [_parse_structured_action(json.loads(content)["action"])]
