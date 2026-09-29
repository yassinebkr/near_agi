"""Episode execution with durable, non-overlapping latency telemetry.

Timing boundaries live here so every benchmark arm remains comparable.
"""

from __future__ import annotations

import time
from typing import Any

from .models import TransitionPrediction


def prediction_error(predicted: TransitionPrediction, observed: Any) -> dict[str, float]:
    """Return per-property absolute error in the normalized label space."""
    keys = ("success", "goal_progress", "information_gain", "risk", "reversible", "needs_more_observation")
    return {key: abs(getattr(predicted, key) - getattr(observed, key)) for key in keys}


async def run_episode(*, run_id: str, task_id: str, environment: Any, planner: Any,
                      predictor: Any | None, policy: Any, store: Any, max_steps: int = 6,
                      max_actions: int = 5, stop_requested: Any | None = None) -> dict[str, Any]:
    """Run one episode and return its outcome plus complete timing telemetry.

    Planner wall time contains cache lookup and candidate generation. Predictor
    wall time encloses the call; reported time is the sum attached to outputs.
    """
    episode_started = time.perf_counter()
    environment_started = time.perf_counter()
    state = environment.reset(task_id)
    environment_ms = (time.perf_counter() - environment_started) * 1000
    seen: dict[str, int] = {}
    action_ids: list[str] = []
    stop_reason = "max_steps"
    errors: list[dict[str, float]] = []
    unsafe_actions = 0
    planner_wall_ms = 0.0
    candidate_generation_ms = 0.0
    candidate_cache_lookup_ms = 0.0
    replayed_candidate_generation_ms = 0.0
    predictor_wall_ms = 0.0
    predictor_reported_ms = 0.0
    policy_ms = 0.0
    candidate_fresh_count = 0
    candidate_cache_hit_count = 0
    candidate_effective_complete = True
    candidate_path = False
    usage_totals: dict[str, float] = {}

    for _ in range(max_steps):
        if stop_requested and stop_requested():
            stop_reason = "interrupted"
            break
        seen[state.state_hash] = seen.get(state.state_hash, 0) + 1
        if seen[state.state_hash] > 1:
            stop_reason = "repeated_state"
            break

        started = time.perf_counter()
        actions = await planner.propose_actions(state, max_actions)
        planner_step_ms = (time.perf_counter() - started) * 1000
        planner_wall_ms += planner_step_ms
        # Direct planners expose no provenance; CachedPlanner publishes the
        # source and original miss latency immediately after every call.
        provenance = dict(getattr(planner, "last_provenance", {}) or {})
        source = provenance.get("candidate_source")
        candidate_path = candidate_path or source in {"fresh", "cache"}
        generation_step_ms = float(provenance.get("candidate_generation_ms") or 0.0)
        lookup_step_ms = float(provenance.get("candidate_cache_lookup_ms") or 0.0)
        replayed_step = provenance.get("replayed_candidate_generation_ms")
        candidate_generation_ms += generation_step_ms
        candidate_cache_lookup_ms += lookup_step_ms
        if source == "fresh":
            candidate_fresh_count += 1
        elif source == "cache":
            candidate_cache_hit_count += 1
            if replayed_step is None or not provenance.get("effective_latency_complete", False):
                candidate_effective_complete = False
            else:
                replayed_candidate_generation_ms += float(replayed_step)

        for key, value in getattr(planner, "last_usage", {}).items():
            if isinstance(value, (int, float)):
                usage_totals[key] = usage_totals.get(key, 0) + value
        if not actions:
            stop_reason = "no_candidates"
            break

        started = time.perf_counter()
        predictions = predictor.predict(state, actions) if predictor else {}
        predictor_step_wall_ms = (time.perf_counter() - started) * 1000
        predictor_step_reported_ms = sum(p.latency_ms for p in predictions.values())
        predictor_wall_ms += predictor_step_wall_ms
        predictor_reported_ms += predictor_step_reported_ms

        started = time.perf_counter()
        chosen = policy.select(state, actions, predictions)
        policy_step_ms = (time.perf_counter() - started) * 1000
        policy_ms += policy_step_ms

        started = time.perf_counter()
        transition = environment.step(chosen)
        environment_step_ms = (time.perf_counter() - started) * 1000
        environment_ms += environment_step_ms

        unsafe_actions += int(transition.observed.risk > 0)
        err = prediction_error(predictions[chosen.action_id], transition.observed) if chosen.action_id in predictions else {}
        if err:
            errors.append(err)
        timings = {
            "planner_wall_ms": planner_step_ms,
            "candidate_generation_ms": generation_step_ms,
            "candidate_cache_lookup_ms": lookup_step_ms,
            "replayed_candidate_generation_ms": replayed_step,
            "candidate_source": source,
            "candidate_cache_key": provenance.get("candidate_cache_key"),
            "candidate_generation_provider": provenance.get("candidate_generation_provider"),
            "candidate_generation_model": provenance.get("candidate_generation_model"),
            "effective_latency_complete": provenance.get("effective_latency_complete") if candidate_path else None,
            "predictor_wall_ms": predictor_step_wall_ms,
            "predictor_reported_ms": predictor_step_reported_ms,
            "policy_ms": policy_step_ms,
            "selector_wall_ms": predictor_step_wall_ms + policy_step_ms if candidate_path else 0.0,
            "environment_ms": environment_step_ms,
            "prediction_error": err,
        }
        store.save_step(run_id, task_id, state.step_index, state_before=state, candidates=actions,
                        predictions=predictions, chosen_action=chosen, state_after=transition.after,
                        observed=transition.observed, reward=transition.reward, timings=timings,
                        error=transition.error)
        action_ids.append(chosen.action_id)
        state = transition.after
        if stop_requested and stop_requested():
            stop_reason = "interrupted"
            break
        if state.terminal:
            stop_reason = "success" if state.success else "terminal_failure"
            break
        if len(action_ids) >= 4 and action_ids[-4:-2] == action_ids[-2:]:
            stop_reason = "oscillation"
            break
        if len(action_ids) >= 2 and action_ids[-1] == action_ids[-2]:
            stop_reason = "repeated_action"
            break

    wall_clock_ms = (time.perf_counter() - episode_started) * 1000
    # Generation and lookup are already inside planner time: do not double-count.
    measured_components = planner_wall_ms + predictor_wall_ms + policy_ms + environment_ms
    framework_overhead_ms = max(0.0, wall_clock_ms - measured_components)
    if candidate_path:
        # Reconstruct cold cost only when every replay retained measured latency.
        effective_end_to_end_ms = wall_clock_ms + replayed_candidate_generation_ms if candidate_effective_complete else None
        candidate_path_temperature = ("mixed" if candidate_fresh_count and candidate_cache_hit_count else
                                      "cold" if candidate_fresh_count else "warm")
    else:
        effective_end_to_end_ms = None
        candidate_effective_complete = False
        candidate_path_temperature = "not_applicable"
    return {
        "task_id": task_id,
        "success": bool(state.success),
        "steps": state.step_index,
        "stop_reason": stop_reason,
        "actions": action_ids,
        "mean_prediction_mae": (sum(sum(x.values()) / len(x) for x in errors) / len(errors) if errors else None),
        "wall_clock_ms": wall_clock_ms,
        "planner_wall_ms": planner_wall_ms,
        "planner_latency_ms": planner_wall_ms,
        "candidate_generation_ms": candidate_generation_ms,
        "candidate_cache_lookup_ms": candidate_cache_lookup_ms,
        "replayed_candidate_generation_ms": replayed_candidate_generation_ms,
        "candidate_fresh_count": candidate_fresh_count,
        "candidate_cache_hit_count": candidate_cache_hit_count,
        "candidate_path_temperature": candidate_path_temperature,
        "effective_end_to_end_ms_reconstructed": effective_end_to_end_ms,
        "effective_end_to_end_complete": candidate_effective_complete,
        "predictor_wall_ms": predictor_wall_ms,
        "predictor_reported_ms": predictor_reported_ms,
        "predictor_latency_ms": predictor_reported_ms,
        "policy_ms": policy_ms,
        "selector_wall_ms": predictor_wall_ms + policy_ms if candidate_path else 0.0,
        "environment_ms": environment_ms,
        "framework_overhead_ms": framework_overhead_ms,
        "usage": usage_totals,
        "unsafe_actions": unsafe_actions,
        "unnecessary_actions": max(0, state.step_index - 2) if state.success else state.step_index,
    }
