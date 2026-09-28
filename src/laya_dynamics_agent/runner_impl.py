from __future__ import annotations

import time
from typing import Any

from .models import TransitionPrediction


def prediction_error(predicted: TransitionPrediction, observed: Any) -> dict[str, float]:
    keys = ("success", "goal_progress", "information_gain", "risk", "reversible", "needs_more_observation")
    return {key: abs(getattr(predicted, key) - getattr(observed, key)) for key in keys}


async def run_episode(*, run_id: str, task_id: str, environment: Any, planner: Any, predictor: Any | None, policy: Any, store: Any, max_steps: int = 6, max_actions: int = 5, stop_requested: Any | None = None) -> dict[str, Any]:
    state = environment.reset(task_id); seen = {}; action_ids = []; stop_reason = "max_steps"; errors = []
    for _ in range(max_steps):
        if stop_requested and stop_requested(): stop_reason = "interrupted"; break
        seen[state.state_hash] = seen.get(state.state_hash, 0) + 1
        if seen[state.state_hash] > 1: stop_reason = "repeated_state"; break
        started = time.perf_counter(); actions = await planner.propose_actions(state, max_actions); planner_ms = (time.perf_counter()-started)*1000
        if not actions: stop_reason = "no_candidates"; break
        predictions = predictor.predict(state, actions) if predictor else {}
        chosen = policy.select(state, actions, predictions); transition = environment.step(chosen)
        err = prediction_error(predictions[chosen.action_id], transition.observed) if chosen.action_id in predictions else {}
        if err: errors.append(err)
        store.save_step(run_id, task_id, state.step_index, state_before=state, candidates=actions, predictions=predictions, chosen_action=chosen, state_after=transition.after, observed=transition.observed, reward=transition.reward, timings={"planner_ms": planner_ms, "predictor_ms": sum(p.latency_ms for p in predictions.values()), "prediction_error": err}, error=transition.error)
        action_ids.append(chosen.action_id); state = transition.after
        if stop_requested and stop_requested(): stop_reason = "interrupted"; break
        if state.terminal: stop_reason = "success" if state.success else "terminal_failure"; break
        if len(action_ids) >= 4 and action_ids[-4:-2] == action_ids[-2:]: stop_reason = "oscillation"; break
        if len(action_ids) >= 2 and action_ids[-1] == action_ids[-2]: stop_reason = "repeated_action"; break
    return {"task_id": task_id, "success": bool(state.success), "steps": state.step_index, "stop_reason": stop_reason, "actions": action_ids, "mean_prediction_mae": (sum(sum(x.values())/len(x) for x in errors)/len(errors) if errors else None)}

