import asyncio
from collections import Counter
from pathlib import Path
from laya_dynamics_agent.benchmark import run_benchmark_suite
from laya_dynamics_agent.cli import concise_error, print_benchmark_summary
from laya_dynamics_agent.models import CandidateAction
from laya_dynamics_agent.planners import _ACTION_SCHEMA, _DIRECT_PROMPT, _parse_structured_action, CachedPlanner, DeterministicDirectPlanner, DeterministicPlanner, OpenRouterPlanner
from laya_dynamics_agent.policies import GPTOnlyPolicy, GreedyUtilityPolicy
from laya_dynamics_agent.predictors import HeuristicPredictor, LayaPredictor, resolve_laya_device
from laya_dynamics_agent.runner_impl import run_episode
from laya_dynamics_agent.sandbox import SUITES, SandboxWebEnvironment, TASKS, answer_matches
from laya_dynamics_agent.storage_v2 import TrajectoryStore

def test_state_hash_is_stable():
    env=SandboxWebEnvironment();a=env.reset("voltage-001");b=env.reset("voltage-001");assert a.state_hash==b.state_hash

def test_deterministic_labels_and_safety():
    env=SandboxWebEnvironment();env.reset("voltage-001");t=env.step(CandidateAction(action_id="x",tool="navigate",args={"path":"/danger"}));assert t.observed.risk==1 and t.observed.reversible==0

def test_laya_question_mapping():
    q=LayaPredictor.questions();assert set(q)=={"success","goal_progress","information_gain","risk","reversible","needs_more_observation"};assert q["information_gain"]["type"]=="score"

def test_episode_and_reconstruction(tmp_path:Path):
    store=TrajectoryStore(tmp_path/"t.sqlite",tmp_path/"logs");store.start_run("r",{"test":True})
    result=asyncio.run(run_episode(run_id="r",task_id="voltage-001",environment=SandboxWebEnvironment(),planner=DeterministicPlanner(),predictor=HeuristicPredictor(),policy=GreedyUtilityPolicy(),store=store))
    assert result["success"] and result["steps"]==2;rows=store.reconstruct("r","voltage-001");assert len(rows)==2 and rows[-1]["state_after"]["success"] is True

def test_gpt_only_control_selects_first():
    env=SandboxWebEnvironment();state=env.reset("voltage-001");actions=asyncio.run(DeterministicPlanner().propose_actions(state,3));assert GPTOnlyPolicy().select(state,actions,{})==actions[0]



def test_openrouter_requires_its_own_key(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    try:
        OpenRouterPlanner()
    except RuntimeError as exc:
        assert "OPENROUTER_API_KEY" in str(exc)
    else:
        raise AssertionError("missing OpenRouter key should fail closed")


def test_graceful_stop_records_partial_run(tmp_path:Path):
    store=TrajectoryStore(tmp_path/"t.sqlite",tmp_path/"logs");store.start_run("interrupt",{"test":True})
    result=asyncio.run(run_episode(run_id="interrupt",task_id="voltage-001",environment=SandboxWebEnvironment(),planner=DeterministicPlanner(),predictor=None,policy=GPTOnlyPolicy(),store=store,stop_requested=lambda:True))
    assert result["stop_reason"]=="interrupted" and result["steps"]==0
    store.finish_run("interrupt","interrupted",result)
    status=store.conn.execute("SELECT status FROM runs WHERE run_id='interrupt'").fetchone()[0]
    assert status=="interrupted"
    assert '"event_type": "run_interrupted"' in (tmp_path/"logs"/"interrupt"/"events.jsonl").read_text()
    store.close()


def test_all_sandbox_tasks_are_solvable(tmp_path: Path):
    for task_id in TASKS:
        store = TrajectoryStore(tmp_path / "tasks.sqlite", tmp_path / "task-logs")
        run_id = "solve-" + task_id
        store.start_run(run_id, {"test": True})
        result = asyncio.run(run_episode(run_id=run_id, task_id=task_id, environment=SandboxWebEnvironment(), planner=DeterministicDirectPlanner(), predictor=None, policy=GPTOnlyPolicy(), store=store))
        assert result["success"] and result["steps"] == 2
        store.close()


def test_challenge_suite_covers_all_tasks_and_multiple_templates():
    assert SUITES["challenge"] == tuple(task_id for task_id in TASKS if not task_id.startswith("final-"))
    assert len(SUITES["challenge"]) == 9
    assert len({TASKS[task_id]["template_id"] for task_id in SUITES["challenge"]}) >= 6


def test_final_suite_is_frozen_and_template_held_out():
    final_ids = SUITES["final"]
    counts = Counter(TASKS[task_id]["template_id"] for task_id in final_ids)
    development_templates = {TASKS[task_id]["template_id"] for task_id in SUITES["challenge"]}
    assert len(final_ids) == 90
    assert set(counts.values()) == {30}
    assert set(counts).isdisjoint(development_templates)


def test_final_dynamic_primary_page_and_answer():
    task_id = "final-security-029"
    task = TASKS[task_id]
    env = SandboxWebEnvironment()
    env.reset(task_id)
    transition = env.step(CandidateAction(action_id="primary", tool="navigate", args={"path": task["primary_path"]}))
    assert transition.after.unknowns == []
    transition = env.step(CandidateAction(action_id="answer", tool="answer", args={"value": f"The minimum safe firmware is {task["answer"]}."}))
    assert transition.after.success is True


def test_candidate_cache_replays_identical_actions(tmp_path: Path):
    state = SandboxWebEnvironment().reset("voltage-001")
    cache = CachedPlanner(DeterministicPlanner(), tmp_path / "candidates.json", seed=7)
    first = asyncio.run(cache.propose_actions(state))
    second = asyncio.run(cache.propose_actions(state))
    assert first == second
    assert cache.cache_misses == 1 and cache.cache_hits == 1


def test_benchmark_report_has_real_direct_arm_and_aggregate(tmp_path: Path, monkeypatch):
    class Event:
        @staticmethod
        def is_set():
            return False
    class Shutdown:
        event = Event()
    monkeypatch.chdir(tmp_path)
    report = asyncio.run(run_benchmark_suite(include_laya=False, provider="deterministic", model=None, shutdown=Shutdown(), seeds=(0,)))
    assert set(report["aggregate"]["modes"]) == {"direct_gpt", "candidates_heuristic"}
    assert report["aggregate"]["episodes"] == len(SUITES["smoke"]) * 2
    assert report["prompt_versions"] == {"direct": "direct-v001", "candidates": "planner-v001"}
    assert len(report["aggregate"]["templates"]) == 3
    comparison = report["aggregate"]["paired_comparisons"]["candidates_heuristic_vs_direct_gpt"]
    assert comparison["pairs"] == 3
    assert comparison["success_delta_ci95"] == [0.0, 0.0]
    assert (tmp_path / "reports/latest/metrics.json").exists()


def test_fresh_candidate_cache_is_campaign_scoped(tmp_path: Path, monkeypatch):
    class Event:
        def is_set(self):
            return False

    class Shutdown:
        event = Event()
    monkeypatch.chdir(tmp_path)
    report = asyncio.run(run_benchmark_suite(include_laya=False, provider="deterministic", model=None, shutdown=Shutdown(), seeds=(0,), fresh_candidate_cache=True, progress=lambda _: None))
    assert report["fresh_candidate_cache"] is True
    assert report["campaign_id"] in report["candidate_cache"]


def test_direct_openrouter_prompt_explicitly_requests_json():
    assert "json" in _DIRECT_PROMPT.lower()


def test_concise_provider_error_extracts_nested_message():
    class ProviderError(Exception):
        body = {"error": {"message": "Provider returned error", "metadata": {"raw": '{"error":{"message":"messages must contain json"}}'}}}
    assert concise_error(ProviderError("verbose fallback")) == "ProviderError: messages must contain json"


def test_terminal_summary_is_compact(capsys):
    report = {"campaign_id": "b-1", "provider": "deterministic", "model": "fixture", "cache_hits": 2, "cache_misses": 1, "aggregate": {"modes": {"direct_gpt": {"episodes": 3, "success_rate": 1.0, "mean_steps": 2.0, "unsafe_actions": 0, "predictor_latency_ms": 0.0}}}}
    print_benchmark_summary(report)
    output = capsys.readouterr().out
    assert "Benchmark complete" in output and "100.0%" in output
    assert "---------------------------" not in output
    assert "Raw metrics: reports/latest/metrics.json" in output


def test_openrouter_schema_is_strict_and_normalizes_nullable_args():
    assert _ACTION_SCHEMA["additionalProperties"] is False
    assert _ACTION_SCHEMA["properties"]["args"]["additionalProperties"] is False
    action = _parse_structured_action({"action_id": "a", "tool": "navigate", "args": {"path": "/forum", "value": None}, "rationale_short": None})
    assert action.args == {"path": "/forum"}


def test_natural_language_answers_from_live_run_are_accepted():
    assert answer_matches("voltage-001", "The official absolute maximum input voltage is 15 V.")
    assert answer_matches("warranty-001", "The Nova Drill’s official warranty duration is 3 years.")
    assert answer_matches("temperature-001", "The maximum operating temperature is 85 °C.")
    assert not answer_matches("voltage-001", "The archived value is 10 V.")
    assert not answer_matches("warranty-001", "The warranty is 2 years.")
    assert not answer_matches("temperature-001", "The relay is rated to 70 C.")


def test_environment_accepts_correct_fact_in_sentence():
    env = SandboxWebEnvironment()
    env.reset("temperature-001")
    env.step(CandidateAction(action_id="source", tool="navigate", args={"path": "/relay"}))
    transition = env.step(CandidateAction(action_id="answer", tool="answer", args={"value": "The Orion Relay's official maximum operating temperature is 85 °C."}))
    assert transition.after.success is True


def test_candidate_cache_persists_and_replays_usage_provenance(tmp_path: Path):
    planner = DeterministicPlanner()
    planner.model = "fixture-model"
    planner.last_usage = {"prompt_tokens": 10, "completion_tokens": 4, "total_tokens": 14, "cost": 0.001}
    state = SandboxWebEnvironment().reset("voltage-001")
    path = tmp_path / "provenance.json"
    fresh = CachedPlanner(planner, path, seed=3)
    asyncio.run(fresh.propose_actions(state))
    payload = __import__("json").loads(path.read_text())
    assert payload["schema_version"] == "2.0"
    entry = next(iter(payload["entries"].values()))
    assert entry["metadata"]["usage"]["total_tokens"] == 14
    replay = CachedPlanner(planner, path, seed=3)
    asyncio.run(replay.propose_actions(state))
    assert replay.replayed_usage["total_tokens"] == 14
    assert replay.unknown_usage_entries == 0


def test_candidate_cache_reads_legacy_entries_and_marks_unknown_usage(tmp_path: Path):
    planner = DeterministicPlanner()
    state = SandboxWebEnvironment().reset("voltage-001")
    path = tmp_path / "legacy.json"
    cache = CachedPlanner(planner, path)
    key = cache._key(state, 5)
    actions = asyncio.run(planner.propose_actions(state))
    path.write_text(__import__("json").dumps({key: [action.model_dump(mode="json") for action in actions]}))
    replay = CachedPlanner(planner, path)
    migrated = __import__("json").loads(path.read_text())
    assert migrated["schema_version"] == "2.0"
    assert asyncio.run(replay.propose_actions(state)) == actions
    assert replay.unknown_usage_entries == 1


def test_laya_device_policy_fails_closed_when_cuda_is_required():
    class Cuda:
        @staticmethod
        def is_available():
            return False
    class Version:
        cuda = "13.0"
    class Torch:
        __version__ = "test"
        cuda = Cuda()
        version = Version()
    assert resolve_laya_device("auto", Torch()) == "cpu"
    assert resolve_laya_device("cpu", Torch()) == "cpu"
    try:
        resolve_laya_device("cuda", Torch())
    except RuntimeError as exc:
        assert "CUDA is unavailable" in str(exc)
    else:
        raise AssertionError("strict CUDA policy must fail closed")


def test_laya_device_policy_selects_available_cuda():
    class Cuda:
        @staticmethod
        def is_available():
            return True
    class Torch:
        cuda = Cuda()
    assert resolve_laya_device("auto", Torch()) == "cuda"
    assert resolve_laya_device("cuda", Torch()) == "cuda"
