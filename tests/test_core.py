import asyncio
from pathlib import Path
from laya_dynamics_agent.benchmark import run_benchmark_suite
from laya_dynamics_agent.cli import concise_error, print_benchmark_summary
from laya_dynamics_agent.models import CandidateAction
from laya_dynamics_agent.planners import _DIRECT_PROMPT, CachedPlanner, DeterministicDirectPlanner, DeterministicPlanner, OpenRouterPlanner
from laya_dynamics_agent.policies import GPTOnlyPolicy, GreedyUtilityPolicy
from laya_dynamics_agent.predictors import HeuristicPredictor, LayaPredictor
from laya_dynamics_agent.runner_impl import run_episode
from laya_dynamics_agent.sandbox import SandboxWebEnvironment, TASKS
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
    assert report["aggregate"]["episodes"] == len(TASKS) * 2
    assert report["prompt_versions"] == {"direct": "direct-v001", "candidates": "planner-v001"}
    assert (tmp_path / "reports/latest/metrics.json").exists()


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
    assert "Raw metrics: reports/latest/metrics.json" in output
