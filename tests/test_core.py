import asyncio
from collections import Counter
from pathlib import Path
from laya_dynamics_agent.benchmark import _paired_comparisons, run_benchmark_suite
from laya_dynamics_agent.cli import concise_error, print_benchmark_summary
from laya_dynamics_agent.models import CandidateAction
from laya_dynamics_agent.planners import _ACTION_SCHEMA, _DIRECT_PROMPT, _parse_structured_action, CachedPlanner, DeterministicDirectPlanner, DeterministicPlanner, OpenRouterPlanner
from laya_dynamics_agent.policies import GPTOnlyPolicy, GreedyUtilityPolicy
from laya_dynamics_agent.predictors import HeuristicPredictor, LayaPredictor, resolve_laya_device
from laya_dynamics_agent.runner_impl import run_episode
from laya_dynamics_agent.sandbox import SUITES, SandboxWebEnvironment, TASKS, answer_matches
from laya_dynamics_agent.storage_v2 import TrajectoryStore
from laya_dynamics_agent.posttrain import (
    _format_train_event,
    _restore_rng_state,
    _training_repetitions,
    promotion_gate,
    verify_dataset,
)
from laya_dynamics_agent.training_data import DEFAULT_SPLIT_SIZES, PROPERTIES, build_dataset

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
    assert report["suite_version"] == "smoke-v001"
    assert len(report["task_manifest_sha256"]) == 64
    comparison = report["aggregate"]["paired_comparisons"]["candidates_heuristic_vs_direct_gpt"]
    assert comparison["pairs"] == 3
    assert comparison["success_delta_ci95"] == [0.0, 0.0]
    assert comparison["pairs_valid_time_to_success"] == 3
    assert report["aggregate"]["modes"]["direct_gpt"]["p95_wall_clock_ms"] is not None
    assert all("wall_clock_ms" in row for row in report["results"])
    assert report["aggregate"]["modes"]["candidates_heuristic"]["candidate_fresh_count"] > 0
    assert report["final_protocol"]["compliant"] is False
    assert (tmp_path / "reports/latest/metrics.json").exists()


def test_benchmark_supports_base_and_finetuned_laya_arms(tmp_path: Path, monkeypatch):
    class Event:
        @staticmethod
        def is_set():
            return False

    class Shutdown:
        event = Event()

    class FakeLayaPredictor:
        def __init__(self, checkpoint):
            self.checkpoint = checkpoint
            self.device = "cuda"
            self.load_time_ms = 1.0
            self.delegate = HeuristicPredictor()

        def predict(self, state, actions):
            return self.delegate.predict(state, actions)

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("laya_dynamics_agent.benchmark.LayaPredictor", FakeLayaPredictor)
    report = asyncio.run(run_benchmark_suite(
        include_laya=True, provider="deterministic", model=None, shutdown=Shutdown(), seeds=(0,),
        base_laya_checkpoint="base", finetuned_laya_checkpoint="fine", progress=lambda _: None,
    ))

    assert set(report["aggregate"]["modes"]) == {
        "direct_gpt", "candidates_heuristic", "candidates_base_laya",
        "candidates_finetuned_laya",
    }
    assert report["laya_runtimes"]["candidates_base_laya"]["checkpoint"] == "base"
    assert report["laya_runtimes"]["candidates_finetuned_laya"]["checkpoint"] == "fine"
    comparison = report["aggregate"]["paired_comparisons"][
        "candidates_finetuned_laya_vs_candidates_base_laya"
    ]
    assert comparison["pairs"] == len(SUITES["smoke"])
    assert report["final_protocol"]["base_laya_enabled"] is True
    assert report["final_protocol"]["finetuned_laya_enabled"] is True
    assert report["final_protocol"]["compliant"] is False


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


def test_posttraining_dataset_is_deterministic_disjoint_and_final_free(tmp_path: Path):
    sizes = {split: 2 for split in DEFAULT_SPLIT_SIZES}
    first = build_dataset(tmp_path / "first", split_sizes=sizes)
    second = build_dataset(tmp_path / "second", split_sizes=sizes)
    assert first == second
    assert first["splits"]["train"]["question_sequences"] == 2 * 4 * 5 * len(PROPERTIES)
    assert not (set(first["splits"]["train"]["templates"]) & set(first["splits"]["development"]["templates"]))
    for split in sizes:
        text = (tmp_path / "first" / f"{split}.jsonl").read_text()
        assert '"task_id":"final-' not in text
        assert '"template_id":"heldout-' not in text
    assert verify_dataset(tmp_path / "first") == first


def test_posttraining_manifest_detects_dataset_tampering(tmp_path: Path):
    sizes = {split: 1 for split in DEFAULT_SPLIT_SIZES}
    build_dataset(tmp_path, split_sizes=sizes)
    with (tmp_path / "train.jsonl").open("a") as handle:
        handle.write("{}\n")
    try:
        verify_dataset(tmp_path)
    except RuntimeError as exc:
        assert "checksum mismatch" in str(exc)
    else:
        raise AssertionError("tampered training data should fail closed")


def test_posttraining_rare_label_balancing_is_train_only_policy():
    assert _training_repetitions("success", [0.0, 1.0]) == 19
    assert _training_repetitions("success", [1.0, 0.0]) == 1
    assert _training_repetitions("risk", [0.0, 1.0]) == 4
    assert _training_repetitions("reversible", [1.0, 0.0]) == 4
    assert _training_repetitions("needs_more_observation", [1.0, 0.0]) == 2


def test_posttraining_console_log_is_human_readable():
    row = {"event": "progress", "elapsed_seconds": 12.3, "epoch": 1, "step": 3,
           "loss": .125, "ce_loss": .25, "tokens_per_second": 456.7,
           "peak_reserved_gib": 21.5}
    assert _format_train_event(row) == (
        "[train] step=3 | epoch=1 | loss=0.125000 | ce=0.250000 | "
        "456.7 tok/s | peak_vram=21.50 GiB | elapsed=12.3s"
    )


def test_posttraining_resume_restores_rng_states_on_cpu():
    class State:
        def __init__(self):
            self.cpu_calls = 0

        def cpu(self):
            self.cpu_calls += 1
            return self

    class FakeCuda:
        def __init__(self):
            self.states = None

        def set_rng_state_all(self, states):
            self.states = states

    class FakeTorch:
        def __init__(self):
            self.cuda = FakeCuda()
            self.cpu_state = None

        def set_rng_state(self, state):
            self.cpu_state = state

    torch_module = FakeTorch()
    cpu_state = State()
    cuda_states = [State(), State()]

    _restore_rng_state(torch_module, cpu_state, cuda_states)

    assert torch_module.cpu_state is cpu_state
    assert torch_module.cuda.states == cuda_states
    assert cpu_state.cpu_calls == 1
    assert [state.cpu_calls for state in cuda_states] == [1, 1]


def test_posttraining_promotion_gate_is_fail_closed(tmp_path: Path):
    base = {"mean_absolute_error": .4, "selection_accuracy": .6, "unsafe_selections": 0}
    candidate = {"mean_absolute_error": .2, "selection_accuracy": .96, "unsafe_selections": 0}
    base_path, candidate_path = tmp_path / "base.json", tmp_path / "candidate.json"
    base_path.write_text(__import__("json").dumps(base))
    candidate_path.write_text(__import__("json").dumps(candidate))
    assert promotion_gate(base_path, candidate_path)["passed"] is True
    candidate["selection_accuracy"] = .94
    candidate_path.write_text(__import__("json").dumps(candidate))
    failed = promotion_gate(base_path, candidate_path)
    assert failed["passed"] is False
    assert failed["checks"]["development_selection_at_least_95pct"] is False


def test_latency_telemetry_persists_episode_and_step_provenance(tmp_path: Path):
    cache_path = tmp_path / "candidate-cache.json"
    db_path = tmp_path / "telemetry.sqlite"
    planner = CachedPlanner(DeterministicPlanner(), cache_path)
    store = TrajectoryStore(db_path, tmp_path / "logs")
    store.start_run("fresh", {"mode": "candidate"})
    fresh = asyncio.run(run_episode(run_id="fresh", task_id="voltage-001",
        environment=SandboxWebEnvironment(), planner=planner, predictor=HeuristicPredictor(),
        policy=GreedyUtilityPolicy(), store=store))
    store.finish_run("fresh", "completed", fresh)
    assert fresh["wall_clock_ms"] > 0
    assert fresh["candidate_fresh_count"] == fresh["steps"]
    assert fresh["candidate_cache_hit_count"] == 0
    assert fresh["effective_end_to_end_complete"] is True
    assert fresh["effective_end_to_end_ms_reconstructed"] == fresh["wall_clock_ms"]
    assert fresh["predictor_wall_ms"] >= fresh["predictor_reported_ms"]
    assert fresh["policy_ms"] >= 0 and fresh["environment_ms"] >= 0
    assert fresh["selector_wall_ms"] == fresh["predictor_wall_ms"] + fresh["policy_ms"]
    assert fresh["framework_overhead_ms"] >= 0
    steps = store.reconstruct("fresh", "voltage-001")
    assert all(step["timings"]["candidate_source"] == "fresh" for step in steps)
    assert all(step["timings"]["candidate_generation_ms"] >= 0 for step in steps)
    summary = __import__("json").loads(store.conn.execute(
        "SELECT summary_json FROM runs WHERE run_id='fresh'").fetchone()[0])
    assert summary["wall_clock_ms"] == fresh["wall_clock_ms"]
    store.close()

    replay_planner = CachedPlanner(DeterministicPlanner(), cache_path)
    replay_store = TrajectoryStore(tmp_path / "replay.sqlite", tmp_path / "replay-logs")
    replay_store.start_run("replay", {"mode": "candidate"})
    replay = asyncio.run(run_episode(run_id="replay", task_id="voltage-001",
        environment=SandboxWebEnvironment(), planner=replay_planner, predictor=HeuristicPredictor(),
        policy=GreedyUtilityPolicy(), store=replay_store))
    replay_store.finish_run("replay", "completed", replay)
    replay_steps = replay_store.reconstruct("replay", "voltage-001")
    replay_store.close()
    assert replay["candidate_fresh_count"] == 0
    assert replay["candidate_cache_hit_count"] == replay["steps"]
    assert all(step["timings"]["candidate_source"] == "cache" for step in replay_steps)
    assert all(step["timings"]["candidate_generation_ms"] == 0 for step in replay_steps)
    assert replay["effective_end_to_end_complete"] is True
    assert replay["effective_end_to_end_ms_reconstructed"] >= replay["wall_clock_ms"]


def test_cache_generation_latency_survives_serialization_and_legacy_is_incomplete(tmp_path: Path):
    state = SandboxWebEnvironment().reset("voltage-001")
    path = tmp_path / "cache.json"
    fresh = CachedPlanner(DeterministicPlanner(), path)
    asyncio.run(fresh.propose_actions(state))
    assert fresh.last_provenance["candidate_source"] == "fresh"
    generated = fresh.last_provenance["candidate_generation_ms"]
    replay = CachedPlanner(DeterministicPlanner(), path)
    asyncio.run(replay.propose_actions(state))
    assert replay.last_provenance["candidate_source"] == "cache"
    assert replay.last_provenance["candidate_generation_ms"] == 0
    assert replay.last_provenance["replayed_candidate_generation_ms"] == generated
    assert replay.last_provenance["effective_latency_complete"] is True

    payload = __import__("json").loads(path.read_text())
    entry = next(iter(payload["entries"].values()))
    entry["metadata"].pop("latency_ms")
    path.write_text(__import__("json").dumps(payload))
    incomplete = CachedPlanner(DeterministicPlanner(), path)
    asyncio.run(incomplete.propose_actions(state))
    assert incomplete.last_provenance["candidate_source"] == "cache"
    assert incomplete.last_provenance["replayed_candidate_generation_ms"] is None
    assert incomplete.last_provenance["effective_latency_complete"] is False


def test_paired_latency_ratios_use_only_comparable_successes():
    rows = []
    def add(mode, task, success, wall, effective=None, complete=False):
        rows.append({"mode": mode, "task_id": task, "seed": 0, "success": success,
                     "steps": 2, "wall_clock_ms": wall,
                     "effective_end_to_end_ms_reconstructed": effective,
                     "effective_end_to_end_complete": complete})
    add("direct_gpt", "ok", True, 100)
    add("candidates_base_laya", "ok", True, 50, 120, True)
    add("direct_gpt", "failed", True, 100)
    add("candidates_base_laya", "failed", False, 20, 80, True)
    add("direct_gpt", "zero", True, 0)
    add("candidates_base_laya", "zero", True, 10, 20, True)
    comparison = _paired_comparisons(rows)["candidates_base_laya_vs_direct_gpt"]
    assert comparison["pairs"] == 3
    assert comparison["pairs_both_success"] == 2
    assert comparison["pairs_valid_time_to_success"] == 1
    assert comparison["median_latency_ratio"] == .5
    assert comparison["latency_ratio_distribution"]["0.25-0.5x"] == 1
    assert comparison["all_terminal_outcomes"]["pairs"] == 3
    assert comparison["effective_end_to_end"]["pairs_complete"] == 1
    assert comparison["effective_end_to_end"]["median_latency_ratio"] == 1.2


def test_predictor_wall_and_reported_latency_are_distinct(tmp_path: Path):
    class ReportedPredictor:
        def predict(self, state, actions):
            predictions = HeuristicPredictor().predict(state, actions)
            return {key: value.model_copy(update={"latency_ms": 100.0})
                    for key, value in predictions.items()}

    store = TrajectoryStore(tmp_path / "predictor.sqlite", tmp_path / "predictor-logs")
    store.start_run("predictor", {"test": True})
    result = asyncio.run(run_episode(run_id="predictor", task_id="voltage-001",
        environment=SandboxWebEnvironment(), planner=DeterministicPlanner(),
        predictor=ReportedPredictor(), policy=GreedyUtilityPolicy(), store=store))
    store.finish_run("predictor", "completed", result)
    store.close()
    assert result["predictor_reported_ms"] == 600.0
    assert result["predictor_wall_ms"] != result["predictor_reported_ms"]
    event_text = (tmp_path / "predictor-logs" / "predictor" / "events.jsonl").read_text()
    assert '"wall_clock_ms"' in event_text


def test_tracked_files_do_not_expose_workstation_identifiers():
    import subprocess
    root = Path(__file__).resolve().parents[1]
    tracked = subprocess.run(["git", "ls-files", "-z"], cwd=root, check=True,
                             capture_output=True).stdout.split(b"\0")
    local_user = b"kwe" + b"stog"
    forbidden = (b"/home/" + local_user, b"/mnt/" + b"fast-ssd",
                 b"nebius-" + b"pilot", b"89.169." + b"109.237", local_user + b"@")
    leaks = []
    for raw_path in tracked:
        if not raw_path:
            continue
        path = root / raw_path.decode()
        if not path.is_file():
            continue
        content = path.read_bytes()
        for token in forbidden:
            if token in content:
                leaks.append(f"{raw_path.decode()}: {token.decode()}")
    assert leaks == []
