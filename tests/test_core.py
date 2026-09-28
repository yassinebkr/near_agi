import asyncio
from pathlib import Path
from laya_dynamics_agent.models import CandidateAction
from laya_dynamics_agent.planners import DeterministicPlanner
from laya_dynamics_agent.policies import GPTOnlyPolicy, GreedyUtilityPolicy
from laya_dynamics_agent.predictors import HeuristicPredictor, LayaPredictor
from laya_dynamics_agent.runner_impl import run_episode
from laya_dynamics_agent.sandbox import SandboxWebEnvironment
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

