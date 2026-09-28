from __future__ import annotations
import argparse, asyncio, importlib.util, json, os, platform, shutil, sqlite3, subprocess, sys, uuid
from pathlib import Path
from .planners import DeterministicPlanner
from .policies import GPTOnlyPolicy, GreedyUtilityPolicy, UtilityWeights
from .predictors import HeuristicPredictor, LayaPredictor
from .runner_impl import run_episode
from .sandbox import SandboxWebEnvironment
from .storage_v2 import TrajectoryStore

def doctor() -> int:
    checks={"python":sys.version.split()[0],"platform":platform.platform(),"sqlite":sqlite3.sqlite_version,"disk_free_gb":round(shutil.disk_usage(".").free/2**30,2),"openai_configured":bool(os.getenv("OPENAI_API_KEY")),"laya_import":importlib.util.find_spec("laya") is not None,"playwright_import":importlib.util.find_spec("playwright") is not None}
    try: checks["nvidia_smi"]=subprocess.run(["nvidia-smi","--query-gpu=name,memory.total","--format=csv,noheader"],capture_output=True,text=True,timeout=5).stdout.strip() or "unavailable"
    except (FileNotFoundError,subprocess.SubprocessError): checks["nvidia_smi"]="unavailable"
    try:
        import torch
        checks.update({"torch":torch.__version__,"torch_cuda":torch.cuda.is_available(),"cuda_version":torch.version.cuda})
    except ImportError: checks["torch"]="not installed"
    print(json.dumps(checks,indent=2));return 0

async def execute(mode:str,report:bool=False)->dict:
    run_id=f"{mode}-{uuid.uuid4().hex[:8]}";store=TrajectoryStore(Path("data/trajectories.sqlite3"),Path("logs/runs"));store.start_run(run_id,{"mode":mode,"planner":"deterministic-fixture","candidate_count":5,"weights":UtilityWeights().__dict__})
    predictor=None;policy=GPTOnlyPolicy()
    if mode=="heuristic":predictor,policy=HeuristicPredictor(),GreedyUtilityPolicy()
    elif mode=="base_laya":predictor,policy=LayaPredictor(os.getenv("LAYA_CHECKPOINT","laya")),GreedyUtilityPolicy()
    result=await run_episode(run_id=run_id,task_id="voltage-001",environment=SandboxWebEnvironment(),planner=DeterministicPlanner(),predictor=predictor,policy=policy,store=store);result.update(run_id=run_id,mode=mode)
    if report:
        root=Path("reports")/run_id;root.mkdir(parents=True,exist_ok=True);(root/"metrics.json").write_text(json.dumps(result,indent=2)+"\n");(root/"summary.md").write_text("# Smoke benchmark\n\nEngineering smoke test only; deterministic fixture planner, no scientific claim.\n\n```json\n"+json.dumps(result,indent=2)+"\n```\n")
    return result

async def benchmark(include_laya:bool)->list[dict]:
    results=[await execute("gpt_only",True),await execute("heuristic",True)]
    if include_laya:results.append(await execute("base_laya",True))
    root=Path("reports/latest");root.mkdir(parents=True,exist_ok=True);(root/"metrics.json").write_text(json.dumps(results,indent=2)+"\n");lines=["# Smoke benchmark","","This validates plumbing only; the planner is a deterministic fixture.","","| mode | success | steps | stop |","|---|---:|---:|---|"]+[f"| {r['mode']} | {r['success']} | {r['steps']} | {r['stop_reason']} |" for r in results];(root/"summary.md").write_text("\n".join(lines)+"\n");return results

def main()->None:
    parser=argparse.ArgumentParser(prog="lda");sub=parser.add_subparsers(dest="command",required=True);sub.add_parser("doctor");sub.add_parser("demo");bench=sub.add_parser("benchmark");bench.add_argument("--suite",default="smoke",choices=["smoke"]);bench.add_argument("--with-laya",action="store_true");args=parser.parse_args()
    if args.command=="doctor":raise SystemExit(doctor())
    result=asyncio.run(execute("heuristic") if args.command=="demo" else benchmark(args.with_laya));print(json.dumps(result,indent=2))

