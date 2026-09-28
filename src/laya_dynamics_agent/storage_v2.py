from __future__ import annotations

import json, sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCHEMA = """CREATE TABLE IF NOT EXISTS runs(run_id TEXT PRIMARY KEY,config_json TEXT NOT NULL,started_at TEXT NOT NULL); CREATE TABLE IF NOT EXISTS steps(run_id TEXT NOT NULL,task_id TEXT NOT NULL,step INTEGER NOT NULL,state_before TEXT NOT NULL,candidates TEXT NOT NULL,predictions TEXT NOT NULL,chosen_action TEXT NOT NULL,state_after TEXT NOT NULL,observed TEXT NOT NULL,reward REAL NOT NULL,timings TEXT NOT NULL,error TEXT,PRIMARY KEY(run_id,task_id,step));"""

def plain(x: Any) -> Any:
    if hasattr(x, "model_dump"): return x.model_dump(mode="json")
    if isinstance(x, dict): return {k: plain(v) for k,v in x.items()}
    if isinstance(x, (list,tuple)): return [plain(v) for v in x]
    return x

class TrajectoryStore:
    def __init__(self, db_path: Path, log_root: Path):
        db_path.parent.mkdir(parents=True,exist_ok=True); self.conn=sqlite3.connect(db_path); self.conn.executescript(SCHEMA); self.log_root=log_root
        columns={row[1] for row in self.conn.execute("PRAGMA table_info(runs)")}
        if "ended_at" not in columns: self.conn.execute("ALTER TABLE runs ADD COLUMN ended_at TEXT")
        if "status" not in columns: self.conn.execute("ALTER TABLE runs ADD COLUMN status TEXT NOT NULL DEFAULT 'running'")
        self.conn.commit()
    def start_run(self,run_id:str,config:dict[str,Any]):
        now=datetime.now(timezone.utc).isoformat(); self.conn.execute("INSERT INTO runs(run_id,config_json,started_at,status) VALUES(?,?,?,'running')",(run_id,json.dumps(config,sort_keys=True),now)); self.conn.commit(); self.event(run_id,"-",0,"run_started",{"config":config})
    def finish_run(self,run_id:str,status:str,summary:dict[str,Any]):
        now=datetime.now(timezone.utc).isoformat(); self.conn.execute("UPDATE runs SET ended_at=?,status=? WHERE run_id=?",(now,status,run_id));self.conn.commit();self.event(run_id,summary.get("task_id","-"),summary.get("steps",0),"run_interrupted" if status=="interrupted" else "run_finished",{"status":status,"summary":summary})
    def close(self):
        self.conn.commit();self.conn.close()
    def save_step(self,run_id:str,task_id:str,step:int,**data:Any):
        enc=lambda x:json.dumps(plain(x),sort_keys=True)
        self.conn.execute("INSERT INTO steps VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",(run_id,task_id,step,enc(data["state_before"]),enc(data["candidates"]),enc(data["predictions"]),enc(data["chosen_action"]),enc(data["state_after"]),enc(data["observed"]),data["reward"],enc(data.get("timings",{})),data.get("error")));self.conn.commit();self.event(run_id,task_id,step,"transition",plain(data))
    def event(self,run_id,task_id,step,event_type,payload):
        path=self.log_root/run_id/"events.jsonl";path.parent.mkdir(parents=True,exist_ok=True);record={"timestamp":datetime.now(timezone.utc).isoformat(),"run_id":run_id,"task_id":task_id,"step":step,"event_type":event_type,"payload":payload}
        with path.open("a",encoding="utf-8") as fh: fh.write(json.dumps(record,sort_keys=True)+"\n")
    def reconstruct(self,run_id,task_id):
        rows=self.conn.execute("SELECT step,state_before,candidates,predictions,chosen_action,state_after,observed,reward,timings,error FROM steps WHERE run_id=? AND task_id=? ORDER BY step",(run_id,task_id));keys=("step","state_before","candidates","predictions","chosen_action","state_after","observed","reward","timings","error");result=[]
        for row in rows:
            item=dict(zip(keys,row))
            for key in ("state_before","candidates","predictions","chosen_action","state_after","observed","timings"):item[key]=json.loads(item[key])
            result.append(item)
        return result

