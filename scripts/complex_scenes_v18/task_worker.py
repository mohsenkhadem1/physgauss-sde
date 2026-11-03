"""Crash-recoverable wrapper for one declared campaign task; no shell execution."""
import argparse,hashlib,json,os,subprocess,time
from pathlib import Path
from scripts.complex_scenes_v18.run_development import ROOT,write


def key(task_id):return hashlib.sha256(task_id.encode()).hexdigest()[:20]


def start_ticks(pid):
    try:return (Path('/proc')/str(pid)/'stat').read_text().split(') ',1)[1].split()[19]
    except (OSError,IndexError):return None


def alive(pid,ticks):
    if start_ticks(pid)!=ticks:return False
    try:return (Path('/proc')/str(pid)/'stat').read_text().split(') ',1)[1].split()[0]!='Z'
    except OSError:return False


def run(plan_path,task_id,state_dir):
    plan_path=Path(plan_path);state_dir=Path(state_dir);raw=plan_path.read_bytes();plan=json.loads(raw)
    matches=[t for t in plan['tasks']if t['id']==task_id]
    if len(matches)!=1:raise ValueError('Unique declared task required')
    task=matches[0];status_path=state_dir/(key(task_id)+'.json')
    # Exclusive claim prevents two supervisors from executing the same task.
    claim=state_dir/(key(task_id)+'.claim');claim.parent.mkdir(parents=True,exist_ok=True)
    with claim.open('x')as f:f.write(str(os.getpid()))
    status=dict(task_id=task_id,state='running',pid=os.getpid(),start_ticks=start_ticks(os.getpid()),plan_sha256=hashlib.sha256(raw).hexdigest(),started_at=time.time(),argv=task['argv'])
    write(status_path,status)
    try:
        child=subprocess.Popen(task['argv'],cwd=ROOT,env=os.environ.copy())
        status['child_pid']=child.pid;status['child_start_ticks']=start_ticks(child.pid);write(status_path,status)
        code=child.wait();status.update(state='complete'if code==0 else'failed',exit_code=code)
    except Exception as e:status.update(state='failed',error=f'{type(e).__name__}: {e}')
    status['finished_at']=time.time();write(status_path,status);return status


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--plan',required=True);p.add_argument('--task-id',required=True);p.add_argument('--state-dir',required=True);a=p.parse_args();s=run(a.plan,a.task_id,a.state_dir);raise SystemExit(0 if s['state']=='complete'else 1)
