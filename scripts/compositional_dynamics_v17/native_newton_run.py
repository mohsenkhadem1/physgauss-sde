"""Bounded native Newton development jobs; every declared case retains terminal status."""
import os,json,hashlib,subprocess,time,signal,concurrent.futures
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];REPORT=ROOT/'reports/compositional-dynamics-v17';BASE=ROOT/'runs/compositional-dynamics-v17/native-newton-dynamics';PLAN=REPORT/'native-newton-dynamics-plan.json'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,d):tmp=p.with_suffix('.pending.json');tmp.write_text(json.dumps(d,indent=2)+'\n');tmp.replace(p)
def kill_group(proc):
    try:os.killpg(proc.pid,signal.SIGTERM)
    except ProcessLookupError:pass
    try:proc.wait(timeout=5)
    except subprocess.TimeoutExpired:pass
    try:os.killpg(proc.pid,signal.SIGKILL)
    except ProcessLookupError:pass
    proc.wait()
def main():
    plan=json.loads(PLAN.read_text());plan_sha=sha(PLAN);review=json.loads((REPORT/'native-newton-dynamics-prefit-independent-review.json').read_text())
    if review.get('passed')is not True or review.get('plan_sha256')!=plan_sha:raise ValueError('Exact source/protocolreview required')
    for path,h in plan['source_sha256'].items():
        if sha(path)!=h:raise ValueError('Source changed')
    BASE.mkdir(exist_ok=True,parents=True);deadline=time.time()+plan['max_stage_seconds']
    with (BASE/'run-start.json').open('x')as f:json.dump(dict(plan_sha256=plan_sha,deadline_epoch=deadline,workers=2,outer_service='RuntimeMaxSec3700 KillModecontrol-group required',source_sha256=sha(__file__)),f)
    def worker(job):
        remaining=deadline-time.time();out=BASE/job['name'];record=dict(job=job,plan_sha256=plan_sha,state='not_started')
        if remaining<=0:return dict(record,error='Stage deadline before dispatch')
        if out.exists():return dict(record,error='Existing output preserved; no silent rerun')
        log=BASE/(job['name']+'.log');t=time.monotonic()
        command=[plan['runtime'],'-m','scripts.compositional_dynamics_v17.native_newton_dynamics','run','--name',job['name'],'--device','cpu']
        with log.open('w')as f:
            proc=subprocess.Popen(command,cwd=ROOT,env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1'),stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
            try:code=proc.wait(timeout=min(plan['max_job_seconds']+30,remaining));record.update(state='complete'if code==0 else'failed',returncode=code)
            except subprocess.TimeoutExpired:kill_group(proc);record.update(state='timeout',error='External job/stage processgroup timeout')
        record['wall_seconds']=time.monotonic()-t
        for name in ['complete.json','progress.json','trajectory.npz']:
            path=out/name
            if path.exists():record[name+'_sha256']=sha(path)
        if (out/'progress.json').exists():record['last_preserved_time_s']=json.loads((out/'progress.json').read_text())['last_saved_time_s']
        return record
    rows=[]
    with concurrent.futures.ThreadPoolExecutor(max_workers=2)as pool:
        futures={pool.submit(worker,j):j for j in plan['jobs']}
        for f in concurrent.futures.as_completed(futures):
            try:row=f.result()
            except Exception as e:row=dict(job=futures[f],plan_sha256=plan_sha,state='failed',error=repr(e))
            rows.append(row);write(BASE/'terminal-progress.json',dict(plan_sha256=plan_sha,rows=rows))
    rows.sort(key=lambda r:next(i for i,j in enumerate(plan['jobs'])if j['name']==r['job']['name']))
    write(BASE/'terminal.json',dict(plan_sha256=plan_sha,rows=rows,expected=len(plan['jobs']),complete=sum(r['state']=='complete'for r in rows),failed=sum(r['state']!='complete'for r in rows),source_sha256=sha(__file__)))
if __name__=='__main__':main()
