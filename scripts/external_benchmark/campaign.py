"""Resumable shared job queue. Run one worker per authorized GPU, after freeze."""
import argparse,fcntl,json,os,subprocess,sys,time
from datetime import datetime,timezone
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from scripts.external_benchmark.common import ROOT,BASE,REPORT,CONFIG,check_deadline,write_json

def commands():
    py=str(ROOT/'runs/phystwin-env/bin/python');jobs=[]
    for case in ['double_stretch_sloth','single_lift_dinosor','double_stretch_zebra']:
        for ablate in [False,True]:
            tag='primary-no-field' if ablate else 'primary'
            cmd=[py,'scripts/external_benchmark/fit.py','--dataset','phystwin','--case',case,'--nodes','300','--iterations','100','--frames','36','--contact','--nonconvex','--materials','6','--tag',tag]
            if ablate:cmd+=['--no-field']
            jobs.append(dict(name=f'{tag}-{case}',command=cmd))
    for episode in range(10):
        for subset in ['object-soft','spatial-soft']:
            for task in [1,2]:
                inputs=['--subset',subset,'--task',str(task),'--episode',str(episode)]
                name=f'{subset}-task{task}-demo{episode}'
                for ablate in [False,True]:
                    tag='primary-no-field' if ablate else 'primary'
                    cmd=[py,'scripts/external_benchmark/fit.py','--dataset','softvt',*inputs,'--iterations','100','--frames','36','--contact','--materials','1','--tag',tag]
                    if ablate:cmd+=['--no-field']
                    jobs.append(dict(name=f'{tag}-{name}',command=cmd))
                jobs.append(dict(name=f'native-{name}',command=[py,'scripts/external_benchmark/native_softvt.py',*inputs,'--epochs','100','--initial','30000','--tag','primary']))
    return jobs

def main():
    p=argparse.ArgumentParser();p.add_argument('--worker',required=True);a=p.parse_args()
    manifest=REPORT/'frozen-protocol.json'
    if not manifest.exists():raise RuntimeError('Freeze protocol before running reserved cases')
    jobs=json.loads(manifest.read_text())['jobs'];queue=BASE/'primary-jobs';queue.mkdir(exist_ok=True,parents=True)
    while True:
        check_deadline();job=None
        with (queue/'lock').open('w') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            for index,item in enumerate(jobs):
                status=queue/f'{index:03}.json'
                if status.exists():continue
                job=item;state=dict(index=index,worker=a.worker,name=job['name'],command=job['command'],state='running',started_utc=datetime.now(timezone.utc).isoformat())
                write_json(status,state);break
        if job is None:break
        log=queue/f'{index:03}.log';print('START',index,job['name'],flush=True)
        deadline=datetime.fromisoformat(json.loads(CONFIG.read_text())['deadline_utc'].replace('Z','+00:00'))
        timeout=min(1800,max(1,(deadline-datetime.now(timezone.utc)).total_seconds()))
        started=time.time()
        try:
            with log.open('w') as output:
                run=subprocess.run(job['command'],cwd=ROOT,stdout=output,stderr=subprocess.STDOUT,timeout=timeout)
            state.update(state='complete' if run.returncode==0 else 'failed',returncode=run.returncode)
        except subprocess.TimeoutExpired:state.update(state='failed',error='Per-job or campaign time limit')
        state.update(elapsed_seconds=time.time()-started,finished_utc=datetime.now(timezone.utc).isoformat(),log=str(log))
        write_json(status,state);print('DONE',index,state['state'],round(state['elapsed_seconds'],1),flush=True)
    print('QUEUE FINISHED',flush=True)

if __name__=='__main__':main()
