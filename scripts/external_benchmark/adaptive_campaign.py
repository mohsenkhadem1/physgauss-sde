"""Refine all SoftVTBench release predictions by a fixed numerical policy."""
import sys,json,time,subprocess,fcntl,argparse
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from scripts.external_benchmark.common import *
p=argparse.ArgumentParser();p.add_argument('--worker',required=True);a=p.parse_args()
queue=BASE/'adaptive-jobs';queue.mkdir(exist_ok=True);jobs=json.loads((REPORT/'frozen-protocol.json').read_text())['jobs']
indices=[i for i,j in enumerate(jobs) if 'softvt' in j['command'] and j['command'][1].endswith('/fit.py')]
while True:
    check_deadline();claim=None;terminal=0
    with (queue/'lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        for index in indices:
            status=queue/f'{index:03}.json'
            if status.exists():
                if json.loads(status.read_text())['state'] in ['complete','failed','unavailable']:terminal+=1
                continue
            original=BASE/'primary-jobs'/f'{index:03}.json'
            if not original.exists():continue
            original=json.loads(original.read_text())
            if original['state']!='complete':
                if original['state']=='failed' and (original.get('repair_attempted') or (BASE/'repairs-complete.json').exists()):
                    write_json(status,dict(index=index,state='unavailable',reason='Original fitting repair failed'));terminal+=1
                continue
            cmd=jobs[index]['command'];value=lambda flag:cmd[cmd.index(flag)+1]
            command=[str(ROOT/'runs/phystwin-env/bin/python'),'scripts/external_benchmark/adaptive_prediction.py','--subset',value('--subset'),'--task',value('--task'),'--episode',value('--episode'),'--source-tag',value('--tag')]
            state=dict(index=index,state='running',worker=a.worker,command=command);write_json(status,state);claim=(status,state,command);break
    if claim is None:
        if terminal==len(indices):
            write_json(BASE/'adaptive-complete.json',dict(completed_utc=datetime.now(timezone.utc).isoformat()));break
        time.sleep(15);continue
    status,state,command=claim;log=queue/f'{state["index"]:03}.log';start=time.time()
    try:
        with log.open('w') as output:r=subprocess.run(command,cwd=ROOT,stdout=output,stderr=subprocess.STDOUT,timeout=1800)
        state.update(state='complete' if r.returncode==0 else 'failed',returncode=r.returncode)
    except subprocess.TimeoutExpired:state.update(state='failed',reason='Prediction refinement timeout')
    state.update(elapsed_seconds=time.time()-start,log=str(log));write_json(status,state)
