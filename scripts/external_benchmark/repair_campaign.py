"""Apply the declared prefix-only feasibility repair with a locked shared queue."""
import sys,time,json,subprocess,fcntl,argparse
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from scripts.external_benchmark.common import *
p=argparse.ArgumentParser();p.add_argument('--worker',default='repair');a=p.parse_args()
queue=BASE/'primary-jobs';count=len(json.loads((REPORT/'frozen-protocol.json').read_text())['jobs'])
archive=BASE/'original-failed-jobs';archive.mkdir(exist_ok=True)
while True:
    check_deadline();claim=None
    with (queue/'repair.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        states=[(p,json.loads(p.read_text())) for p in sorted(queue.glob('*.json'))]
        for path,state in states:
            if state['state']!='failed' or state.get('repair_attempted'):continue
            if 'No valid fitting iterate' not in Path(state['log']).read_text() or 'softvt' not in state['command']:continue
            if not (archive/path.name).exists():write_json(archive/path.name,state)
            state.update(state='repair_running',repair_attempted=True,repair_worker=a.worker,original_failure=dict(returncode=state.get('returncode'),log=state['log']))
            command=state['command'].copy();command[1]='scripts/external_benchmark/retry_fit.py';command+=['--device','cpu']
            state['repair_command']=command;write_json(path,state);claim=(path,state,command);break
    if claim is None:
        if len(states)==count and all(s['state'] in ['complete','failed'] for _,s in states):
            write_json(BASE/'repairs-complete.json',dict(completed_utc=datetime.now(timezone.utc).isoformat()));break
        time.sleep(15);continue
    path,state,command=claim;repair_log=queue/f'{state["index"]:03}-repair.log';started=time.time();print('REPAIR',state['index'],flush=True)
    try:
        with repair_log.open('w') as output:result=subprocess.run(command,cwd=ROOT,stdout=output,stderr=subprocess.STDOUT,timeout=1800)
        state.update(state='complete' if result.returncode==0 else 'failed',returncode=result.returncode)
    except subprocess.TimeoutExpired:state.update(state='failed',error='Repair timeout')
    state.update(repair_elapsed_seconds=time.time()-started,repair_log=str(repair_log));write_json(path,state)
    print('REPAIR DONE',state['index'],state['state'],flush=True)
