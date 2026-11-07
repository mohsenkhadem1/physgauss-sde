"""Execute every frozen drop case; preserve failures, respect campaign deadline."""
from pathlib import Path
import datetime,json,os,subprocess,sys,time
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT));from scripts.gap_first_v01.drop_protocol import verify
BASE=ROOT/'runs/gap-first-v01/drop';REPORT=ROOT/'reports/gap-first-v01/drop';PY=BASE/'env/bin/python'
def save(d):
 d['updated_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat();p=REPORT/'supervisor.json';tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(d,indent=2));tmp.replace(p)
def main():
 p=verify();deadline=datetime.datetime.fromisoformat(p['deadline_utc']).timestamp();state=dict(pid=os.getpid(),status='running',completed=[],failures=[],current=None);env=dict(os.environ,CUDA_VISIBLE_DEVICES='0',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='4')
 for asset in p['assets']:
  for case in p['cases']:
   verify();key=asset+'/'+case;result=BASE/'cases'/asset/case/'result.json';state['current']=key;save(state)
   if time.time()>=deadline:state['status']='deadline_incomplete';save(state);return
   if result.exists():
    if json.loads(result.read_text()).get('complete'):state['completed'].append(key);continue
    state['failures'].append(dict(case=key,reason='retained previous incomplete case'));continue
   logpath=BASE/'logs'/f'{asset}-{case}.log';logpath.parent.mkdir(exist_ok=True)
   with logpath.open('a') as log:
    child=subprocess.Popen([str(PY),'scripts/gap_first_v01/drop_run.py','--asset',asset,'--case',case],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT);state['worker_pid']=child.pid;save(state)
    while child.poll() is None:
     if time.time()>=deadline:child.terminate();child.wait(timeout=30);break
     time.sleep(2)
   if child.returncode==0 and result.exists() and json.loads(result.read_text()).get('complete'):state['completed'].append(key)
   else:state['failures'].append(dict(case=key,returncode=child.returncode,log=str(logpath.relative_to(ROOT))))
   save(state)
 state['status']='cases_complete' if not state['failures'] else 'cases_finished_with_failures';state['current']=None;save(state)
 for script in ['drop_analysis.py','drop_native_metrics.py','drop_visualize.py']:
  with (BASE/(script+'.log')).open('w') as log:q=subprocess.run([str(PY),'scripts/gap_first_v01/'+script],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
  state.setdefault('postprocess',{})[script]=q.returncode;save(state)
 state['status']='complete' if not state['failures'] and all(v==0 for v in state['postprocess'].values()) else 'finished_with_failures';save(state)
if __name__=='__main__':main()
