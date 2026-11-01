"""Separate pinned n6/q4 development quadrature check; no test generation."""
from pathlib import Path
import sys,json
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import numpy as np
from scripts.physical_refinement_v15 import development_mechanics_queue as worker
from scripts.physical_refinement_v15.mechanics_adapter import specification
from scripts.physical_refinement_v15.mechanics_pilot import BASE,REPORT,write,sha
from scripts.physical_refinement_v15.study import base_cases,development_actions
OUT=BASE/'development/quadrature-queue';MANIFEST=OUT/'manifest.json'
def prepare():
 poses=np.asarray(list(development_actions().values()));truth=REPORT/'development-truth-plan.json';plan=json.loads(truth.read_text());items=[]
 for c in base_cases():
  if abs(c['material_gradient'])==.8:items.append(dict(name=c['name'],case={k:v for k,v in c.items() if k!='name'},root=BASE/'development/hypotheses'/c['name']))
 for c in plan['scenes']:items.append(dict(name=c['name'],case=c['case'],root=BASE/'development/worlds'/c['name']))
 sources={str(truth.relative_to(ROOT)):sha(truth)};jobs=[]
 for item in items:
  spec=specification(item['case'],poses,6,4);sources.update(spec['source_sha256']);p=OUT/'specs'/f"{item['name']}.json";write(p,dict(case=item['case'],poses=poses.tolist()));baseline=item['root']/'n6';record=json.loads((baseline/'complete.json').read_text());sources[str((baseline/'complete.json').relative_to(ROOT))]=sha(baseline/'complete.json')
  jobs.append(dict(id=f"{item['name']}-n6-q4",name=item['name'],n=6,order=4,spec=str(p.relative_to(ROOT)),out=str((item['root']/'n6-q4').relative_to(ROOT)),baseline=str(baseline.relative_to(ROOT)),budget_seconds=600))
 for p in (Path(__file__),ROOT/'scripts/physical_refinement_v15/study.py',ROOT/'scripts/physical_refinement_v15/development_mechanics_queue.py'):sources[str(p.relative_to(ROOT))]=sha(p)
 m=dict(scope='Development quadrature qualification only:8corners+4offgridworlds×8poses; no new hypotheses or held-out data',jobs=jobs,action_names=list(development_actions()),source_sha256=sources,spec_sha256={j['spec']:sha(ROOT/j['spec']) for j in jobs},gates=worker.GATES,workers=3,per_job_seconds=600,stage_seconds=1800)
 if MANIFEST.exists():
  if json.loads(MANIFEST.read_text())!=m:raise ValueError('Changed quadrature manifest')
 else:write(MANIFEST,m)
 return m

def compare(job,m):
 rows=[]
 for i,action in enumerate(m['action_names']):
  dirs=[ROOT/job[key]/f'pose-{i:03d}' for key in ('baseline','out')]
  if not all((p/'complete.json').exists() for p in dirs):rows.append(dict(action=action,passed=False,reason='Missing completed pose'));continue
  for p in dirs:
   record=json.loads((p/'complete.json').read_text())
   if any(sha(p/k)!=h for k,h in record['artifact_sha256'].items()):raise ValueError('Changed completed state')
  x,y=[np.load(p/'private-state.npz') for p in dirs];sx,sy=[np.load(p/'surface.npz')['vertices'] for p in dirs];wx,wy=[np.load(p/'reaction-wrench.npz') for p in dirs];d=x['field']-y['field'];s=sx-sy;f=np.linalg.norm(wx['forces_N']-wy['forces_N'],axis=1);t=np.linalg.norm(wx['torques_Nm']-wy['torques_Nm'],axis=1)
  metrics=dict(field_rms_m=float(np.sqrt(np.mean(d*d))),field_max_m=float(abs(d).max()),surface_rms_m=float(np.sqrt(np.mean(s*s))),surface_max_m=float(abs(s).max()),force_difference_N=f.tolist(),torque_difference_Nm=t.tolist());passed=all(metrics[k]<=m['gates'][k] for k in ('field_rms_m','field_max_m','surface_rms_m','surface_max_m')) and bool((f<=1e-4+.005*np.linalg.norm(wy['forces_N'],axis=1)).all() and (t<=1e-5+.005*np.linalg.norm(wy['torques_Nm'],axis=1)).all());rows.append(dict(action=action,passed=passed,**metrics))
 return dict(name=job['name'],quadrature_pair=[3,4],mesh=6,passed=all(r['passed'] for r in rows),rows=rows)

def phase(m):
 # Isolated launcher, same bounded-worker contract; explicit --order4.
 import concurrent.futures,queue,subprocess,os,signal,time
 pending=queue.Queue()
 for job in m['jobs']:pending.put(job)
 def run(index):
  while True:
   try:j=pending.get_nowait()
   except queue.Empty:return
   worker.guard(m);status=OUT/'jobs'/f"{j['id']}.json";log=OUT/'logs'/f"{j['id']}.log";log.parent.mkdir(parents=True,exist_ok=True);start=time.monotonic();write(status,dict(state='running',job=j,device=f'cuda:{index}',manifest_sha256=sha(MANIFEST)))
   args=[sys.executable,'-B',str(ROOT/'scripts/physical_refinement_v15/mechanics_adapter.py'),'--spec',str(ROOT/j['spec']),'--out',str(ROOT/j['out']),'--n','6','--order','4','--device',f'cuda:{index}','--budget-seconds','600'];env=os.environ.copy();env.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1');timeout=False
   with log.open('a') as stream:
    p=subprocess.Popen(args,stdout=stream,stderr=subprocess.STDOUT,env=env,start_new_session=True)
    try:code=p.wait(timeout=610)
    except subprocess.TimeoutExpired:timeout=True;os.killpg(p.pid,signal.SIGKILL);code=p.wait()
   worker.guard(m);complete=ROOT/j['out']/'complete.json';state='complete' if code==0 and complete.exists() else 'failed';write(status,dict(state=state,job=j,exit_code=code,timed_out=timeout,elapsed_seconds=time.monotonic()-start,manifest_sha256=sha(MANIFEST),complete_sha256=sha(complete) if complete.exists() else None));print(j['id'],state,flush=True)
 with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
  for task in [pool.submit(run,i) for i in range(3)]:task.result()

def main():
 m=prepare();phase(m);comparisons=[compare(j,m) for j in m['jobs']];jobs=[json.loads((OUT/'jobs'/f"{j['id']}.json").read_text()) for j in m['jobs']];write(REPORT/'development-quadrature-qualification.json',dict(scope=m['scope'],manifest_sha256=sha(MANIFEST),jobs=jobs,comparisons=comparisons,expected_pose_checks=96,actual_pose_checks=sum(len(c['rows']) for c in comparisons),all_gates_pass=all(c['passed'] for c in comparisons),failed_jobs=[j['job']['id'] for j in jobs if j['state']!='complete']))
if __name__=='__main__':prepare() if '--prepare-only' in sys.argv else main()
