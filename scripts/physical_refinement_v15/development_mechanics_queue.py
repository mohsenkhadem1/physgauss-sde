"""Three-worker public-development bank only; no reserved truth generation."""
from pathlib import Path
import argparse,concurrent.futures,json,os,queue,signal,subprocess,sys,time
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import numpy as np
from scripts.physical_refinement_v15.study import base_cases,development_actions
from scripts.physical_refinement_v15.mechanics_adapter import specification
from scripts.physical_refinement_v15.mechanics_pilot import BASE,REPORT,write,sha
OUT=BASE/'development';MANIFEST=OUT/'mechanics-jobs.json'
GATES=dict(field_rms_m=2e-5,field_max_m=1e-4,surface_rms_m=2e-5,surface_max_m=1e-4,force_floor_N=1e-4,force_relative=.005,torque_floor_Nm=1e-5,torque_relative=.005)
def prepare():
 actions=development_actions();poses=np.asarray(list(actions.values()));cases=base_cases();jobs=[];sources={}
 for case in cases:
  name=case['name'];physical={k:v for k,v in case.items() if k!='name'}
  for n in ([4,6] if abs(case['material_gradient'])==.8 else [4]):
   spec=specification(physical,poses,n,3);sources.update(spec['source_sha256']);path=OUT/'specs'/f'{name}.json';write(path,dict(case=physical,poses=poses.tolist(),action_names=list(actions)));jobs.append(dict(id=f'{name}-n{n}',name=name,n=n,spec=str(path.relative_to(ROOT)),out=str((OUT/'hypotheses'/name/f'n{n}').relative_to(ROOT)),budget_seconds=600))
 for path in (Path(__file__),ROOT/'scripts/physical_refinement_v15/study.py',ROOT/'tests/test_physical_refinement_v15_mechanics.py'):sources[str(path.relative_to(ROOT))]=sha(path)
 manifest=dict(scope='Public development-only20kinematic hypotheses; n4modelbank,8gradient/layout/ratio cornern6checks, conditionaln8onlyif4→6insufficient; no reserved objects/actions',cases=cases,action_names=list(actions),poses_m_rad=poses.tolist(),jobs=jobs,source_sha256=sources,spec_sha256={j['spec']:sha(ROOT/j['spec']) for j in jobs},gates=GATES,workers=3,devices=['cuda:0','cuda:1','cuda:2'],per_job_seconds=600,external_timeout_seconds=610)
 if MANIFEST.exists():
  if json.loads(MANIFEST.read_text())!=manifest:raise ValueError('Refuse changed development job manifest')
 else:write(MANIFEST,manifest)
 return manifest

def guard(m):
 for path,digest in {**m['source_sha256'],**m['spec_sha256']}.items():
  if sha(ROOT/path)!=digest:raise ValueError('Pinned development source/design changed: '+path)

def phase(jobs,m):
 pending=queue.Queue()
 for j in jobs:pending.put(j)
 def worker(index):
  while True:
   try:job=pending.get_nowait()
   except queue.Empty:return
   guard(m);status=OUT/'jobs'/f"{job['id']}.json";log=OUT/'logs'/f"{job['id']}.log";log.parent.mkdir(parents=True,exist_ok=True);started=time.monotonic()
   args=[sys.executable,'-B',str(ROOT/'scripts/physical_refinement_v15/mechanics_adapter.py'),'--spec',str(ROOT/job['spec']),'--out',str(ROOT/job['out']),'--n',str(job['n']),'--device',f'cuda:{index}','--budget-seconds','600']
   write(status,dict(state='running',job=job,device=f'cuda:{index}',manifest_sha256=sha(MANIFEST),started_unix=time.time()));environment=os.environ.copy();environment.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1');timed_out=False
   with log.open('a') as stream:
    child=subprocess.Popen(args,stdout=stream,stderr=subprocess.STDOUT,env=environment,start_new_session=True)
    try:code=child.wait(timeout=610)
    except subprocess.TimeoutExpired:
     timed_out=True;os.killpg(child.pid,signal.SIGKILL);code=child.wait()
   guard(m);complete=ROOT/job['out']/'complete.json';success=code==0 and complete.exists();write(status,dict(state='complete' if success else 'failed',job=job,device=f'cuda:{index}',exit_code=code,timed_out=timed_out,elapsed_seconds=time.monotonic()-started,manifest_sha256=sha(MANIFEST),complete_sha256=sha(complete) if complete.exists() else None,log=str(log.relative_to(ROOT))));print(job['id'],'complete' if success else 'FAILED',flush=True)
 with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
  for result in [pool.submit(worker,i) for i in range(3)]:result.result()

def compare(name,a,b,m):
 rows=[];complete=True
 for i,action in enumerate(m['action_names']):
  dirs=[OUT/'hypotheses'/name/f'n{n}'/f'pose-{i:03d}' for n in (a,b)]
  if not all((p/'complete.json').exists() for p in dirs):rows.append(dict(action=action,passed=False,reason='Missing qualified pose completion'));complete=False;continue
  for p in dirs:
   record=json.loads((p/'complete.json').read_text())
   if any(sha(p/k)!=h for k,h in record['artifact_sha256'].items()):raise ValueError('Changed completed mechanics state')
  x,y=[np.load(p/'private-state.npz') for p in dirs];sx,sy=[np.load(p/'surface.npz')['vertices'] for p in dirs];wx,wy=[np.load(p/'reaction-wrench.npz') for p in dirs]
  d=x['field']-y['field'];s=sx-sy;f=np.linalg.norm(wx['forces_N']-wy['forces_N'],axis=1);t=np.linalg.norm(wx['torques_Nm']-wy['torques_Nm'],axis=1)
  metrics=dict(field_rms_m=float(np.sqrt(np.mean(d*d))),field_max_m=float(abs(d).max()),surface_rms_m=float(np.sqrt(np.mean(s*s))),surface_max_m=float(abs(s).max()),force_difference_N=f.tolist(),torque_difference_Nm=t.tolist())
  passed=all(metrics[k]<=GATES[k] for k in ('field_rms_m','field_max_m','surface_rms_m','surface_max_m')) and bool((f<=GATES['force_floor_N']+.005*np.linalg.norm(wy['forces_N'],axis=1)).all() and (t<=GATES['torque_floor_Nm']+.005*np.linalg.norm(wy['torques_Nm'],axis=1)).all())
  rows.append(dict(action=action,passed=passed,**metrics))
 return dict(name=name,mesh_pair=[a,b],complete=complete,passed=all(r['passed'] for r in rows),rows=rows)

def main():
 m=prepare();guard(m);phase([j for j in m['jobs'] if j['n']==4],m);phase([j for j in m['jobs'] if j['n']==6],m)
 checks=[compare(j['name'],4,6,m) for j in m['jobs'] if j['n']==6];write(REPORT/'development-bank-initial-qualification.json',dict(manifest_sha256=sha(MANIFEST),comparisons=checks))
 extra=[]
 for c in checks:
  if not c['passed']:
   job=next(j for j in m['jobs'] if j['name']==c['name'] and j['n']==6).copy();job.update(id=f"{c['name']}-n8",n=8,out=str((OUT/'hypotheses'/c['name']/'n8').relative_to(ROOT)));extra.append(job)
 write(OUT/'conditional-n8-jobs.json',dict(reason='Predeclared4→6 insufficiency only',jobs=extra,manifest_sha256=sha(MANIFEST)))
 if extra:phase(extra,m)
 final=[compare(j['name'],6,8,m) for j in extra];all_jobs=m['jobs']+extra;states=[json.loads((OUT/'jobs'/f"{j['id']}.json").read_text()) for j in all_jobs]
 write(REPORT/'development-bank-mechanics.json',dict(scope=m['scope'],manifest_sha256=sha(MANIFEST),initial_comparisons=checks,refined_comparisons=final,jobs=states,failures=[s['job']['id'] for s in states if s['state']!='complete'],all_spatial_gates_pass=all(c['passed'] or any(f['name']==c['name'] and f['passed'] for f in final) for c in checks)))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--prepare-only',action='store_true');a=p.parse_args();prepare() if a.prepare_only else main()
