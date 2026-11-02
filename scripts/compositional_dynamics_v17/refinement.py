"""Prospective bounded n2/3/4 dynamics refinement; original prototype untouched."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import json,time,importlib.util,multiprocessing,shutil,sys,argparse
from pathlib import Path
import torch
from scripts.compositional_dynamics_v17.prototype import ROOT,REPORT,sha,write
OLDPLAN=REPORT/'prototype-plan.json';PLAN=REPORT/'refinement-plan.json';RAW=ROOT/'runs/compositional-dynamics-v17/refinement'

def task_id(t):return f"{t['case']}-n{t['n']}-dt{t['dt']:.8f}-vq{t['volume_order']}-sq{t['surface_order']}"
def prepare():
 if PLAN.exists()or RAW.exists():raise ValueError('Preserve existing refinement')
 old=json.loads(OLDPLAN.read_text());tasks=[]
 for case in old['cases']:
  for n in [2,3,4]:
   for dt,bq,sq in [(25e-6,3,8),(50e-6,3,8),(25e-6,4,8)]+([(25e-6,3,16)]if case=='drop'else[]):
    t=dict(case=case,n=n,dt=dt,volume_order=bq,surface_order=sq,reuse=(n==2 and dt==50e-6));t['id']=task_id(t);tasks.append(t)
 sources=dict(old['source_sha256']);sources[str(Path(__file__).relative_to(ROOT))]=sha(__file__)
 artifacts=[OLDPLAN,REPORT/'prototype-results.json',REPORT/'prototype-final-independent-audit.json',REPORT/'operator-profile.json']
 for case in old['cases']:
  folder=ROOT/'runs/compositional-dynamics-v17/prototype'/f'{case}-n2-dt0.00005000';artifacts.extend([folder/'trajectory.npz',folder/'complete.json'])
 write(PLAN,dict(scope='Development refinement only; no higher loads, learned training or reserved generation. Original failures preserved.',prototype_plan_sha256=sha(OLDPLAN),tasks=tasks,expected=len(tasks),new=sum(not t['reuse']for t in tasks),workers=2,seconds=3600,device='cpu',python=sys.version,torch_version=torch.__version__,rms_definition=old['rms_definition'],gates=old['gates'],acceptance='Conjoin same-mesh50→25us temporal, volume3→4, drop surface8→16, adjacent2→3 and3→4 spatial, finite/J/penetration andfineststepenergy. Latest3→4 governscandidate; do notselectearlierpassingpairiflaterfails. No declaredcontinuumtruth.',source_sha256=sources,input_sha256={str(p.relative_to(ROOT)):sha(p)for p in artifacts},estimated_seconds='CPU2 approx25–40min basedforce/massprofile; 3600s hardcap retainsallmissing identities.'))

def verify(plan):
 for name in ['source_sha256','input_sha256']:
  if any(sha(ROOT/f)!=h for f,h in plan[name].items()):raise ValueError('Refinement source/input hash mismatch')

def run_task(args):
 task,deadline=args;torch.set_num_threads(1);plan=json.loads(PLAN.read_text());verify(plan);folder=RAW/task['id'];folder.mkdir(parents=True,exist_ok=True);p=json.loads(OLDPLAN.read_text());oldkey=f"{task['case']}-n{task['n']}-dt{task['dt']:.8f}"
 try:
  if task['reuse']:
   src=ROOT/'runs/compositional-dynamics-v17/prototype'/oldkey;row=json.loads((src/'complete.json').read_text())
   if row['plan_sha256']!=plan['prototype_plan_sha256']or not row['completed']or row['trajectory_sha256']!=sha(src/'trajectory.npz'):raise ValueError('Original cache mismatch')
   shutil.copyfile(src/'trajectory.npz',folder/'trajectory.npz');row.update(plan_sha256=sha(PLAN),reuse=dict(source=str(src.relative_to(ROOT)),complete_sha256=sha(src/'complete.json'),trajectory_sha256=sha(src/'trajectory.npz')))
  else:
   spec=importlib.util.spec_from_file_location('v17_private_refinement_prototype',ROOT/'scripts/compositional_dynamics_v17/prototype.py');impl=importlib.util.module_from_spec(spec);spec.loader.exec_module(impl)
   impl.PLAN=PLAN;impl.RAW=RAW/task['id']/'work';constructor=impl.DynamicContactModel
   def configured(*args,**kwargs):
    kwargs['order']=task['volume_order'];return constructor(*args,**kwargs)
   impl.DynamicContactModel=configured;p['surface_order']=task['surface_order'];row=impl.run_case(task['case'],task['n'],task['dt'],p,deadline)
   generated=impl.RAW/oldkey;shutil.copyfile(generated/'trajectory.npz',folder/'trajectory.npz')
  row.update(task=task,task_id=task['id'],path=str(folder.relative_to(ROOT)));write(folder/'complete.json',row);return row
 except Exception as e:
  row=dict(task=task,task_id=task['id'],path=str(folder.relative_to(ROOT)),plan_sha256=sha(PLAN),completed=False,error=f'{type(e).__name__}: {e}');write(folder/'complete.json',row);return row

def main():
 plan=json.loads(PLAN.read_text());verify(plan)
 if RAW.exists():raise ValueError('Preserve previous refinement')
 RAW.mkdir(parents=True);start=time.monotonic();deadline=start+plan['seconds']-30;report=dict(plan_sha256=sha(PLAN),expected=plan['expected'],expected_keys=[t['id']for t in plan['tasks']],rows=[],terminal=False)
 pool=multiprocessing.get_context('spawn').Pool(plan['workers'])
 try:
  results=pool.imap_unordered(run_task,[(t,deadline-15)for t in plan['tasks']])
  for _ in plan['tasks']:
   report['rows'].append(results.next(timeout=max(.01,deadline-time.monotonic())));write(REPORT/'refinement-results.json',report)
 except Exception as e:report['stage_error']=f'{type(e).__name__}: {e}'
 finally:
  pool.terminate();pool.join();rows=[];missing=[]
  for t in plan['tasks']:
   f=RAW/t['id']/'complete.json'
   if f.exists():
    r=json.loads(f.read_text())
    if r['plan_sha256']!=sha(PLAN)or r['task_id']!=t['id']:raise ValueError('Mismatched harvested task')
    rows.append(r)
   else:missing.append(t['id'])
  verify(plan);report.update(rows=rows,terminal=True,complete=not missing,completed=sum(x['completed']for x in rows),missing_keys=missing,seconds=time.monotonic()-start);write(REPORT/'refinement-results.json',report)
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--prepare',action='store_true');a=ap.parse_args();prepare()if a.prepare else main()
