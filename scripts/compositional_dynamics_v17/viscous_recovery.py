"""Prospective bounded retry of only terminal-original incomplete viscous jobs.

No checkpoint stitching, changed physics, or overwrite of original failures.
"""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import json,time,multiprocessing,argparse,importlib.util,shutil,subprocess
from pathlib import Path
from scripts.compositional_dynamics_v17.viscous_development import ROOT,REPORT,sha,write
OLDPLAN=REPORT/'viscous-development-plan.json';OLDRESULT=REPORT/'viscous-development-results.json'
PLAN=REPORT/'viscous-recovery-plan.json';RAW=ROOT/'runs/compositional-dynamics-v17/viscous-recovery'
def select_cases(plan,report):
 if not report.get('terminal'):raise ValueError('Original stage must be terminal')
 rows={r['name']:r for r in report['rows']}
 if len(rows)!=len(report['rows'])or set(rows)-{c['name']for c in plan['cases']}:raise ValueError('Duplicate/undeclared original identities')
 for c in plan['cases']:
  if c['name']in rows and rows[c['name']]['case']!=c:raise ValueError('Original case changed')
 return [c for c in plan['cases']if not rows.get(c['name'],{}).get('completed',False)]
def validate_original_assets(p,report,root=ROOT):
 for r in report['rows']:
  if r.get('plan_sha256')!=p['recovery']['original_plan_sha256']or type(r.get('completed'))is not bool:raise ValueError('Original row plan/completion type mismatch')
  folder=root/r['path'];completion=folder/'complete.json';trajectory=folder/'trajectory.npz'
  if not completion.exists()or json.loads(completion.read_text())!=r:raise ValueError('Original row/completion mismatch')
  if r['completed']and not trajectory.exists():raise ValueError('Completed original has no trajectory')
  if trajectory.exists()and sha(trajectory)!=r.get('trajectory_sha256'):raise ValueError('Original trajectory digest mismatch')

def prepare():
 if PLAN.exists()or RAW.exists():raise ValueError('Preserve existing recovery')
 p=json.loads(OLDPLAN.read_text());p['seconds']=7200;p['workers']=2;p['scope']='Separate bounded completion of original incomplete viscous jobs, same physics from initial state; original failures retained'
 p['recovery']=dict(original_plan_sha256=sha(OLDPLAN),selection='Only original declared cases missing or completed=false after terminal original stage; completed original cases are not recomputed. All original rows preserved by immutable runtime snapshot.',original_unit='compositional-v17-viscous-development.service',restart='Exact initial state, no checkpoint stitching or method changes')
 p['source_sha256']=dict(p['source_sha256']);p['source_sha256'][str(Path(__file__).relative_to(ROOT))]=sha(__file__);test=ROOT/'scripts/compositional_dynamics_v17/test_viscous_recovery.py';p['source_sha256'][str(test.relative_to(ROOT))]=sha(test);p['input_sha256']=dict(p['input_sha256']);p['input_sha256'][str(OLDPLAN.relative_to(ROOT))]=sha(OLDPLAN);write(PLAN,p)
def verify(p):
 for group in ['source_sha256','input_sha256']:
  if any(sha(ROOT/f)!=h for f,h in p[group].items()):raise ValueError('Recovery source/input digest mismatch')
def worker(arg):
 spec=importlib.util.spec_from_file_location('private_viscous_recovery',ROOT/'scripts/compositional_dynamics_v17/viscous_development.py');impl=importlib.util.module_from_spec(spec);spec.loader.exec_module(impl);impl.PLAN=PLAN;impl.RAW=RAW;return impl.worker(arg)
def main():
 p=json.loads(PLAN.read_text());verify(p)
 if RAW.exists():raise ValueError('Preserve previous recovery')
 status=subprocess.check_output(['systemctl','--user','show',p['recovery']['original_unit'],'-p','ActiveState','-p','MainPID'],text=True)
 if 'MainPID=0'not in status or any(x in status for x in ['ActiveState=active','ActiveState=activating','ActiveState=deactivating']):raise ValueError('Original service still live')
 original=json.loads(OLDRESULT.read_text())
 if original['plan_sha256']!=p['recovery']['original_plan_sha256']:raise ValueError('Original result/plan mismatch')
 selected=select_cases(p,original);validate_original_assets(p,original);available=int(next(l.split()[1]for l in Path('/proc/meminfo').read_text().splitlines()if l.startswith('MemAvailable:')))*1024
 if available<p['minimum_available_memory_GiB']*2**30:raise RuntimeError('Memory headroom unavailable')
 RAW.mkdir(parents=True);snapshot=RAW/'original-terminal-results.json';shutil.copyfile(OLDRESULT,snapshot);manifest=dict(plan_sha256=sha(PLAN),original_result_sha256=sha(OLDRESULT),snapshot_sha256=sha(snapshot),original_service_status=status,selected_names=[c['name']for c in selected],completed_original_names=[c['name']for c in p['cases']if c not in selected],original_artifacts={})
 for r in original['rows']:
  folder=ROOT/r['path']
  for name in ['complete.json','trajectory.npz']:
   f=folder/name
   if f.exists():manifest['original_artifacts'][str(f.relative_to(ROOT))]=sha(f)
 write(RAW/'runtime-input-freeze.json',manifest);start=time.monotonic();deadline=start+p['seconds']-30;report=dict(plan_sha256=sha(PLAN),runtime_freeze_sha256=sha(RAW/'runtime-input-freeze.json'),selected_names=manifest['selected_names'],rows=[],terminal=False,original_results_preserved=True);out=REPORT/'viscous-recovery-results.json';pool=multiprocessing.get_context('spawn').Pool(p['workers'])
 try:
  it=pool.imap_unordered(worker,[(c,deadline-15)for c in selected])
  for _ in selected:report['rows'].append(it.next(timeout=max(.01,deadline-time.monotonic())));write(out,report)
 except Exception as ex:report['error']=f'{type(ex).__name__}: {ex}'
 finally:
  pool.terminate();pool.join();rows=[]
  for c in selected:
   f=RAW/c['name']/'complete.json'
   if f.exists():
    r=json.loads(f.read_text())
    if r['plan_sha256']!=sha(PLAN)or r['case']!=c:raise ValueError('Recovery identity mismatch')
    rows.append(r)
  verify(p)
  if sha(OLDRESULT)!=manifest['original_result_sha256']or any(sha(ROOT/f)!=h for f,h in manifest['original_artifacts'].items()):raise ValueError('Original artifacts changed')
  report.update(rows=rows,terminal=True,completed=sum(r['completed']for r in rows),missing_keys=[c['name']for c in selected if c['name']not in{r['name']for r in rows}],seconds=time.monotonic()-start);write(out,report)
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--prepare',action='store_true');a=ap.parse_args();prepare()if a.prepare else main()
