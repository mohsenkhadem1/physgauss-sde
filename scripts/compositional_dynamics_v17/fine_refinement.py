"""Conditional 4/6 then 6/8 qualification, original energy and fixed gates."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import json,time,multiprocessing,shutil,argparse
from pathlib import Path
import numpy as np
from scripts.compositional_dynamics_v17 import refinement as old
from scripts.compositional_dynamics_v17.refinement_analysis import comparison,physical_events
ROOT,REPORT,sha,write=old.ROOT,old.REPORT,old.sha,old.write
PLAN=REPORT/'fine-refinement-plan.json';RAW=ROOT/'runs/compositional-dynamics-v17/fine-refinement'
def key(case,n,dt=25e-6,bq=3,sq=8):return old.task_id(dict(case=case,n=n,dt=dt,volume_order=bq,surface_order=sq))
def tasks(case,n):
 return [dict(case=case,n=n,dt=dt,volume_order=bq,surface_order=sq,reuse=False,id=key(case,n,dt,bq,sq))for dt,bq,sq in [(25e-6,3,8),(12.5e-6,3,8),(25e-6,4,8)]+([(25e-6,3,16)]if case=='drop'else[])]
def prepare():
 if PLAN.exists()or RAW.exists():raise ValueError('Preserve prepared plans/results')
 prior=json.loads(old.PLAN.read_text());profile=json.loads((REPORT/'fine-profile.json').read_text());sources=dict(prior['source_sha256'])
 for f in [Path(__file__),ROOT/'scripts/compositional_dynamics_v17/refinement_analysis.py',ROOT/'scripts/compositional_dynamics_v17/prototype_analysis.py',ROOT/'scripts/compositional_dynamics_v17/sequence_metrics.py',ROOT/'scripts/compositional_dynamics_v17/fine_profile.py',ROOT/'scripts/compositional_dynamics_v17/test_fine_refinement.py']:sources[str(f.relative_to(ROOT))]=sha(f)
 artifacts=[old.PLAN,REPORT/'refinement-results.json',REPORT/'refinement-analysis.json',REPORT/'refinement-final-independent-audit.json',REPORT/'fine-profile.json'];reuse=[]
 for r in json.loads((REPORT/'refinement-results.json').read_text())['rows']:
  if r['n']==4:
   reuse.append(r['task_id']);artifacts.extend([ROOT/r['path']/'complete.json',ROOT/r['path']/'trajectory.npz'])
 all_tasks=[t for n in [6,8]for c in ['release','drop']for t in tasks(c,n)]
 write(PLAN,dict(scope='Development only; bounded maximum n8, no changed constitutive law/load/gates or reserved generation',source_sha256=sources,input_sha256={str(f.relative_to(ROOT)):sha(f)for f in artifacts},prototype_plan_sha256=prior['prototype_plan_sha256'],gates=prior['gates'],rms_definition=prior['rms_definition'],workers=4,seconds=10800,minimum_available_memory_GiB=32,tasks=all_tasks,reuse_keys=reuse,expected_maximum=len(all_tasks)+len(reuse),conditional='Run n6 both cases; n8 only for case failing full4/6 conjunction, and if remaining time exceeds 1.5*profile n8 halfstep duration plus120s. Unlaunched n8 explicitly skipped or budget-missing. No further mesh/dt ladder.',acceptance='Selected4/6 pair if all checks pass; otherwise latest6/8 if launched, with BOTH-member time,volume,contact,energy,J,penetration and spatial/events. n4 cached50/25us; n6/n8 actual25/12.5us. All spatial/volume/contact at25us. No fallback to earlier pair after later failure.',profile=profile['rows'],resource_estimate='CPU4 cap3h; n8 halfstep singleworker~72min before contention; no GPU. Runtime cost guard may leave higher case explicitly incomplete.'))
def normalized(row,task):
 row=dict(row)
 for name in ['case','n','dt']:
  if name in row and row[name]!=task[name]:raise ValueError('Conflicting failed/successful task identity')
  row[name]=task[name]
 if row['task_id']!=task['id']:raise ValueError('Conflicting task key')
 return row
def worker(arg):
 old.PLAN=PLAN;old.RAW=RAW;row=normalized(old.run_task(arg),arg[0]);write(RAW/arg[0]['id']/'complete.json',row);return row
def verify(p):old.verify(p)
def analyze(rows,case,lo,hi):
 p=json.loads(old.OLDPLAN.read_text());rs={r['task_id']:r for r in rows};data={};events={}
 for k,r in rs.items():
  if r['case']!=case or r['n']not in [lo,hi]:continue
  f=ROOT/r['path']/'trajectory.npz'
  if r.get('completed')and f.exists():
   if sha(f)!=r['trajectory_sha256']:raise ValueError('Trajectory digest mismatch')
   data[k]=dict(np.load(f));events[k]=physical_events(data[k],r,p)
 checks=[]
 for n in [lo,hi]:
  pairs=[('time',key(case,n,50e-6)if n==4 else key(case,n),key(case,n)if n==4 else key(case,n,12.5e-6)),('volume',key(case,n),key(case,n,bq=4))]
  if case=='drop':pairs.append(('contact',key(case,n),key(case,n,sq=16)))
  for typ,a,b in pairs:checks.append(dict(type=typ,n=n,**comparison(a,b,rs,data,events,p)))
 checks.append(dict(type='spatial',pair=[lo,hi],**comparison(key(case,lo),key(case,hi),rs,data,events,p)))
 return dict(pair=[lo,hi],passed=all(c['passed']for c in checks),checks=checks)
def main():
 p=json.loads(PLAN.read_text());verify(p)
 available=int(next(line.split()[1]for line in Path('/proc/meminfo').read_text().splitlines()if line.startswith('MemAvailable:')))*1024
 if available<p['minimum_available_memory_GiB']*2**30:raise RuntimeError('Insufficient available memory for declared CPU4 stage')
 if RAW.exists():raise ValueError('Preserve old output')
 RAW.mkdir(parents=True);start=time.monotonic();deadline=start+p['seconds']-45;rows=[];report=dict(plan_sha256=sha(PLAN),rows=rows,terminal=False,decisions={},analyses={},expected_maximum=p['expected_maximum']);out=REPORT/'fine-refinement-results.json'
 for k in p['reuse_keys']:
  src=old.RAW/k;dest=RAW/k;dest.mkdir();r=json.loads((src/'complete.json').read_text());shutil.copyfile(src/'trajectory.npz',dest/'trajectory.npz');r.update(plan_sha256=sha(PLAN),path=str(dest.relative_to(ROOT)),reuse=dict(source=str(src.relative_to(ROOT)),complete_sha256=sha(src/'complete.json'),trajectory_sha256=sha(src/'trajectory.npz')));write(dest/'complete.json',r);rows.append(r)
 pool=multiprocessing.get_context('spawn').Pool(p['workers']);declared=[]
 try:
  for n in [6,8]:
   scheduled=[]
   for case in ['release','drop']:
    if n==8:
     if report['analyses'][case]['passed']:report['decisions'][case]='n8_not_required_4_6_passed';continue
     cost=1.5*2*next(x['drop25us_estimate_seconds']for x in p['profile']if x['n']==8)*(0.04/0.25 if case=='release'else 1)+120
     if deadline-time.monotonic()<cost:report['decisions'][case]='n8_budget_missing';continue
    scheduled+=tasks(case,n)
   declared+=scheduled;results=pool.imap_unordered(worker,[(t,deadline-15)for t in scheduled])
   for _ in scheduled:rows.append(results.next(timeout=max(.01,deadline-time.monotonic())));write(out,report)
   for case in ['release','drop']:
    if any(t['case']==case for t in scheduled):
     previous=report['analyses'].get(case);report['analyses'][case]=analyze(rows,case,4 if n==6 else 6,n)
     if previous:report.setdefault('preserved_4_6',{})[case]=previous
   write(out,report)
 except Exception as e:report['error']=f'{type(e).__name__}: {e}'
 finally:
  pool.terminate();pool.join();harvest=[]
  for t in p['tasks']:
   f=RAW/t['id']/'complete.json'
   if f.exists():
    r=normalized(json.loads(f.read_text()),t)
    if r['plan_sha256']!=sha(PLAN)or r['task_id']!=t['id']:raise ValueError('Harvest identity mismatch')
    harvest.append(r)
  rows[:]=[r for r in rows if r['task_id']in p['reuse_keys']]+harvest;verify(p)
  for c in ['release','drop']:
   latest=8 if any(t['case']==c and t['n']==8 for t in declared)else 6;report['analyses'][c]=analyze(rows,c,6 if latest==8 else 4,latest)
  report['not_scheduled_keys']=[t['id']for t in p['tasks']if t['id']not in{q['id']for q in declared}];report['memory_available_at_launch_bytes']=available
  report.update(terminal=True,completed=sum(r.get('completed',False)for r in rows),attempted=len(rows),scheduled_keys=p['reuse_keys']+[t['id']for t in declared],missing_keys=[t['id']for t in declared if t['id']not in{r['task_id']for r in rows}],all_cases_qualified=all(x['passed']for x in report['analyses'].values()),seconds=time.monotonic()-start);write(out,report)
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--prepare',action='store_true');a=ap.parse_args();prepare()if a.prepare else main()
