"""Conditional smooth-cycle spatial/volume/time qualification, frozen waveform."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import json,time,multiprocessing,importlib.util,argparse,shutil
from pathlib import Path
import numpy as np
from scripts.compositional_dynamics_v17 import position_cycle as original
from scripts.compositional_dynamics_v17.sequence_metrics import matched_sequence_metrics
ROOT,REPORT,sha,write=original.ROOT,original.REPORT,original.sha,original.write
PLAN=REPORT/'cycle-refinement-plan.json';RAW=ROOT/'runs/compositional-dynamics-v17/cycle-refinement'
def task(n,dt=25e-6,order=3):return dict(name=f'cycle-n{n}-dt{dt:.8f}-vq{order}',dt_s=dt,n=n,volume_order=order)
def prepare():
 if PLAN.exists()or RAW.exists():raise ValueError('Preserve previous cycle qualification')
 p=json.loads(original.PLAN.read_text());p['workers']=2;p['seconds']=1800;p['scope']='Conditional n3/4 qualification, n6 only if needed and bounded runtime permits; unchanged smoothposition commands, material/support laws and gates'
 p['tasks']=[task(3,order=4)]+[task(n,dt,q)for n in [4,6]for dt,q in [(25e-6,3),(12.5e-6,3),(25e-6,4)]];p['reuse']=[task(3),task(3,12.5e-6)]
 p['acceptance']='Conjoin BOTH members timestep25/12.5us,volume3/4,spatial and individualenergy/J. Accept3/4 ifpassed; otherwise only6 if full newchecks canfitbudget. No best-of earlier pair.'
 p['conditional_n6_seconds_required']=4000;p['cost_basis']='Measured CPU n6 force70.4ms+two masssolves17.2ms =>~1051s per.3s/25us; half2102s,q4~2xforce. TwoCPU complete3jobs estimated~3100s plusconservativeoverhead. Current1800s mayexplicitlyleave n6 for separateapprovedextension.'
 p['source_sha256']=dict(p['source_sha256']);p['source_sha256'][str(Path(__file__).relative_to(ROOT))]=sha(__file__)
 for f in [ROOT/'scripts/compositional_dynamics_v17/sequence_metrics.py',ROOT/'scripts/compositional_dynamics_v17/test_cycle_refinement.py']:p['source_sha256'][str(f.relative_to(ROOT))]=sha(f)
 inputs=[original.PLAN,REPORT/'position-cycle-results.json',REPORT/'position-cycle-final-independent-audit.json',REPORT/'fine-profile.json']
 for c in p['cases']:inputs += [original.RAW/c['name']/'complete.json',original.RAW/c['name']/'trajectory.npz']
 p['input_sha256']={str(f.relative_to(ROOT)):sha(f)for f in inputs};p['original_plan_sha256']=sha(original.PLAN);write(PLAN,p)
def verify(p):
 for group in ['source_sha256','input_sha256']:
  if any(sha(ROOT/f)!=h for f,h in p[group].items()):raise ValueError('Pinned source/input changed')
def worker(args):
 t,deadline=args;spec=importlib.util.spec_from_file_location('private_cycle_refinement',ROOT/'scripts/compositional_dynamics_v17/position_cycle.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);m.PLAN=PLAN;m.RAW=RAW;ctor=m.DynamicContactModel;build=m.build
 def configured(*a,**kw):kw['order']=t['volume_order'];return ctor(*a,**kw)
 m.DynamicContactModel=configured;m.build=lambda _:build(t['n']);r=m.worker((t,deadline));r.update(n=t['n'],volume_order=t['volume_order']);write(RAW/t['name']/'complete.json',r);return r

def comparison(a,b,rows,p):
 output=dict(a=a,b=b,passed=False)
 if any(k not in rows or not rows[k]['completed']for k in [a,b]):return dict(**output,error='Incomplete/failed case retained')
 data=[]
 for key in [a,b]:
  f=RAW/key/'trajectory.npz'
  if sha(f)!=rows[key]['trajectory_sha256']:raise ValueError('Changed trajectory')
  data.append(dict(np.load(f)))
 x,y=data;ref=y['surface_frames'].reshape(len(y['times_s']),-1,3);pred=x['surface_frames'].reshape(len(x['times_s']),-1,3);m=matched_sequence_metrics(y['surface_rest'].reshape(-1,3),y['times_s'],ref,x['times_s'],pred,np.ones(ref.shape[1]),characteristic_length_m=.06)
 checks=dict(world=bool(m['world_coordinate_rms_m']<=p['gates']['world_coordinate_RMS_m']),shape=bool(m['shape_coordinate_rms_m']<=p['gates']['shape_coordinate_RMS_m']),energy_J=all(rows[k]['diagnostic_gates_pass']for k in [a,b]));return dict(**{k:v for k,v in output.items()if k!='passed'},passed=all(checks.values()),checks=checks,world_coordinate_RMS_m=float(m['world_coordinate_rms_m']),shape_coordinate_RMS_m=float(m['shape_coordinate_rms_m']))
def analyze(rows,p,lo,hi):
 by={r['name']:r for r in rows};checks=[]
 for n in [lo,hi]:
  for typ,a,b in [('time',task(n)['name'],task(n,12.5e-6)['name']),('volume',task(n)['name'],task(n,order=4)['name'])]:checks.append(dict(type=typ,n=n,**comparison(a,b,by,p)))
 checks.append(dict(type='spatial',pair=[lo,hi],**comparison(task(lo)['name'],task(hi)['name'],by,p)));return dict(pair=[lo,hi],passed=all(c['passed']for c in checks),checks=checks)
def main():
 p=json.loads(PLAN.read_text());verify(p)
 if RAW.exists():raise ValueError('Preserve earlier output')
 avail=int(next(l.split()[1]for l in Path('/proc/meminfo').read_text().splitlines()if l.startswith('MemAvailable:')))*1024
 if avail<p['minimum_available_memory_GiB']*2**30:raise RuntimeError('Insufficient available memory')
 RAW.mkdir(parents=True);start=time.monotonic();deadline=start+p['seconds']-30;rows=[];report=dict(plan_sha256=sha(PLAN),rows=rows,terminal=False);out=REPORT/'cycle-refinement-results.json'
 for t in p['reuse']:
  src=original.RAW/f"cycle-dt{t['dt_s']:.8f}";r=json.loads((src/'complete.json').read_text())
  if not r['completed']or r['plan_sha256']!=p['original_plan_sha256']or sha(src/'trajectory.npz')!=r['trajectory_sha256']:raise ValueError('Original cache mismatch')
  dst=RAW/t['name'];dst.mkdir();shutil.copyfile(src/'trajectory.npz',dst/'trajectory.npz');r.update(name=t['name'],case=t,n=t['n'],volume_order=t['volume_order'],path=str(dst.relative_to(ROOT)),plan_sha256=sha(PLAN),reuse=dict(source=str(src.relative_to(ROOT)),complete_sha256=sha(src/'complete.json'),trajectory_sha256=sha(src/'trajectory.npz')));write(dst/'complete.json',r);rows.append(r)
 pool=multiprocessing.get_context('spawn').Pool(p['workers']);scheduled=[];latest=4
 try:
  for level in [4,6]:
   if level==6:
    report['preserved_3_4']=report['analysis']
    if report['analysis']['passed']:report['n6_decision']='not_required_3_4_passed';break
    if deadline-time.monotonic()<p['conditional_n6_seconds_required']:report['n6_decision']='budget_insufficient_separate_extension_required';break
   selected=[t for t in p['tasks']if t['n']<=4]if level==4 else[t for t in p['tasks']if t['n']==6];scheduled+=selected;latest=level;it=pool.imap_unordered(worker,[(t,deadline-15)for t in selected])
   for _ in selected:rows.append(it.next(timeout=max(.01,deadline-time.monotonic())));write(out,report)
   report['analysis']=analyze(rows,p,3 if level==4 else 4,level);write(out,report)
 except Exception as ex:report['error']=f'{type(ex).__name__}: {ex}'
 finally:
  pool.terminate();pool.join();harvest=[]
  for t in p['reuse']+p['tasks']:
   f=RAW/t['name']/'complete.json'
   if f.exists():
    r=json.loads(f.read_text())
    if r['plan_sha256']!=sha(PLAN)or r['case']!=t:raise ValueError('Harvest identity mismatch')
    harvest.append(r)
  rows[:]=harvest;verify(p);report.update(analysis=analyze(rows,p,3 if latest==4 else 4,latest),terminal=True,completed=sum(r['completed']for r in rows),scheduled_names=[t['name']for t in p['reuse']+scheduled],missing_keys=[t['name']for t in scheduled if t['name']not in{r['name']for r in rows}],not_scheduled_keys=[t['name']for t in p['tasks']if t not in scheduled],seconds=time.monotonic()-start);write(out,report)
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--prepare',action='store_true');a=ap.parse_args();prepare()if a.prepare else main()
