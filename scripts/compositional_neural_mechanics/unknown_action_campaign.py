"""Prospective six-object conditional unknown-physics/action benchmark.

Separate truth exporter, input-only fitting workers, terminal-gated scorer.
"""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import argparse,json,signal,subprocess,sys,time,concurrent.futures
from pathlib import Path
import numpy as np
import torch
from scripts.compositional_neural_mechanics.local_action_pilot import serial,fit,prediction,ACTIONS
from scripts.compositional_neural_mechanics.full_continuous_inference_pilot import build,pack
from scripts.compositional_neural_mechanics.conditional_inference_pilot import POINTS,BULK,INTERFACE,predict
from scripts.compositional_neural_mechanics.scenes import sample_scene,poses
from scripts.compositional_neural_mechanics.mechanics import GeometrySpec
from scripts.compositional_neural_mechanics.gauss_newton_inference import GaussNewtonMAP
from scripts.compositional_neural_mechanics.mechanics_uncertainty import local_covariance,public_position_jacobian
from scripts.compositional_neural_mechanics.local_uncertainty import rank_position_probes
from scripts.compositional_neural_mechanics.dataset_runner import write,npz
from scripts.compositional_neural_mechanics_reporting.legacy_dependency_audit import closure,sha
ROOT=Path(__file__).resolve().parents[2];R=ROOT/'reports/compositional-neural-mechanics-v16';B=ROOT/'runs/compositional-neural-mechanics-v16/unknown-action-reserved';PLAN=R/'unknown-action-method-freeze.json';MODELS=('analytic_fitted','shared_laws_fitted')

def prepare():
 if PLAN.exists()or B.exists():raise ValueError('Preserve prior fresh study')
 draft=R/'unknown-action-prospective-draft.json';plan=json.loads(draft.read_text());preflight=R/'unknown-action-development-preflight.json'
 if len(json.loads(preflight.read_text())['rows'])!=2 or any(x['state']!='complete'for x in json.loads(preflight.read_text())['rows']):raise ValueError('Development preflight required')
 plan.update(status='frozen_pending_independent_launch_review',fit_seconds=900,max_evaluations=40,stage_seconds=21600,workers=4,source_sha256=closure([Path(__file__).resolve(),ROOT/'scripts/compositional_neural_mechanics/unknown_action_score.py',ROOT/'tests/test_unknown_action_v16.py']),artifact_sha256={str(p.relative_to(ROOT)):sha(p)for p in (draft,preflight,BULK,INTERFACE)},scope=__doc__)
 plan.pop('not_yet_done',None)
 snapshots=R/'unknown-action-source-snapshots';snapshots.mkdir(exist_ok=False)
 for name,h in plan['source_sha256'].items():(snapshots/(h+'.py')).write_bytes((ROOT/name).read_bytes())
 write(PLAN,plan)

def guard():
 p=json.loads(PLAN.read_text())
 for table in ('source_sha256','artifact_sha256'):
  for path,h in p[table].items():
   if sha(ROOT/path)!=h:raise ValueError('Frozen source/input changed')
 return p

def held_commands(g):
 a=np.zeros((2,6));a[1,4]=.30;b=np.zeros((2,6));b[0,0]=-.07*g.dimensions[0];b[1,0]=.07*g.dimensions[0];b[0,3]=-.30;b[1,3]=.30
 return [a,b]

def generate():
 p=guard();B.mkdir(parents=True,exist_ok=False);write(B/'stage-clock.json',dict(plan_sha256=sha(PLAN),deadline_unix=time.time()+p['stage_seconds']));deadline=json.loads((B/'stage-clock.json').read_text())['deadline_unix'];rng=np.random.default_rng(p['scene_seed']);noise=np.random.default_rng(p['noise_seed']).normal(size=(6,4,26,3))*50e-6;rows=[]
 def alarm(*args):raise TimeoutError('Reference solve300s cap')
 signal.signal(signal.SIGALRM,alarm)
 for i,cell in enumerate(p['cells']):
  scene=sample_scene(cell,rng,f'unknown-reserved-{i}',gravity=(0,0,-9.81))
  for z in scene['interface_latents']:z[2]=p['interface_families'][i]
  g=GeometrySpec(**scene['geometry']);categories=dict(bulk_family=scene['bulk_family'],interface_family=p['interface_families'][i],heterogeneity=scene['factors']['heterogeneity']);commands=poses(g);observed=[commands[a]for a in ('initial',*ACTIONS)];held=held_commands(g);folder=B/f'world{i}';folder.mkdir()
  public=dict(geometry=scene['geometry'],categories=categories,points=POINTS.tolist(),observed_poses=[a.tolist()for a in observed],held_poses=[a.tolist()for a in held]);write(folder/'public.json',public)
  teacher=build(g,categories,[dict(pose=observed[0],points=POINTS,displacement=np.zeros_like(POINTS))],False);teacher.kw['n']=4;values=[];row=dict(world=i)
  try:
   for pose in [*observed,*held]:
    remaining=deadline-time.time()
    if remaining<=0:raise TimeoutError('Whole6hourstage deadline')
    signal.setitimer(signal.ITIMER_REAL,min(300,remaining))
    try:value=predict(teacher,pack(scene),pose)
    finally:signal.setitimer(signal.ITIMER_REAL,0)
    if any(not np.isfinite(x).all()for x in value.values()):raise ValueError('Nonfinite reference')
    values.append(value)
   for j,a in enumerate(('initial',*ACTIONS)):npz(folder/(a+'.npz'),displacement=values[j]['displacement']+noise[i,j])
   private=folder/'private';private.mkdir();write(private/'scene.json',scene);npz(private/'held-truth.npz',**{f'held{j}':v['displacement']for j,v in enumerate(values[4:])},parameters=pack(scene))
   inputs={str(x.relative_to(ROOT)):sha(x)for x in folder.iterdir()if x.is_file()};privatefiles={str(x.relative_to(ROOT)):sha(x)for x in private.iterdir()}
   row.update(state='complete',public_sha256=inputs,private_sha256=privatefiles)
  except Exception as exc:row.update(state='failed',error=repr(exc))
  rows.append(row);write(B/'generation-progress.json',dict(plan_sha256=sha(PLAN),rows=rows))
 guard();write(B/'generation-complete.json',dict(plan_sha256=sha(PLAN),rows=rows))

def input_guard(world):
 p=guard();manifest=json.loads((B/'generation-complete.json').read_text())
 if manifest['plan_sha256']!=sha(PLAN)or len(manifest['rows'])!=6 or {r['world']for r in manifest['rows']}!=set(range(6)):raise ValueError('Exact generation commitment required')
 start=json.loads((B/'run-start.json').read_text())
 if start['plan_sha256']!=sha(PLAN)or start['generation_sha256']!=sha(B/'generation-complete.json')or start['stage_clock_sha256']!=sha(B/'stage-clock.json'):raise ValueError('Generation/runtime commitment changed')
 row=manifest['rows'][world]
 if row['world']!=world or row['state']!='complete':raise ValueError('Reference generation failed; retain failed world')
 for path,h in row['public_sha256'].items():
  if sha(ROOT/path)!=h:raise ValueError('Allowed input changed')
 return p

def worker(world,model):
 torch.set_num_threads(1);p=input_guard(world);folder=B/f'world{world}';output=folder/model;output.mkdir(exist_ok=False);public=json.loads((folder/'public.json').read_text());q=np.asarray(public['points']);observed=np.asarray(public['observed_poses']);learned=model=='shared_laws_fitted'
 with np.load(folder/'initial.npz',allow_pickle=False)as data:obs=dict(pose=observed[0],points=q,displacement=data['displacement'].copy())
 estimator=build(GeometrySpec(**public['geometry']),public['categories'],[obs],learned);initial=fit(estimator,None,p);write(output/'initial-fit.json',serial(initial));theta=initial['parameters'];cov=local_covariance(GaussNewtonMAP(estimator),theta,fit_status=initial)
 tasks=[dict(name=f'held{j}',pose=np.asarray(pose),points=q)for j,pose in enumerate(public['held_poses'])];task=public_position_jacobian(estimator,theta,tasks)
 candidateJ={a:public_position_jacobian(estimator,theta,[dict(name=a,pose=observed[j],points=q)])['whitened_jacobian']for j,a in enumerate(ACTIONS,1)}
 ranking=rank_position_probes(cov['covariance'],task['position_jacobian_m'],candidateJ);random=str(np.random.default_rng(p['random_policy_seed']).choice(ACTIONS,size=6)[world]);selection={**{'fixed_'+a:a for a in ACTIONS},'uniform_random':random,'target_task_variance':ranking[0]['action']}
 write(output/'selection.json',dict(plan_sha256=sha(PLAN),policies=selection,ranking=ranking,initial_fit_sha256=sha(output/'initial-fit.json')))
 predictions={'initial':prediction(estimator,initial,tasks)};fits={}
 for j,a in enumerate(ACTIONS,1):
  try:
   with np.load(folder/(a+'.npz'),allow_pickle=False)as data:new=dict(pose=observed[j],points=q,displacement=data['displacement'].copy())
   after=build(GeometrySpec(**public['geometry']),public['categories'],[obs,new],learned);result=fit(after,theta,p);fits[a]=serial(result);predictions[a]=prediction(after,result,tasks)
  except Exception as exc:fits[a]=dict(incomplete=True,optimizer_success=False,error=repr(exc));predictions[a]=dict(error=repr(exc))
  write(output/'fits.json',fits)
 arrays={}
 for a,value in predictions.items():
  if 'mean'in value:arrays[a+'_mean']=value['mean'];arrays[a+'_sd']=value['sd']
 npz(output/'forecasts.npz',**arrays);input_guard(world)
 write(output/'complete.json',dict(plan_sha256=sha(PLAN),generation_sha256=sha(B/'generation-complete.json'),evidence_sha256={name:sha(output/name)for name in ('initial-fit.json','selection.json','fits.json','forecasts.npz')},diagnostics=serial({a:{k:v for k,v in value.items()if k not in ('mean','sd')}for a,value in predictions.items()})))

def run():
 p=guard()
 with (B/'run-start.json').open('x')as stream:json.dump(dict(plan_sha256=sha(PLAN),start=time.time(),generation_sha256=sha(B/'generation-complete.json'),stage_clock_sha256=sha(B/'stage-clock.json')),stream)
 start=time.monotonic();clock=json.loads((B/'stage-clock.json').read_text())
 if clock['plan_sha256']!=sha(PLAN):raise ValueError('Stage clock changed')
 deadline=clock['deadline_unix'];jobs=[(i,m)for i in range(6)for m in MODELS]
 def one(job):
  world,model=job;log=B/f'fit-{world}-{model}.log'
  if time.time()>=deadline:return dict(world=world,model=model,exit_code=None,error='Whole6hourstage deadline; job not started')
  with log.open('x')as stream:
   child=subprocess.Popen([sys.executable,'-m','scripts.compositional_neural_mechanics.unknown_action_campaign','--worker',str(world),'--model',model],stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
   try:code=child.wait(timeout=max(.01,min(4000,deadline-time.time())))
   except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);code=child.wait()
  return dict(world=world,model=model,exit_code=code)
 with concurrent.futures.ThreadPoolExecutor(max_workers=p['workers'])as pool:status=list(pool.map(one,jobs))
 write(B/'fits-terminal.json',dict(plan_sha256=sha(PLAN),generation_sha256=sha(B/'generation-complete.json'),run_start_sha256=sha(B/'run-start.json'),jobs=status,elapsed_seconds=time.monotonic()-start))
 # Deliberately no scoring here: separately audited scorer is a later explicit stage.

if __name__=='__main__':
 parser=argparse.ArgumentParser();g=parser.add_mutually_exclusive_group(required=True);g.add_argument('--prepare',action='store_true');g.add_argument('--generate',action='store_true');g.add_argument('--run',action='store_true');g.add_argument('--worker',type=int);parser.add_argument('--model',choices=MODELS);a=parser.parse_args()
 if a.prepare:prepare()
 elif a.generate:generate()
 elif a.run:run()
 else:worker(a.worker,a.model)
