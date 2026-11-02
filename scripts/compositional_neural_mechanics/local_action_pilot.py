"""Two exposed DEVELOPMENT worlds; target-task-aware local Gaussian action diagnostic."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import argparse,json,signal,subprocess,sys,time
from pathlib import Path
import numpy as np
import torch
from scripts.compositional_neural_mechanics.full_continuous_inference_pilot import build
from scripts.compositional_neural_mechanics.conditional_inference_pilot import BULK,INTERFACE
from scripts.compositional_neural_mechanics.gauss_newton_inference import GaussNewtonMAP
from scripts.compositional_neural_mechanics.mechanics_uncertainty import local_covariance,public_position_jacobian
from scripts.compositional_neural_mechanics.local_uncertainty import rank_position_probes,predictive_variance
from scripts.compositional_neural_mechanics.mechanics import GeometrySpec
from scripts.compositional_neural_mechanics.dataset_runner import write,npz
from scripts.compositional_neural_mechanics_reporting.legacy_dependency_audit import closure,sha
ROOT=Path(__file__).resolve().parents[2];REPORT=ROOT/'reports/compositional-neural-mechanics-v16';BASE=ROOT/'runs/compositional-neural-mechanics-v16';OUT=BASE/'local-action-development';PLAN=REPORT/'local-action-development-plan.json'
ACTIONS=('stretch','right_bend','opposed_twist')

def serial(x):
 if isinstance(x,np.ndarray):return x.tolist()
 if isinstance(x,np.generic):return x.item()
 if isinstance(x,dict):return {k:serial(v)for k,v in x.items()}
 if isinstance(x,(list,tuple)):return [serial(v)for v in x]
 return x

def prepare():
 if PLAN.exists()or OUT.exists():raise ValueError('Preserve prior plan')
 inputs=[BULK,INTERFACE,REPORT/'gauss-newton-development-results.json']
 inputs += [BASE/f'full-continuous-inference-pilot/world{i}/observations-and-development-truth.npz'for i in (0,1)]
 inputs += [REPORT/f'full-continuous-inference-pilot-world{i}.json'for i in (0,1)]
 plan=dict(scope=__doc__,worlds=[0,1],model='learned',conditioned='Known geometry/categorical families/heterogeneity/gravity/density;15continuous unknowns',active=list(range(15)),max_evaluations=40,fit_seconds=900,stage_seconds=4000,workers=2,seed=926847,actions=list(ACTIONS),policies=['fixed_'+a for a in ACTIONS]+['random','target_task_variance'],initial='Neutral, onlyobs0, original unitprior/bounds',discrepancy_variance_m2=0.,nominal_interval=.90,limitations='Exposed development cases; local untruncated Gaussian not calibrated; known target commands; no new views;26points already tracked; no best-fixed outcome selection.',source_sha256=closure([Path(__file__).resolve(),ROOT/'tests/test_local_action_pilot_v16.py',ROOT/'tests/test_mechanics_uncertainty_v16.py']),artifact_sha256={str(p.relative_to(ROOT)):sha(p)for p in inputs})
 write(PLAN,plan)
 # Exporter is separate from worker: worker never opens the combined truth archive.
 OUT.mkdir();exports={}
 for i in (0,1):
  prior=json.loads((REPORT/f'full-continuous-inference-pilot-world{i}.json').read_text());folder=OUT/f'world{i}';folder.mkdir()
  with np.load(BASE/f'full-continuous-inference-pilot/world{i}/observations-and-development-truth.npz',allow_pickle=False)as raw:
   public=dict(geometry=prior['scene']['geometry'],categories=prior['categories_conditioned'],points=raw['points'].tolist(),observed_poses=raw['observed_poses'].tolist(),held_poses=raw['held_poses'].tolist())
   write(folder/'public.json',public);npz(folder/'initial.npz',displacement=raw['obs_0'])
   for j,a in enumerate(ACTIONS,1):npz(folder/(a+'.npz'),displacement=raw[f'obs_{j}'])
  for p in folder.iterdir():exports[str(p.relative_to(ROOT))]=sha(p)
 write(OUT/'export-manifest.json',dict(plan_sha256=sha(PLAN),artifact_sha256=exports))

def guard():
 plan=json.loads(PLAN.read_text())
 for table in ('source_sha256','artifact_sha256'):
  for p,h in plan[table].items():
   if sha(ROOT/p)!=h:raise ValueError('Plan evidence changed')
 export=json.loads((OUT/'export-manifest.json').read_text())
 if export['plan_sha256']!=sha(PLAN):raise ValueError('Export plan changed')
 for p,h in export['artifact_sha256'].items():
  if sha(ROOT/p)!=h:raise ValueError('Export changed')
 return plan

def fit(estimator,initial,plan):
 def alarm(*args):raise TimeoutError('External900s fit deadline')
 signal.signal(signal.SIGALRM,alarm);signal.setitimer(signal.ITIMER_REAL,plan['fit_seconds'])
 try:return GaussNewtonMAP(estimator).fit(active=np.arange(15),initial=initial,max_evaluations=40,seconds=899)
 finally:signal.setitimer(signal.ITIMER_REAL,0)

def prediction(estimator,result,commands):
 cov=local_covariance(GaussNewtonMAP(estimator),result['parameters'],fit_status=result);task=public_position_jacobian(estimator,result['parameters'],commands)
 variance=predictive_variance(task['position_jacobian_m'],cov['covariance'],0.)
 return dict(mean=task['predicted_displacement_m'],sd=np.sqrt(variance),near_bound_parameters=cov['near_bound_parameters'],limitations=cov['limitations'],precision_condition=cov['precision_condition'])

def worker(index):
 torch.set_num_threads(1);plan=guard();folder=OUT/f'world{index}';public=json.loads((folder/'public.json').read_text());points=np.asarray(public['points']);poses=np.asarray(public['observed_poses'])
 with np.load(folder/'initial.npz')as data:initial_obs=dict(pose=poses[0],points=points,displacement=data['displacement'].copy())
 estimator=build(GeometrySpec(**public['geometry']),public['categories'],[initial_obs],True);initial=fit(estimator,None,plan);write(folder/'initial-fit.json',serial(initial))
 commands=[dict(name=f'held{j}',pose=np.asarray(p),points=points)for j,p in enumerate(public['held_poses'])]
 covariance=local_covariance(GaussNewtonMAP(estimator),initial['parameters'],fit_status=initial)
 task=public_position_jacobian(estimator,initial['parameters'],commands)
 candidates={a:public_position_jacobian(estimator,initial['parameters'],[dict(name=a,pose=poses[j],points=points)])['whitened_jacobian']for j,a in enumerate(ACTIONS,1)}
 ranking=rank_position_probes(covariance['covariance'],task['position_jacobian_m'],candidates)
 random=np.random.default_rng(plan['seed']).choice(ACTIONS,size=2)[index]
 selections={**{'fixed_'+a:a for a in ACTIONS},'random':str(random),'target_task_variance':ranking[0]['action']}
 # Durable selection precedes opening ANY candidate observation file.
 write(folder/'selection.json',dict(plan_sha256=sha(PLAN),policies=selections,ranking=ranking,initial_fit_sha256=sha(folder/'initial-fit.json')))
 outputs={'initial':prediction(estimator,initial,commands)};fits={}
 for j,a in enumerate(ACTIONS,1):
  try:
   with np.load(folder/(a+'.npz'))as data:observed=dict(pose=poses[j],points=points,displacement=data['displacement'].copy())
   after=build(GeometrySpec(**public['geometry']),public['categories'],[initial_obs,observed],True);result=fit(after,initial['parameters'],plan);fits[a]=serial(result);outputs[a]=prediction(after,result,commands)
  except Exception as error:fits[a]=dict(incomplete=True,optimizer_success=False,error=repr(error));outputs[a]=dict(error=repr(error))
  write(folder/'fits.json',fits)
 arrays={}
 for a,value in outputs.items():
  if 'mean'in value:arrays[a+'_mean']=value['mean'];arrays[a+'_sd']=value['sd']
 npz(folder/'forecasts.npz',**arrays)
 write(folder/'complete.json',dict(plan_sha256=sha(PLAN),selection_sha256=sha(folder/'selection.json'),forecast_sha256=sha(folder/'forecasts.npz'),fits_sha256=sha(folder/'fits.json'),diagnostics=serial({k:{j:v for j,v in x.items()if j not in ('mean','sd')}for k,x in outputs.items()})))

def score():
 plan=guard();commits={};invalid={}
 for i in (0,1):
  folder=OUT/f'world{i}'
  if not (folder/'complete.json').exists():
   if not (OUT/'run-status.json').exists():raise ValueError('All workers must be terminal before scoring')
   commits[str((OUT/'run-status.json').relative_to(ROOT))]=sha(OUT/'run-status.json');continue
  status=json.loads((folder/'complete.json').read_text())
  selection=json.loads((folder/'selection.json').read_text())
  if status['plan_sha256']!=sha(PLAN) or selection['plan_sha256']!=sha(PLAN):raise ValueError('Completion/selection plan mismatch')
  policies=selection['policies']
  if set(policies)!=set(plan['policies']) or any(a not in ACTIONS for a in policies.values()) or any(policies['fixed_'+a]!=a for a in ACTIONS):raise ValueError('Exact five-policy mapping required')
  invalid[i]=set()
  with np.load(folder/'forecasts.npz',allow_pickle=False)as values:
   if set(values.files)-{a+s for a in ('initial',*ACTIONS)for s in ('_mean','_sd')}:raise ValueError('Undeclared forecast keys')
   for a in ('initial',*ACTIONS):
    if any(a+k not in values for k in ('_mean','_sd')) or any(values[a+k].shape!=(156,) or not np.isfinite(values[a+k]).all()for k in ('_mean','_sd')) or (values[a+'_sd']<0).any():invalid[i].add(a)
  for file,key in [('forecasts.npz','forecast_sha256'),('selection.json','selection_sha256'),('fits.json','fits_sha256')]:
   if sha(folder/file)!=status[key]:raise ValueError('Forecast commitment changed')
  commits[str((folder/'complete.json').relative_to(ROOT))]=sha(folder/'complete.json')
 rows=[]
 for i in (0,1):
  folder=OUT/f'world{i}'
  if not (folder/'complete.json').exists():
   rows.extend(dict(world=i,policy=policy,command=j,state='failed',error='Worker did not commit all forecasts within bounded stage')for policy in ['initial',*json.loads(PLAN.read_text())['policies']]for j in (0,1));continue
  selection=json.loads((folder/'selection.json').read_text())['policies']
  with np.load(folder/'forecasts.npz')as pred,np.load(BASE/f'full-continuous-inference-pilot/world{i}/observations-and-development-truth.npz')as truth:
   for policy,a in {'initial':'initial',**selection}.items():
    for j in (0,1):
     row=dict(world=i,policy=policy,action=a,command=j)
     if a in invalid[i]:row.update(state='failed',error='Missing/nonfinite/malformed forecast or negative standard deviation')
     else:
      target=truth[f'future_{j}_displacement'].ravel();n=len(target);mean=pred[a+'_mean'][j*n:(j+1)*n];sd=pred[a+'_sd'][j*n:(j+1)*n];error=mean-target
      row.update(state='complete',position_coordinate_rmse_m=float(np.sqrt(np.mean(error**2))),nominal90_coordinate_coverage=float(np.mean(abs(error)<=1.6448536269514722*sd)),mean_band_halfwidth_m=float(np.mean(1.6448536269514722*sd)))
     rows.append(row)
 write(REPORT/'local-action-development-results.json',dict(plan_sha256=sha(PLAN),commit_sha256=commits,rows=rows,scope='Two exposed development objects; approximate local Gaussian intervals with zero model discrepancy, not calibrated; all fixed controls retained.'))

def run():
 plan=guard()
 with (OUT/'run-start.json').open('x')as marker:json.dump(dict(plan_sha256=sha(PLAN),started_unix=time.time()),marker)
 start=time.monotonic();jobs=[]
 for i in (0,1):
  stream=(OUT/f'world{i}.log').open('w');p=subprocess.Popen([sys.executable,'-m',__name__ if __name__!='__main__'else 'scripts.compositional_neural_mechanics.local_action_pilot','--worker',str(i)],stdout=stream,stderr=subprocess.STDOUT,start_new_session=True);jobs.append((p,stream))
 for p,_ in jobs:
  try:p.wait(timeout=max(.01,4000-(time.monotonic()-start)))
  except subprocess.TimeoutExpired:break
 for p,stream in jobs:
  if p.poll()is None:os.killpg(p.pid,signal.SIGKILL);p.wait()
  stream.close()
 write(OUT/'run-status.json',dict(exit_codes=[p.returncode for p,_ in jobs],seconds=time.monotonic()-start))
 score()
if __name__=='__main__':
 p=argparse.ArgumentParser();g=p.add_mutually_exclusive_group(required=True);g.add_argument('--prepare',action='store_true');g.add_argument('--run',action='store_true');g.add_argument('--worker',type=int,choices=[0,1]);a=p.parse_args()
 if a.prepare:prepare()
 elif a.run:run()
 else:worker(a.worker)
