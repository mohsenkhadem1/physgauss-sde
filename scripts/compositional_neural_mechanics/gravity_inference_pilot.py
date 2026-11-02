"""Known-gravity position-only DEVELOPMENT pilot; preserves zero-gravity pilot."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import sys,ast,time,signal,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from scripts.compositional_neural_mechanics.conditional_inference_pilot import problem,predict,POINTS,GEOMETRY,BULK,INTERFACE,sha,write
from scripts.compositional_neural_mechanics.joint_inference import neutral_parameters
R=ROOT/'reports/compositional-neural-mechanics-v16';OUT=ROOT/'runs/compositional-neural-mechanics-v16/gravity-inference-pilot'
def closure(start):
 seen=set();todo=list(start)
 while todo:
  p=todo.pop().resolve()
  if p in seen:continue
  seen.add(p)
  for node in ast.walk(ast.parse(p.read_text())):
   names=[node.module] if isinstance(node,ast.ImportFrom) and node.module else [a.name for a in node.names]if isinstance(node,ast.Import)else []
   for name in names:
    if name.split('.')[0] not in ('scripts','experiments'):continue
    q=ROOT/(name.replace('.','/')+'.py')
    if q.exists():todo.append(q)
 return sorted(seen)
def make(observations,learned=False):
 p=problem(observations,learned);p.scene['gravity']=[0.,0.,-9.81];p.scene['density_kg_m3']=125.;return p

def main():
 torch.set_num_threads(1)
 if OUT.exists():raise ValueError('Preserve prior gravity pilot')
 sources=closure([Path(__file__)])+[BULK,INTERFACE]
 plan=dict(scope='Known density/gravity, position-only conditional DEVELOPMENT inference; same n2 analytic teacher, no reserved cases',seed=926607,gravity_m_s2=[0.,0.,-9.81],density_kg_m3=125.,geometry=GEOMETRY.__dict__,n=2,points=POINTS.tolist(),active=[0,3,5],truth=[.25,350.,800.],known_nuisance='shape0,gradient0,center.5,width.3,strength.5,MR/quartic/h0',position_noise_m=50e-6,force_observation_count=0,profile_alphas=np.linspace(-.6,.6,9).tolist(),profile='Actual equilibrium solves at every alpha and both identification poses; no scaling shortcut',fit={'iterations':30,'evaluations':80,'fit_seconds':300,'whole_seconds':1200,'prior':'neutral independent unit Gaussian on packed descriptors'},source_sha256={str(p.relative_to(ROOT)):sha(p) for p in sources})
 write(R/'gravity-inference-pilot-plan.json',plan);OUT.mkdir(parents=True)
 start=time.monotonic();deadline=start+1200
 def timeout(*args):raise TimeoutError('Gravity pilot external time budget')
 signal.signal(signal.SIGALRM,timeout);signal.setitimer(signal.ITIMER_REAL,1200)
 poses=[]
 a=np.zeros((2,6));a[0,0]=-.008;a[1,0]=.008;poses.append(a)
 a=np.zeros((2,6));a[1,5]=.25;poses.append(a)
 held=np.zeros((2,6));held[1,4]=.30
 dummy=[dict(pose=a,points=POINTS,displacement=np.zeros_like(POINTS))for a in poses]
 teacher=make(dummy);truth=neutral_parameters();truth[[0,3,5]]=[.25,np.log(350.),np.log(800.)]
 clean=[predict(teacher,truth,a)for a in poses];future=predict(teacher,truth,held)
 rng=np.random.default_rng(926607)
 observations=[dict(pose=a,points=POINTS,displacement=c['displacement']+rng.normal(size=c['displacement'].shape)*50e-6)for a,c in zip(poses,clean)]
 assert all(set(o)=={'pose','points','displacement'} for o in observations)
 np.savez_compressed(OUT/'observations-and-development-truth.npz',truth=truth,poses=poses,points=POINTS,heldout_pose=held,**{f'obs_{i}':o['displacement']for i,o in enumerate(observations)},**{f'future_{k}':v for k,v in future.items()})
 report=dict(plan_sha256=sha(R/'gravity-inference-pilot-plan.json'),scope=plan['scope'],profile=[],fits=[],force_observation_count=0)
 for alpha in plan['profile_alphas']:
  z=truth.copy();z[[0,3,5]]+=alpha;loss=0.;state_shift=[]
  for i,(pose,observation)in enumerate(zip(poses,observations)):
   prediction=predict(teacher,z,pose);loss+=.5*float(np.sum(((prediction['displacement']-observation['displacement'])/50e-6)**2));state_shift.append(float(np.sqrt(np.mean((prediction['displacement']-clean[i]['displacement'])**2))))
  report['profile'].append(dict(alpha=alpha,position_likelihood=loss,prior=.5*float(np.sum((z-neutral_parameters())**2)),position_shift_from_truth_rms_m=state_shift))
  write(R/'gravity-inference-pilot-results.json',report)
 for learned in (False,True):
  fitproblem=make(observations,learned);remaining=deadline-time.monotonic();row=dict(model='learned' if learned else 'analytic',track='V_known_gravity',force_observation_count=0)
  if remaining<=1:row.update(incomplete=True,error='Whole pilot deadline');report['fits'].append(row);continue
  signal.setitimer(signal.ITIMER_REAL,min(300,remaining))
  try:
   answer=fitproblem.fit(active=[0,3,5],max_iterations=30,max_evaluations=80,seconds=min(299,remaining));row.update({k:v.tolist()if isinstance(v,np.ndarray)else v for k,v in answer.items()})
  except Exception as error:row.update(incomplete=True,optimizer_success=False,error=f'{type(error).__name__}: {error}')
  signal.setitimer(signal.ITIMER_REAL,max(.01,deadline-time.monotonic()))
  if 'parameters'in row:
   try:
    forecast=predict(fitproblem,row['parameters'],held);row['withheld_error']={k:float(np.sqrt(np.mean((forecast[k]-future[k])**2)))for k in future};np.savez_compressed(OUT/f"forecast-{row['model']}.npz",**forecast)
   except Exception as error:row['forecast_failure']=f'{type(error).__name__}: {error}'
  report['fits'].append(row);write(R/'gravity-inference-pilot-results.json',report);print(row['model'],row.get('loss'),row.get('withheld_error'),flush=True)
 report['elapsed_seconds']=time.monotonic()-start;report['source_hashes_unchanged']=all(sha(ROOT/p)==h for p,h in plan['source_sha256'].items());write(R/'gravity-inference-pilot-results.json',report);signal.setitimer(signal.ITIMER_REAL,0)
if __name__=='__main__':main()
