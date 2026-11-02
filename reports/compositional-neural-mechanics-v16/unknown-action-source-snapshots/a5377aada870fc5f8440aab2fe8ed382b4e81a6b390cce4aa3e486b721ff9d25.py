"""Explicit development conditional mechanism pilot; no reserved scene reader."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import sys,json,hashlib,itertools,time,signal
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from scripts.compositional_neural_mechanics.joint_inference import JointMAP,neutral_parameters
from scripts.compositional_neural_mechanics.mechanics import GeometrySpec
from scripts.compositional_neural_mechanics.scenes import make_model
from scripts.compositional_neural_mechanics.scale_separated_material import ScaleSeparatedMaterialLaw
from scripts.compositional_neural_mechanics.interfaces import SharedInterfaceLaw
R=ROOT/'reports/compositional-neural-mechanics-v16';OUT=ROOT/'runs/compositional-neural-mechanics-v16/conditional-inference-pilot'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(x,indent=2)+'\n');tmp.replace(p)
def predict(problem,z,pose):
 model=make_model(problem.scene,**problem.kw);problem.bind(model,torch.tensor(z,dtype=torch.float64));u=None
 for a in problem.continuation:
  model.set_controls(pose*a);u=model.solve(initial=u,force_tolerance_N=1e-10,maximum_iterations=60)['displacement_local_m']
 _,_,_,(f,t)=model.interface_response(torch.tensor(u,dtype=torch.float64,requires_grad=True),create_graph=False)
 return dict(displacement=model.sample_material(u,POINTS),forces=torch.stack(f).detach().numpy(),torques=torch.stack(t).detach().numpy())
POINTS=np.array([p for p in itertools.product([0.,.5,1.],repeat=3) if p!=(.5,.5,.5)])
GEOMETRY=GeometrySpec((.20,.08,.08),.15,.006)
def problem(observations,learned=False):
 bulk=interface=None
 if learned:
  bulk=ScaleSeparatedMaterialLaw().double();bulk.load_state_dict(torch.load(BULK,weights_only=False)['state_dict'])
  interface=SharedInterfaceLaw();interface.load_state_dict(torch.load(INTERFACE,weights_only=False)['state_dict'])
 return JointMAP(GEOMETRY,bulk_family='mooney_rivlin',interface_family=0,heterogeneity=0,observations=observations,position_sigma=50e-6,force_sigma=.02,torque_sigma=.001,n=2,bulk_law=bulk,interface_law=interface)
BULK=ROOT/'runs/compositional-neural-mechanics-v16/scale-separated-prototype/shared-law.pt';INTERFACE=ROOT/'runs/compositional-neural-mechanics-v16/interface-wide-range-warm/shared-interface.pt'
def main():
 torch.set_num_threads(1)
 if OUT.exists():raise ValueError('Preserve previous pilot')
 files=list((ROOT/'scripts/compositional_neural_mechanics').glob('*.py'))+[ROOT/'experiments/campaign_static_p2_block.py',ROOT/'experiments/campaign_static_uniaxial_bar.py',BULK,INTERFACE]
 # Existing dependency filenames are pinned without assuming optional modules.
 files=[p for p in files if p.exists()]
 plan=dict(scope='DEVELOPMENT conditional three-parameter same-mesh validation only',seed=926601,geometry=GEOMETRY.__dict__,n=2,active=[0,3,5],truth={'logscale':.25,'left_K':350.,'right_K':800.},known_nuisance='neutral shape0,gradient0,centers.5,widths.3,strength.5; MR/quartic/h0',noise={'position_m':50e-6,'force_N':.02,'torque_Nm':.001},profile_alphas=np.linspace(-.6,.6,9).tolist(),profile='Exact co-scaling identity with explicit two endpoint equilibrium checks',fit={'max_iterations':30,'max_evaluations':80,'per_fit_seconds':300,'whole_pilot_seconds':1200,'prior':'independent unit Gaussian in parameter pack around neutral initialization'},source_sha256={str(p.relative_to(ROOT)):sha(p)for p in files})
 write(R/'conditional-inference-pilot-plan.json',plan);OUT.mkdir(parents=True)
 started=time.monotonic();deadline=started+1200
 def timeout(*a):raise TimeoutError('External development pilot/fit time cap')
 signal.signal(signal.SIGALRM,timeout);signal.setitimer(signal.ITIMER_REAL,1200)
 poses=[]
 p=np.zeros((2,6));p[0,0]=-.04*.20;p[1,0]=.04*.20;poses.append(p)
 p=np.zeros((2,6));p[1,5]=.25;poses.append(p)
 withheld=np.zeros((2,6));withheld[1,4]=.30
 dummy=[dict(pose=p,points=POINTS,displacement=np.zeros_like(POINTS))for p in poses]
 base=problem(dummy);truth=neutral_parameters();truth[[0,3,5]]=[.25,np.log(350.),np.log(800.)]
 clean=[predict(base,truth,p)for p in poses];future=predict(base,truth,withheld)
 rng=np.random.default_rng(926601);obs=[]
 for p,c in zip(poses,clean):
  o=dict(pose=p,points=POINTS)
  for key,sigma in [('displacement',50e-6),('forces',.02),('torques',.001)]:o[key]=c[key]+rng.normal(size=c[key].shape)*sigma
  obs.append(o)
 np.savez_compressed(OUT/'observations-and-development-truth.npz',points=POINTS,poses=poses,truth=truth,withheld_pose=withheld,**{f'obs_{i}_{k}':v for i,o in enumerate(obs)for k,v in o.items()},**{f'future_{k}':v for k,v in future.items()})
 report={'plan_sha256':sha(R/'conditional-inference-pilot-plan.json'),'scope':plan['scope'],'profile':[],'endpoint_checks':[],'fits':[]}
 for a in (-.6,.6):
  z=truth.copy();z[[0,3,5]]+=a
  for i,p in enumerate(poses):
   c=predict(base,z,p);report['endpoint_checks'].append(dict(alpha=a,pose=i,position_max_m=float(np.abs(c['displacement']-clean[i]['displacement']).max()),force_scaling_max_N=float(np.abs(c['forces']-np.exp(a)*clean[i]['forces']).max()),torque_scaling_max_Nm=float(np.abs(c['torques']-np.exp(a)*clean[i]['torques']).max())))
 for a in plan['profile_alphas']:
  losses={track:0. for track in ('V','VF')}
  for o,c in zip(obs,clean):
   v=.5*np.sum(((c['displacement']-o['displacement'])/50e-6)**2);vf=v+.5*np.sum(((np.exp(a)*c['forces']-o['forces'])/.02)**2)+.5*np.sum(((np.exp(a)*c['torques']-o['torques'])/.001)**2)
   losses['V']+=v;losses['VF']+=vf
  z=truth.copy();z[[0,3,5]]+=a
  report['profile'].append(dict(alpha=a,likelihood=losses,prior=.5*float(np.sum((z-neutral_parameters())**2))))
 write(R/'conditional-inference-pilot-results.json',report)
 for learned in (False,True):
  for track in ('V','VF'):
   current=[{k:v for k,v in o.items()if track=='VF' or k not in ('forces','torques')}for o in obs];fitproblem=problem(current,learned)
   row=dict(model='learned' if learned else 'analytic',track=track);remaining=deadline-time.monotonic()
   if remaining<=1:row.update(incomplete=True,error='Whole pilot deadline');report['fits'].append(row);continue
   signal.setitimer(signal.ITIMER_REAL,min(300,remaining))
   try:
    answer=fitproblem.fit(active=[0,3,5],max_iterations=30,max_evaluations=80,seconds=min(299,remaining));row.update({k:(v.tolist()if isinstance(v,np.ndarray)else v)for k,v in answer.items()})
   except Exception as e:row.update(incomplete=True,optimizer_success=False,error=f'{type(e).__name__}: {e}')
   signal.setitimer(signal.ITIMER_REAL,max(.01,deadline-time.monotonic()))
   if 'parameters' in row:
    try:
     prediction=predict(fitproblem,row['parameters'],withheld);row['withheld_error']={k:float(np.sqrt(np.mean((prediction[k]-future[k])**2)))for k in future}
     np.savez_compressed(OUT/f"forecast-{row['model']}-{track}.npz",**prediction)
    except Exception as e:row['forecast_failure']=f'{type(e).__name__}: {e}'
   report['fits'].append(row);write(R/'conditional-inference-pilot-results.json',report);print(row['model'],track,row.get('loss'),row.get('withheld_error'),flush=True)
 report['elapsed_seconds']=time.monotonic()-started;report['source_hashes_unchanged']=all(sha(ROOT/p)==h for p,h in plan['source_sha256'].items());write(R/'conditional-inference-pilot-results.json',report);signal.setitimer(signal.ITIMER_REAL,0)
if __name__=='__main__':main()
