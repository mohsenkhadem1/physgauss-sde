"""Bounded same-state viscous component and device cost, no production rerouting."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import signal,time,json,sys,subprocess
import numpy as np
import torch
from scripts.compositional_dynamics_v17 import viscous_development as pilot

def main():
 signal.signal(signal.SIGALRM,lambda *_: (_ for _ in ()).throw(TimeoutError('60second profile cap')));signal.alarm(60);torch.set_num_threads(1);p=json.loads(pilot.PLAN.read_text());pilot.verify(p);c=dict(p['cases'][1]);path=pilot.ROOT/'runs/compositional-dynamics-v17/stronger-development/finite-striker/trajectory.npz';d=np.load(path);idx=50;c['sphere_center_m']=d['sphere_centers_m'][idx,0].tolist();c['sphere_velocity_m_s']=d['sphere_velocities_m_s'][idx,0].tolist();census=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True);g=[l.split(',')for l in census.splitlines()if int(l.split(',')[0])==1][0]
 if int(g[1])<8192 or int(g[2])>20:raise ValueError('GPU1 lacks profile headroom')
 constructor=pilot.DynamicContactModel;rows=[];values={}
 for device in ['cpu','cuda:1']:
  def configured(*args,**kwargs):return constructor(*args,**kwargs,device=device)
  pilot.DynamicContactModel=configured;m,e,_=pilot.build(p,c);s=e.initial_state(u=d['u'][idx],v=d['velocities'][idx]);values[device]=s;times={k:[]for k in ['elastic_contact','viscous','mass','midpoint_step']}
  for _ in range(3):
   start=time.monotonic();e.evaluate(s.u,s.spheres);times['elastic_contact'].append(time.monotonic()-start)
   start=time.monotonic();e.viscous(s.u,s.v);times['viscous'].append(time.monotonic()-start)
   start=time.monotonic();e.acceleration(s.force);times['mass'].append(time.monotonic()-start)
   start=time.monotonic();e.step(s,25e-6);times['midpoint_step'].append(time.monotonic()-start)
  rows.append(dict(device=device,component_median_seconds={k:float(np.median(v))for k,v in times.items()}));del e,m
  if device!='cpu':torch.cuda.empty_cache()
 a,b=values['cpu'],values['cuda:1'];pilot.write(pilot.REPORT/'viscous-component-profile.json',dict(source_sha256=pilot.sha(__file__),plan_sha256=pilot.sha(pilot.PLAN),state_sha256=pilot.sha(path),state_frame=idx,python=sys.version,torch_version=torch.__version__,numpy_version=np.__version__,census=census,rows=rows,agreement=dict(force_max_abs_N=float(np.max(abs(a.force-b.force))),energy_abs_J=abs(a.potential-b.potential),power_abs_W=abs(a.diagnostics['viscous_power_W']-b.diagnostics['viscous_power_W'])),scope='Same-state component profiling only; viscosity remains NumPy/CPU and mass CPU even when elastic force uses GPU. Production untouched.'))
if __name__=='__main__':main()
