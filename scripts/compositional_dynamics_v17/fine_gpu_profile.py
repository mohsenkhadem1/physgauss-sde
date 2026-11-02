"""Same-state n6/n8 CPU/GPU1 profile only; no trajectory rerouting."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import time,json,signal,sys,subprocess
from pathlib import Path
import numpy as np
import torch
from scripts.compositional_dynamics_v17.prototype import ROOT,REPORT,PLAN,sha,write,verify
from scripts.compositional_dynamics_v17.dynamics import DynamicContactModel,VelocityVerlet
from scripts.compositional_dynamics_v17.materials import StableNeoHookean
from scripts.compositional_neural_mechanics.mechanics import GeometrySpec
from scripts.compositional_neural_mechanics.contact_mechanics import RigidPlane

def main():
 signal.signal(signal.SIGALRM,lambda *_: (_ for _ in ()).throw(TimeoutError('120second profile cap')));signal.alarm(120);torch.set_num_threads(1);p=json.loads(PLAN.read_text());verify(p)
 census=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True);g=[l.split(',')for l in census.splitlines()if int(l.split(',')[0])==1][0]
 if int(g[1])<8192 or int(g[2])>15:raise ValueError('GPU1 insufficient profile headroom')
 out=dict(scope='Read-only force/mass profile, frozen running CPU trajectories unchanged',python=sys.version,torch_version=torch.__version__,numpy_version=np.__version__,census=census,source_sha256=sha(__file__),prototype_plan_sha256=sha(PLAN),rows=[])
 for n in [6,8]:
  values={}
  for device in ['cpu','cuda:1']:
   start=time.monotonic();m=DynamicContactModel(GeometrySpec(dimensions=(.12,.06,.06)),n=n,order=3,support_mode='free',surface_order=8,bulk_energy=StableNeoHookean(12000,.35),density=625,gravity=(0,0,-9.81),colliders=(RigidPlane(point=(0,0,-.05),penalty_N_m3=4e7,name='floor'),),device=device);e=VelocityVerlet(m);setup=time.monotonic()-start;u=np.zeros_like(m.rest);u[:,2]=-.2*m.rest[:,2]-.03;f=e.evaluate(u,{})
   times=[];mass=[]
   for _ in range(3):
    start=time.monotonic();f=e.evaluate(u,{});times.append(time.monotonic()-start);start=time.monotonic();e.acceleration(f[0]);mass.append(time.monotonic()-start)
   values[device]=f;out['rows'].append(dict(n=n,device=device,nodes=len(m.rest),setup_seconds=setup,force_median_seconds=float(np.median(times)),mass_median_seconds=float(np.median(mass)),total_step_estimate_seconds=float(np.median(times)+np.median(mass))))
   del e,m
   if device!='cpu':torch.cuda.empty_cache()
  a,b=values['cpu'],values['cuda:1'];out['rows'][-1]['agreement']=dict(force_max_abs_N=float(np.max(abs(a[0]-b[0]))),force_relative_L2=float(np.linalg.norm(a[0]-b[0])/np.linalg.norm(a[0])),energy_abs_J=float(abs(a[3]-b[3])))
 verify(p);write(REPORT/'fine-gpu-profile.json',out)
if __name__=='__main__':main()
