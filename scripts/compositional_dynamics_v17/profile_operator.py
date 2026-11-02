"""Bounded same-state CPU/GPU1 profile; no dynamic rollout or model changes."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import time,json,subprocess,signal,csv,io,sys
from pathlib import Path
import numpy as np
import torch
from scripts.compositional_dynamics_v17.prototype import ROOT,REPORT,PLAN,sha,write,verify
from scripts.compositional_dynamics_v17.dynamics import DynamicContactModel,VelocityVerlet
from scripts.compositional_dynamics_v17.materials import StableNeoHookean
from scripts.compositional_neural_mechanics.mechanics import GeometrySpec
from scripts.compositional_neural_mechanics.contact_mechanics import RigidPlane

def main():
 def timeout(*a):raise TimeoutError('90second profile bound')
 signal.signal(signal.SIGALRM,timeout);signal.alarm(90);torch.set_num_threads(1);p=json.loads(PLAN.read_text());verify(p)
 census=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True);gpu=[a for a in csv.reader(io.StringIO(census))if int(a[0])==1][0]
 if int(gpu[1])<8192 or int(gpu[2])>10:raise ValueError('GPU1 currently lacks profiling headroom')
 output=dict(python_version=sys.version,torch_version=torch.__version__,numpy_version=np.__version__,compiler=os.environ.get('CC'),source_sha256=sha(__file__),prototype_plan_sha256=sha(PLAN),census=census,scope='Same-state stableNH force/mass profile only; no timestep simulation or new physicsqualification',rows=[])
 for n in [2,4]:
  states={}
  for device in ['cpu','cuda:1']:
   start=time.monotonic();m=DynamicContactModel(GeometrySpec(dimensions=(.12,.06,.06)),n=n,order=3,support_mode='free',surface_order=8,bulk_energy=StableNeoHookean(),gravity=(0,0,-9.81),colliders=(RigidPlane(point=(0,0,-.05),penalty_N_m3=4e7,name='floor'),),device=device);engine=VelocityVerlet(m);setup=time.monotonic()-start;u=np.zeros_like(m.rest);u[:,2]=-.2*m.rest[:,2]-.03
   force,*_=engine.evaluate(u,{})
   times=[];mass=[]
   for _ in range(10):
    start=time.monotonic();result=engine.evaluate(u,{});times.append(time.monotonic()-start);start=time.monotonic();engine.acceleration(result[0]);mass.append(time.monotonic()-start)
   states[device]=result;output['rows'].append(dict(n=n,device=device,nodes=len(m.rest),setup_seconds=setup,force_median_seconds=float(np.median(times)),mass_solve_median_seconds=float(np.median(mass)),potential_J=result[3],contact_force=[r['force_on_rigid_N'].tolist()for r in result[4]['contacts']]))
   del engine,m
   if device!='cpu':torch.cuda.empty_cache()
  a,b=states['cpu'],states['cuda:1'];output['rows'][-1]['cpu_gpu_agreement']=dict(force_max_abs_N=float(np.max(abs(a[0]-b[0]))),force_relative_L2=float(np.linalg.norm(a[0]-b[0])/np.linalg.norm(a[0])),energy_absolute_J=abs(a[3]-b[3]))
 verify(p);write(REPORT/'operator-profile.json',output)
if __name__=='__main__':main()
