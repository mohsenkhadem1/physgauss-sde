"""Public z=[0,0] MR/Yeoh cost profile; no fresh-cohort latent draws."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import time,signal,subprocess,json,sys
from pathlib import Path
import numpy as np
import torch
from scripts.compositional_dynamics_v17.prototype import ROOT,REPORT,sha,write
from scripts.compositional_dynamics_v17.dynamics import DynamicContactModel,VelocityVerlet
from scripts.compositional_neural_mechanics.mechanics import GeometrySpec
from scripts.compositional_neural_mechanics.contact_mechanics import RigidPlane
from scripts.compositional_neural_mechanics.materials import teacher_energy
from scripts.compositional_neural_mechanics.gravity_inference_pilot import closure

def main():
 signal.signal(signal.SIGALRM,lambda *_: (_ for _ in ()).throw(TimeoutError('120second teacher profile cap')));signal.alarm(120);torch.set_num_threads(1);census=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True);gpu=[l.split(',')for l in census.splitlines()if int(l.split(',')[0])==0][0]
 if int(gpu[1])<8192 or int(gpu[2])>20:raise ValueError('GPU0 lacks profiling headroom')
 rows=[]
 for family in ['mooney_rivlin','yeoh']:
  for n in [4,6]:
   values={}
   for device in ['cpu','cuda:0']:
    start=time.monotonic();m=DynamicContactModel(GeometrySpec(dimensions=(.12,.06,.06)),n=n,order=3,support_mode='free',surface_order=8,bulk_energy=lambda F,z:teacher_energy(F,z,family),material_latent_field=lambda q:q.new_tensor([0.,0.]).expand(*q.shape[:-1],2),gravity=(0,0,-9.8),density=625,colliders=(RigidPlane(point=(0,0,-.07),penalty_N_m3=4e7,name='floor'),),device=device);e=VelocityVerlet(m);setup=time.monotonic()-start;u=np.zeros_like(m.rest);u[:,2]=-.2*m.rest[:,2]-.05;s=e.initial_state(u=u);values[device]=s;times=[]
    for _ in range(3):
     start=time.monotonic();e.step(s,25e-6);times.append(time.monotonic()-start)
    rows.append(dict(family=family,n=n,device=device,setup_seconds=setup,midpoint_public_state='Affine20%compression+translation into4mm planecontact; profilingonly',verlet_step_median_seconds=float(np.median(times))));del e,m
    if device!='cpu':torch.cuda.empty_cache()
   a,b=values['cpu'],values['cuda:0'];rows[-1]['cpu_gpu_agreement']=dict(force_max_abs_N=float(np.max(abs(a.force-b.force))),energy_abs_J=abs(a.potential-b.potential))
 paths=closure([Path(__file__)]);write(REPORT/'cohort-teacher-profile.json',dict(source_sha256={str(p.relative_to(ROOT)):sha(p)for p in paths},python=sys.version,torch_version=torch.__version__,numpy_version=np.__version__,census=census,known_public_latent=[0.,0.],rows=rows,scope='Exact family operator cost/roundoff only; no freshprivateparameters orcohortgeneration. Threeisolatedsame-stateVerletsteps, notqualifiedtrajectory.'))
if __name__=='__main__':main()
