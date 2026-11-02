"""Bounded CPU cost guard at higher mesh resolution, no time integration."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import time,json,signal
from pathlib import Path
import numpy as np
import torch
from scripts.compositional_dynamics_v17.prototype import ROOT,REPORT,PLAN,sha,write,verify
from scripts.compositional_dynamics_v17.dynamics import DynamicContactModel,VelocityVerlet
from scripts.compositional_dynamics_v17.materials import StableNeoHookean
from scripts.compositional_neural_mechanics.mechanics import GeometrySpec
from scripts.compositional_neural_mechanics.contact_mechanics import RigidPlane

def main():
 signal.signal(signal.SIGALRM,lambda *_: (_ for _ in ()).throw(TimeoutError('120second CPU profile cap')));signal.alarm(120);torch.set_num_threads(1);p=json.loads(PLAN.read_text());verify(p);rows=[]
 for n in [6,8]:
  start=time.monotonic();m=DynamicContactModel(GeometrySpec(dimensions=(.12,.06,.06)),n=n,order=3,support_mode='free',surface_order=8,bulk_energy=StableNeoHookean(12000,.35),density=625,gravity=(0,0,-9.81),colliders=(RigidPlane(point=(0,0,-.05),penalty_N_m3=4e7,name='floor'),));engine=VelocityVerlet(m);setup=time.monotonic()-start;u=np.zeros_like(m.rest);u[:,2]=-.2*m.rest[:,2]-.03
  times=[];mass=[]
  for _ in range(4):
   start=time.monotonic();out=engine.evaluate(u,{});times.append(time.monotonic()-start);start=time.monotonic();engine.acceleration(out[0]);mass.append(time.monotonic()-start)
  rows.append(dict(n=n,nodes=len(m.rest),setup_seconds=setup,force_median_seconds=float(np.median(times)),mass_median_seconds=float(np.median(mass)),drop25us_estimate_seconds=10000*float(np.median(times)+np.median(mass)),minimum_J=out[4]['minimum_J']))
 write(REPORT/'fine-profile.json',dict(source_sha256=sha(__file__),prototype_plan_sha256=sha(PLAN),rows=rows,scope='Same-state CPU timing only, not dynamic stability qualification. Original constitutive source unchanged.'))
if __name__=='__main__':main()
