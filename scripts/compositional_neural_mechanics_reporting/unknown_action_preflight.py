"""CPU timing/numerical check on existing exposed development world0 only."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import json,time,signal
import numpy as np
import torch
from scripts.compositional_neural_mechanics.full_continuous_inference_pilot import build,pack
from scripts.compositional_neural_mechanics.conditional_inference_pilot import predict
from scripts.compositional_neural_mechanics.mechanics import GeometrySpec
from scripts.compositional_neural_mechanics.local_action_pilot import ROOT,REPORT,BASE,write
from scripts.compositional_neural_mechanics_reporting.legacy_dependency_audit import sha,closure
from pathlib import Path

def main():
 torch.set_num_threads(1);target=REPORT/'unknown-action-development-preflight.json'
 if target.exists():raise ValueError('Preserve preflight')
 source=REPORT/'full-continuous-inference-pilot-world0.json';archive=BASE/'full-continuous-inference-pilot/world0/observations-and-development-truth.npz';prior=json.loads(source.read_text())
 plan=dict(scope=__doc__,n=4,order=3,per_solve_seconds=300,source_sha256=closure([Path(__file__).resolve()]),input_sha256={str(p.relative_to(ROOT)):sha(p)for p in (source,archive)})
 write(REPORT/'unknown-action-development-preflight-plan.json',plan)
 def timeout(*args):raise TimeoutError('Bounded300s development solve')
 signal.signal(signal.SIGALRM,timeout);rows=[]
 with np.load(archive)as raw:
  commands=[raw['observed_poses'][0],raw['held_poses'][0]]
  teacher=build(GeometrySpec(**prior['scene']['geometry']),prior['categories_conditioned'],[dict(pose=commands[0],points=raw['points'],displacement=np.zeros_like(raw['points']))],False);teacher.kw['n']=4
  for index,pose in enumerate(commands):
   start=time.monotonic();signal.setitimer(signal.ITIMER_REAL,300)
   try:
    value=predict(teacher,pack(prior['scene']),pose)
    row=dict(command=index,state='complete',seconds=time.monotonic()-start,finite=all(np.isfinite(x).all()for x in value.values()))
    if index==1:row['n2_n4_coordinate_difference_m']=float(np.sqrt(np.mean((value['displacement']-raw['future_0_displacement'])**2)))
   except Exception as error:row=dict(command=index,state='failed',error=repr(error),seconds=time.monotonic()-start)
   finally:signal.setitimer(signal.ITIMER_REAL,0)
   rows.append(row);write(target,dict(plan=plan,rows=rows,scope='Development timing/numerical discrepancy only; no new scene or inference tuning.'))
if __name__=='__main__':main()
