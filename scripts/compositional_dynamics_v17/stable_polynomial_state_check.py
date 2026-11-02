"""Read-only pointwise operator comparison; no integration or frozen-source changes."""
import json
from pathlib import Path
import numpy as np
import torch
from scripts.compositional_dynamics_v17.prototype import ROOT,REPORT,sha,write,DynamicContactModel,GeometrySpec,RigidPlane,StableNeoHookean
from scripts.compositional_dynamics_v17.stable_polynomial import StableNeoHookeanPolynomial

def main():
 torch.set_num_threads(1);rows=[];inputs={}
 for case in ['release','drop']:
  path=ROOT/f'runs/compositional-dynamics-v17/prototype/{case}-n2-dt0.00005000/trajectory.npz';inputs[str(path.relative_to(ROOT))]=sha(path)
  with np.load(path)as d:
   indices=sorted(set([0,len(d['u'])//2,len(d['u'])-1,int(np.argmax(np.linalg.norm(d['u'],axis=(1,2))))]))
   m=DynamicContactModel(GeometrySpec(dimensions=(.12,.06,.06)),n=2,order=3,support_mode='free',bulk_energy=StableNeoHookean(12000,.35),gravity=(0,0,-9.81)if case=='drop'else(0,0,0),density=625,colliders=(RigidPlane(point=(0,0,-.05),penalty_N_m3=4e7,name='floor'),)if case=='drop'else(),surface_order=8)
   for i in indices:
    m.bulk_energy=StableNeoHookean(12000,.35);a=m.evaluate(d['u'][i]);m.bulk_energy=StableNeoHookeanPolynomial(12000,.35);b=m.evaluate(d['u'][i])
    df=np.max(np.abs(a['residual']-b['residual']));de=abs(a['potential_J']-b['potential_J']);rows.append(dict(case=case,frame=i,time_s=float(d['times_s'][i]),force_max_abs_N=float(df),energy_abs_J=float(de),passed=bool(df<1e-10 and de<1e-12)))
 paths=[Path(__file__),ROOT/'scripts/compositional_dynamics_v17/stable_polynomial.py',ROOT/'scripts/compositional_dynamics_v17/test_stable_polynomial.py']
 write(REPORT/'stable-polynomial-state-check.json',dict(source_sha256={str(p.relative_to(ROOT)):sha(p)for p in paths},input_sha256=inputs,rows=rows,passed=all(r['passed']for r in rows),scope='Pointwise same saved states only; additive future higher-derivative operator. No frozen operator replacement or new trajectories.'))
if __name__=='__main__':main()
