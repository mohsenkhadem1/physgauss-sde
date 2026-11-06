"""Exact dynamic scale ambiguity when mass is unknown; known force breaks it.

This is a constructed two-hypothesis observability test, not a calibrated posterior
or a general active-policy benchmark. All hypotheses are simulated before reading
the probing observation or separate force-reversal evaluation trajectory.
"""
import argparse,json,time,hashlib
from pathlib import Path
import numpy as np
import torch
from scripts.solver_driven_gaussians.dynamics import DynamicFEM

def main():
 p=argparse.ArgumentParser();p.add_argument('--gpu',default='1');a=p.parse_args();torch.set_num_threads(2)
 base=Path('runs/solver-driven-gaussians-v02');out=base/'ambiguity';out.mkdir(exist_ok=True)
 while not (base/'selected-learned-fit.json').exists():time.sleep(10)
 fit=json.loads((base/'selected-learned-fit.json').read_text());states={}
 for factor in [1.,2.]:
  m=DynamicFEM(density=625*factor,device='cuda:'+a.gpu);m.floor_penalty*=factor;theta=np.array(fit['parameters']);theta[:10]*=factor
  for case in ['calibration','side_poke','force_pulse','force_reverse']:
   result=m.rollout(theta,case=case,dt=.0001);x=result['x'].cpu().numpy();states[(factor,case)]=x;np.savez_compressed(out/f'factor{int(factor)}-{case}.npz',x=x,times=result['times']);print(factor,case,flush=True)
 ambiguity={case:float(np.sqrt(np.mean(np.sum((states[(1.,case)]-states[(2.,case)])**2,-1)))*1000) for case in ['calibration','side_poke','force_pulse']}
 assert np.max(abs(states[(1.,'calibration')]-states[(2.,'calibration')]))<1e-10
 action=max(['side_poke','force_pulse'],key=lambda k:ambiguity[k]);assert action=='force_pulse'
 seal=dict(candidate_disagreement_mm=ambiguity,selected_probe=action,new_view_disagreement_mm=0.,reason='Identical geometry/covariance trajectories under a common mass/stiffness/attachment/contact scaling; views and position-controlled actions cannot resolve this gauge. A calibrated force does not scale.',fit_sha256=hashlib.sha256((base/'selected-learned-fit.json').read_bytes()).hexdigest(),probe_targets_read=False,prediction_hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in out.glob('factor*.npz')})
 (out/'probe-selection.json').write_text(json.dumps(seal,indent=2));print(seal['candidate_disagreement_mm'],flush=True)
if __name__=='__main__':main()
