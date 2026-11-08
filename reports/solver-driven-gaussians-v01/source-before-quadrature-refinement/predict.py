"""Changed-control forward prediction; accepts fit seal, never target states."""
import argparse,json,time,hashlib
from pathlib import Path
import numpy as np
import torch
from scripts.solver_driven_gaussians.dynamics import DynamicFEM

def main():
 p=argparse.ArgumentParser();p.add_argument('--fit',required=True);p.add_argument('--output',required=True);p.add_argument('--gpu',default='0');p.add_argument('--dt',type=float,default=.0001);a=p.parse_args()
 torch.set_num_threads(2);fit=json.loads(Path(a.fit).read_text());assert fit['completed'] and not fit['future_positions_read'];m=DynamicFEM(device='cuda:'+a.gpu);out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
 for case in ['calibration','side_poke','early_release','force_pulse','higher_drop']:
  if (out/(case+'.json')).exists():continue
  start=time.time();r=m.rollout(fit['parameters'],case=case,kind=fit['kind'],dt=a.dt)
  np.savez_compressed(out/(case+'.npz'),x=r['x'].detach().cpu().numpy(),v=r['v'].detach().cpu().numpy(),times=r['times'])
  rec=dict(completed=True,case=case,minimum_J=r['minimum_J'],max_penetration_m=r['max_penetration_m'],seconds=time.time()-start,fit_sha256=hashlib.sha256(Path(a.fit).read_bytes()).hexdigest(),prediction_sha256=hashlib.sha256((out/(case+'.npz')).read_bytes()).hexdigest(),target_states_read=False,dt=a.dt)
  (out/(case+'.json')).write_text(json.dumps(rec,indent=2));print(rec,flush=True)
if __name__=='__main__':main()
