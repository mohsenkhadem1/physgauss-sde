"""Generator-only entry point; hidden parameters never exported as observations."""
import os
os.environ.setdefault('OMP_NUM_THREADS','2')
import argparse,json,time,hashlib
from pathlib import Path
import numpy as np
import torch
from scripts.solver_driven_gaussians.dynamics import DynamicFEM

def reference_parameters():
    return np.array([1.35,1500.,1500.,1500.,1500.,0.,0.,0.,0.,350.,2.5])

def main():
    p=argparse.ArgumentParser();p.add_argument('--gpu',default='1');p.add_argument('--fine',action='store_true');p.add_argument('--resolution',type=int);p.add_argument('--dt',type=float,default=.0001);p.add_argument('--output');p.add_argument('--cases',nargs='+');a=p.parse_args()
    n=a.resolution or (16 if a.fine else 8)
    torch.set_num_threads(2);torch.cuda.set_device(int(a.gpu));m=DynamicFEM(cells=(n,n//2,n//2),device='cuda:'+a.gpu)
    out=Path(a.output) if a.output else Path('runs/solver-driven-gaussians-v02')/('reference-fine' if a.fine else 'reference');out.mkdir(exist_ok=True)
    for case in (a.cases or (['calibration','higher_drop'] if a.fine else ['calibration','side_poke','early_release','force_pulse','higher_drop'])):
        if (out/(case+'.npz')).exists():continue
        start=time.time();r=m.rollout(reference_parameters(),case=case,kind='reference',dt=a.dt)
        np.savez_compressed(out/(case+'.npz'),x=r['x'].cpu().numpy(),v=r['v'].cpu().numpy(),times=r['times'],rest=m.rest.cpu().numpy(),elements=m.elements.cpu().numpy())
        report=dict(case=case,completed=True,seconds=time.time()-start,minimum_J=r['minimum_J'],max_penetration_m=r['max_penetration_m'],dt=a.dt,cells=m.cells,sha256=hashlib.sha256((out/(case+'.npz')).read_bytes()).hexdigest())
        (out/(case+'.json')).write_text(json.dumps(report,indent=2));print(report,flush=True)
        if case=='calibration' and not a.fine and not a.output:
            keep=r['times']<=.14+1e-9;obs=out.parent/'observations';obs.mkdir(exist_ok=True)
            np.savez_compressed(obs/'calibration.npz',x=r['x'][keep].cpu().numpy(),times=r['times'][keep],rest=m.rest.cpu().numpy())
            (obs/'scope.json').write_text(json.dumps(dict(input='Calibration position trajectories only, before release; no material/force labels or future tracks.',geometry='Known block volume and mass density in this qualification.',source='Oracle-tracking diagnostic; separate video-reconstruction fit required.'),indent=2))
if __name__=='__main__':main()
