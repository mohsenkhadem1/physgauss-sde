"""Generator-only entry point; hidden parameters never exported as observations."""
import os
os.environ.setdefault('OMP_NUM_THREADS','2')
import argparse,json,time,hashlib
from pathlib import Path
import numpy as np
import torch
from scripts.solver_driven_gaussians.dynamics import DynamicFEM

def reference_parameters():
    return np.array([1.35,500.,0.,0.,0.,0.,0.,0.,0.,120.,2.5])

def main():
    p=argparse.ArgumentParser();p.add_argument('--gpu',default='1');p.add_argument('--fine',action='store_true');a=p.parse_args()
    torch.set_num_threads(2);torch.cuda.set_device(int(a.gpu));m=DynamicFEM(cells=(16,8,8) if a.fine else (8,4,4),device='cuda:'+a.gpu)
    out=Path('runs/solver-driven-gaussians-v01')/('reference-fine' if a.fine else 'reference');out.mkdir(exist_ok=True)
    for case in (['calibration','higher_drop'] if a.fine else ['calibration','side_poke','early_release','force_pulse','higher_drop']):
        if (out/(case+'.npz')).exists():continue
        start=time.time();r=m.rollout(reference_parameters(),case=case,kind='reference',dt=.0001)
        np.savez_compressed(out/(case+'.npz'),x=r['x'].cpu().numpy(),v=r['v'].cpu().numpy(),times=r['times'],rest=m.rest.cpu().numpy(),elements=m.elements.cpu().numpy())
        report=dict(case=case,completed=True,seconds=time.time()-start,minimum_J=r['minimum_J'],max_penetration_m=r['max_penetration_m'],dt=.0001,cells=m.cells,sha256=hashlib.sha256((out/(case+'.npz')).read_bytes()).hexdigest())
        (out/(case+'.json')).write_text(json.dumps(report,indent=2));print(report,flush=True)
        if case=='calibration' and not a.fine:
            keep=r['times']<=.14+1e-9;obs=out.parent/'observations';obs.mkdir(exist_ok=True)
            np.savez_compressed(obs/'calibration.npz',x=r['x'][keep].cpu().numpy(),times=r['times'][keep],rest=m.rest.cpu().numpy())
            (obs/'scope.json').write_text(json.dumps(dict(input='Calibration position trajectories only, before release; no material/force labels or future tracks.',geometry='Known block volume and mass density in this qualification.',source='Oracle-tracking diagnostic; separate video-reconstruction fit required.'),indent=2))
if __name__=='__main__':main()
