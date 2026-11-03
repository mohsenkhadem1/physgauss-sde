"""Development global-stiffness warm start, not the final per-edge baseline.

Optimizer sees only separately exported calibration tracks. Forecast targets are
not imported. The native force/common-boundary adapter is stated in each result.
"""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import argparse,hashlib,json,time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from scipy.optimize import minimize_scalar
from scripts.complex_scenes_v18.spring_scene import run
from scripts.complex_scenes_v18.run_development import ROOT,write


def fit(case,variant):
    source=ROOT/'runs/complex-scenes-v18/development-prefixes'/f'{case}-v{variant}';meta=json.loads((source/'metadata.json').read_text());target=source/'observations.npz'
    if hashlib.sha256(target.read_bytes()).hexdigest()!=meta['observations_sha256']:raise ValueError('Prefix hash mismatch')
    a=np.load(target);observed=a['positions_m'];indices=a['surface_indices'];times=a['times_s'];rows=[]
    out=ROOT/'runs/complex-scenes-v18/spring-scale-fit-01'/f'{case}-v{variant}';out.mkdir(parents=True,exist_ok=False);start=time.monotonic()
    def objective(k):
        label=f'spring-scale-fit-01/{case}-v{variant}/trial-{len(rows):03d}'
        args=SimpleNamespace(case=case,variant=variant,label=label,log10_k=float(k),dt=25e-6,duration=meta['calibration_duration_s'],seconds=1800)
        status=run(args);row=dict(index=len(rows),log10_k=float(k),completed=status['completed'],loss_m2=1e3,path=label)
        if status['completed']:
            p=ROOT/'runs/complex-scenes-v18'/label/f'{case}-v{variant}'/'trajectory.npz';pred=np.load(p)
            if pred['surface_positions_m'][:, :, indices].shape!=observed.shape or not np.allclose(pred['times_s'],times,atol=1e-9):raise ValueError('Matched calibration clock/shape required')
            row['loss_m2']=float(np.mean(np.sum((pred['surface_positions_m'][1:,:,indices]-observed[1:])**2,axis=-1)))
        else:row['error']=status.get('error')
        rows.append(row);write(out/'status.json',dict(pid=os.getpid(),completed=False,trials=rows,observations_sha256=meta['observations_sha256']));return row['loss_m2']
    # Initialization range search only. Final native per-edge fitting is separate.
    objective(-.3)
    result=minimize_scalar(objective,bounds=(-2.,2.),method='bounded',options=dict(maxiter=12,xatol=.02))
    valid=[r for r in rows if r['completed']];best=min(valid,key=lambda r:r['loss_m2'])if valid else None
    write(out/'fit-seal.json',dict(completed=best is not None,best=best,trials=rows,optimizer_converged=bool(result.success),observations_sha256=meta['observations_sha256'],source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),scope='Development scalar warm start, known boundary; no per-edge or held benchmark claim',forecast_targets_read=False,elapsed_s=time.monotonic()-start))
    if best is not None:
        run(SimpleNamespace(case=case,variant=variant,label=f'spring-scale-fit-01/{case}-v{variant}/sealed-rollout',log10_k=best['log10_k'],dt=25e-6,duration=None,seconds=1800))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--case',required=True);p.add_argument('--variant',type=int,default=0);a=p.parse_args();torch.set_num_threads(1);fit(a.case,a.variant)
