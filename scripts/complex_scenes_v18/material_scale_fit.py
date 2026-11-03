"""Paired development fitting of frozen and extended shared material laws."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import argparse,hashlib,json,time
import numpy as np
import torch
from scipy.optimize import minimize,minimize_scalar
from scripts.complex_scenes_v18.material_rollout import run
from scripts.complex_scenes_v18.run_development import ROOT,write


def fit(case,variant,extended):
    source=ROOT/'runs/complex-scenes-v18/development-prefixes'/f'{case}-v{variant}';meta=json.loads((source/'metadata.json').read_text());target=source/'observations.npz'
    if hashlib.sha256(target.read_bytes()).hexdigest()!=meta['observations_sha256']:raise ValueError('Prefix changed')
    a=np.load(target);observed=a['positions_m'];indices=a['surface_indices'];times=a['times_s'];rows=[]
    mode='bulk-extension'if extended else'frozen';rootlabel=f'material-scale-fit-01/{mode}/{case}-v{variant}';out=ROOT/'runs/complex-scenes-v18'/rootlabel;out.mkdir(parents=True,exist_ok=False);start=time.monotonic()
    def objective(p):
        p=np.atleast_1d(p);parameters=dict(scale=float(p[0]),log_bulk=float(p[1])if extended else 0.,family_flag=-1.,shape=0.)
        label=rootlabel+f'/trial-{len(rows):03d}';status=run(case,variant,parameters,label=label,duration=meta['calibration_duration_s'])
        row=dict(index=len(rows),parameters=parameters,completed=status['completed'],loss_m2=1e3,path=label)
        if status['completed']:
            pred=np.load(ROOT/'runs/complex-scenes-v18'/label/f'{case}-v{variant}'/'trajectory.npz')
            if pred['surface_positions_m'][:,:,indices].shape!=observed.shape or not np.allclose(pred['times_s'],times,atol=1e-9):raise ValueError('Calibration clock/shape')
            row['loss_m2']=float(np.mean(np.sum((pred['surface_positions_m'][1:,:,indices]-observed[1:])**2,axis=-1)))
        else:row['error']=status.get('error')
        rows.append(row);write(out/'status.json',dict(pid=os.getpid(),trials=rows,completed=False));return row['loss_m2']
    objective([0.,0.])
    if extended:result=minimize(objective,[0.,0.],method='Powell',bounds=[[-2.,2.],[-1.5,1.5]],options=dict(maxfev=28,xtol=.05,ftol=.001))
    else:result=minimize_scalar(objective,bounds=(-2.,2.),method='bounded',options=dict(maxiter=12,xatol=.02))
    valid=[r for r in rows if r['completed']];best=min(valid,key=lambda r:r['loss_m2'])if valid else None
    write(out/'fit-seal.json',dict(completed=best is not None,best=best,trials=rows,optimizer_converged=bool(result.success),observations_sha256=meta['observations_sha256'],scope='Development known-boundary material ablation; no held or reference-qualified claim',forecast_targets_read=False,elapsed_s=time.monotonic()-start))
    if best:run(case,variant,best['parameters'],label=rootlabel+'/sealed-rollout')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--case',required=True);p.add_argument('--variant',type=int,default=0);p.add_argument('--extended',action='store_true');a=p.parse_args();torch.set_num_threads(1);fit(a.case,a.variant,a.extended)
