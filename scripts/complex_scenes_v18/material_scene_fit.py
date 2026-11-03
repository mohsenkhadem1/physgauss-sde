"""Development-only generic material fitting on shared calibration prefixes.

Every scene gets the same spatial field; no family label selects heterogeneity.
Known boundaries remain an explicit diagnostic assumption. This is not held OOD.
"""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import argparse,hashlib,json,time
from pathlib import Path
import numpy as np
import torch
from scipy.optimize import minimize
from scripts.complex_scenes_v18.material_rollout import run
from scripts.complex_scenes_v18.run_development import ROOT,write


def fit(case,variant,label='material-scene-fit-01',evaluations=64,law_type='learned',prefix=None,initialize_fit=None):
    source=Path(prefix)if prefix else ROOT/'runs/complex-scenes-v18/development-prefixes'/f'{case}-v{variant}'
    meta=json.loads((source/'metadata.json').read_text());target=source/'observations.npz'
    if hashlib.sha256(target.read_bytes()).hexdigest()!=meta['observations_sha256']:raise ValueError('Prefix integrity')
    with np.load(target)as a:
        observed=a['positions_m'].copy();indices=a['surface_indices'].copy();times=a['times_s'].copy()
    if meta['case']!=case or meta['variant']!=variant:raise ValueError('Calibration scene identity mismatch')
    query=None
    if 'public_query'in meta:
        from scripts.complex_scenes_v18.world_queries import Query
        query=Query(**meta['public_query'])
    if not np.isfinite(observed).all()or np.any(np.diff(times)<=0)or not np.isclose(times[-1],meta['calibration_duration_s'],atol=1e-9):raise ValueError('Invalid calibration observations')
    count=observed.shape[1]
    names=['scale','log_bulk','contrast','center']+(['second_scale']if count==2 else[])
    bounds=[(-2.,2.),(-1.5,1.5),(-1.5,1.5),(.2,.8)]+([(-1.5,1.5)]if count==2 else[])
    initial=np.array([0.,0.,0.,.5]+([0.]if count==2 else[]))
    initialization=None
    if initialize_fit:
        previous=Path(initialize_fit)/'fit-seal.json';prior=json.loads(previous.read_text())
        prior_law=prior.get('law_type')
        if prior_law is None:
            from scripts.compositional_neural_mechanics.conditional_inference_pilot import BULK
            legacy=json.loads((previous.parent/'sealed-rollout'/f'{case}-v{variant}'/'complete.json').read_text())
            if legacy.get('checkpoint_sha256')==hashlib.sha256(Path(BULK).read_bytes()).hexdigest() and legacy['parameters']==prior['best']['parameters']:prior_law='learned'
        if not prior['completed'] or prior['forecast_targets_read'] or prior['observations_sha256']!=meta['observations_sha256'] or prior_law!=law_type:raise ValueError('Initialization must use the same calibration observations and material law')
        initial=np.array([prior['best']['parameters'][name]for name in names])
        if not np.isfinite(initial).all() or any(not lo<=v<=hi for v,(lo,hi)in zip(initial,bounds)):raise ValueError('Initialization outside declared parameter space')
        initialization=dict(fit_seal_sha256=hashlib.sha256(previous.read_bytes()).hexdigest(),path=str(previous),prior_evaluations=len(prior['trials']),parameters=initial.tolist())
    rootlabel=f'{label}/{case}-v{variant}';out=ROOT/'runs/complex-scenes-v18'/rootlabel
    out.mkdir(parents=True,exist_ok=False);rows=[];start=time.monotonic()
    plan=dict(development_only=True,known_boundary=True,forecast_targets_read=False,law_type=law_type,
              parameters=names,bounds=bounds,maximum_evaluations=evaluations,
              observations_sha256=meta['observations_sha256'],
              scope='Generic spatial material field; known scene boundaries; analytic control has oracle constitutive-family advantage'if law_type=='analytic'else'Generic spatial material field, frozen pretrained constitutive network; known scene boundaries',
              source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()for p in [ROOT/'scripts/complex_scenes_v18/material_rollout.py',ROOT/'scripts/complex_scenes_v18/material_scene_fit.py']})
    plan.update(calibration_prefix=str(source),prefix_metadata_sha256=hashlib.sha256((source/'metadata.json').read_bytes()).hexdigest(),public_query=meta.get('public_query'),evaluation_split=meta.get('split','development'),development_only=meta.get('split','development')=='development')
    plan['initialization']=initialization
    if law_type=='analytic':
        source=ROOT/'scripts/complex_scenes_v18/analytic_control.py'
        plan['source_sha256'][str(source.relative_to(ROOT))]=hashlib.sha256(source.read_bytes()).hexdigest()
    write(out/'plan.json',plan)
    def objective(values):
        parameters=dict(zip(names,map(float,values)));parameters.update(family_flag=-1.,shape=0.)
        trial=rootlabel+f'/trial-{len(rows):03d}'
        status=run(case,variant,parameters,label=trial,duration=meta['calibration_duration_s'],law_type=law_type,query=query,split=meta.get('split','development'))
        row=dict(index=len(rows),parameters=parameters,completed=status['completed'],loss_m2=1e3,path=trial)
        if status['completed']:
            with np.load(ROOT/'runs/complex-scenes-v18'/trial/f'{case}-v{variant}'/'trajectory.npz')as pred:
                positions=pred['surface_positions_m'][:,:,indices]
                if positions.shape!=observed.shape or not np.allclose(pred['times_s'],times,atol=1e-9):raise ValueError('Calibration clock/shape')
                row['loss_m2']=float(np.mean(np.sum((positions[1:]-observed[1:])**2,axis=-1)))
        else:row['error']=status.get('error')
        rows.append(row);write(out/'status.json',dict(pid=os.getpid(),trials=rows,completed=False,elapsed_s=time.monotonic()-start))
        return row['loss_m2']
    result=minimize(objective,initial,method='Powell',bounds=bounds,options=dict(maxfev=evaluations,xtol=.05,ftol=.001))
    valid=[r for r in rows if r['completed']and np.isfinite(r['loss_m2'])]
    best=min(valid,key=lambda r:r['loss_m2'])if valid else None
    write(out/'fit-seal.json',dict(**plan,completed=best is not None,best=best,trials=rows,optimizer_converged=bool(result.success),elapsed_s=time.monotonic()-start))
    if best:run(case,variant,best['parameters'],label=rootlabel+'/sealed-rollout',law_type=law_type,query=query,split=meta.get('split','development'))
    return best is not None


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--case',required=True);p.add_argument('--variant',type=int,default=0);p.add_argument('--label',default='material-scene-fit-01');p.add_argument('--evaluations',type=int,default=64);p.add_argument('--law-type',choices=['learned','analytic'],default='learned');p.add_argument('--prefix')
    p.add_argument('--initialize-fit')
    a=p.parse_args();torch.set_num_threads(1);raise SystemExit(0 if fit(a.case,a.variant,a.label,a.evaluations,a.law_type,a.prefix,a.initialize_fit)else 1)
