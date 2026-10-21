"""Validation-only calibration of TRUE-parameter surrogate probe discrepancy.

This is not inference. Labels may be read only for the declared validation
calibration/assessment. Candidate calibration residuals are frozen first; the
held-out cases and independent evaluation controls cannot affect covariance.
No fitted bias is subtracted and no posterior-parameter variance is added.
"""
import os
os.environ.setdefault('TORCH_DISABLE_NATIVE_JIT', '1')
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import numpy as np
import torch
from math import erf
from gaussian_operator import physical_vector, category_labels, deformation_diagnostics
from gaussian_acquisition_sensor import public_ray_queries, predict_ranges, measure_ranges
from train_gaussian_inverse import load_frozen_operator


def validate_roles(plan, calibration, assessment):
    if not calibration or not assessment or set(calibration)&set(assessment):
        raise ValueError('Nonempty disjoint calibration and assessment cases required')
    allowed={c['episode_id'] for c in plan['episodes'] if c['split']=='validation'}
    if not set(calibration+assessment)<=allowed:
        raise ValueError('This development calibration permits VALIDATION cases only, never TEST')


def fit_discrepancy(residuals, times_s, floor_m=1e-7):
    """Moment fit on [independent reset sequences,times,rays], no mean removal.

    Within-ray variance, zero-centered sequence means and deviations from each
    sequence mean yield nonnegative conservative covariance components. An AR1
    frame-common term plus a sequence-common term avoids treating consecutive
    video frames or nearby rays as independent evidence. This is a small-sample
    working likelihood, not a certified physical noise distribution.
    """
    r=np.asarray(residuals,dtype=float); times=np.asarray(times_s,dtype=float)
    if r.ndim!=3 or min(r.shape)<2 or not np.isfinite(r).all():
        raise ValueError('Need finite [sequences>=2,times>=2,rays>=2] residuals')
    if times.shape!=(r.shape[1],) or np.any(np.diff(times)<=0):
        raise ValueError('Strictly increasing times required')
    means=r.mean(-1); grand=means.mean(-1,keepdims=True)
    independent=max(float(np.mean((r-means[...,None])**2)/(1-1/r.shape[-1])),floor_m**2)
    frame=max(float(np.mean((means-grand)**2)-independent/r.shape[-1]),0.)
    sequence=max(float(np.mean(grand**2)-independent/(r.shape[1]*r.shape[2])),0.)
    centered=means-grand
    denominator=float(np.sum(centered[:,:-1]**2))
    rho=float(np.clip(np.sum(centered[:,:-1]*centered[:,1:])/denominator,0.,.99)) if denominator>0 else 0.
    return {'independent_std_m':float(np.sqrt(independent)),
            'frame_common_std_m':float(np.sqrt(frame)),
            'sequence_common_std_m':float(np.sqrt(sequence)),
            'frame_ar1':rho,'reference_dt_s':float(np.median(np.diff(times))),
            'calibration_sequences':r.shape[0], 'frames_per_sequence':r.shape[1],
            'rays_per_frame':r.shape[2], 'unsubtracted_global_bias_m':float(r.mean()),
            'calibration_range_rmse_m':float(np.sqrt(np.mean(r*r))),
            'method':'Zero-mean residual second moments; no bias correction; AR1 frame common plus sequence common'}


def probe_discrepancy_covariance(times_s, rays, parameters):
    times=np.asarray(times_s,dtype=float)
    if times.ndim!=1 or rays<1 or not np.isfinite(times).all():raise ValueError('Invalid queries')
    a=float(parameters['independent_std_m']); b=float(parameters['frame_common_std_m']); c=float(parameters['sequence_common_std_m'])
    rho=float(parameters['frame_ar1']);dt=float(parameters['reference_dt_s'])
    if a<=0 or min(b,c)<0 or not 0<=rho<1 or dt<=0:raise ValueError('Invalid covariance parameters')
    distance=abs(times[:,None]-times[None,:])/dt
    correlation=np.eye(len(times)) if rho==0 else rho**distance
    n=len(times)*rays
    return a*a*np.eye(n)+b*b*np.kron(correlation,np.ones((rays,rays)))+c*c*np.ones((n,n))


def diagnostics(residuals, covariance):
    r=np.asarray(residuals);flat=r.reshape(len(r),-1)
    cholesky=np.linalg.cholesky(covariance)
    whitened=np.linalg.solve(cholesky,flat.T).T
    sigma=np.sqrt(np.diag(covariance));z=flat/sigma
    # Scaled marginal CRPS and proper joint Gaussian log score. Correlations
    # enter the log score/whitening; marginal coverage is explicitly marginal.
    phi=np.exp(-.5*z*z)/np.sqrt(2*np.pi)
    cdf=.5*(1+np.vectorize(erf)(z/np.sqrt(2)))
    crps=sigma*(z*(2*cdf-1)+2*phi-1/np.sqrt(np.pi))
    n=flat.shape[-1]
    return {'range_rmse_m':float(np.sqrt(np.mean(flat**2))),
            'signed_range_bias_m':float(flat.mean()),
            'marginal_90_coverage':float(np.mean(abs(z)<=1.6448536269514722)),
            'marginal_90_mean_width_m':float(np.mean(2*1.6448536269514722*sigma)),
            'marginal_crps_m':float(crps.mean()),
            'joint_nll_per_scalar':float(np.mean(.5*(np.sum(whitened**2,axis=1)+n*np.log(2*np.pi)+2*np.log(np.diag(cholesky)).sum()))/n),
            'mean_whitened_squared_residual':float(np.mean(whitened**2))}


def sequence(model,pool,case,action,query,frame_ids,device):
    eid=case['episode_id'];private=pool/'private'/eid
    with np.load(private/'reference.npz') as a:reference={k:a[k] for k in a.files}
    spec=json.loads((private/'truth/boundaries.json').read_text())
    physical=physical_vector(case,spec,reference['rest_points_m']);physical[16:22]=0;physical[16:19]=action['peak_force_N']
    bc,_=category_labels(case);family=action.get('force_type_index',0)
    menu=json.loads((pool/'public'/eid/'action-menu.json').read_text());times=np.asarray(menu['sensor']['times_s'])
    rest=torch.tensor(reference['rest_points_m'],dtype=torch.float32,device=device)
    with torch.no_grad():
        q=model(rest,torch.tensor(times,dtype=torch.float32,device=device),torch.tensor(physical[None],device=device),torch.tensor([bc],device=device),torch.tensor([family],device=device))[0].cpu().numpy()
    predicted=np.stack([predict_ranges(q[i:i+1],reference,query)[0] for i in frame_ids])
    directory=pool/'outcomes'/eid/action['action_id']/'observations/depth'
    measured=np.stack([measure_ranges(np.load(directory/f'{i:05d}.npz')['depth_m'],query) for i in frame_ids])
    with np.load(private/action['action_id']/'trajectory.npz') as a:truth=a['positions_m']
    return {'episode_id':eid,'action_id':action['action_id'],'residual_m':predicted-measured,
            'predicted_range_m':predicted,'measured_range_m':measured,
            'coordinate_rmse_m':float(np.sqrt(np.mean((q-truth)**2))),
            'deformation':deformation_diagnostics(torch.tensor(q),torch.tensor(reference['rest_points_m'],dtype=torch.float32),torch.tensor(reference['tetrahedra'],dtype=torch.long))}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['dataset','pool','forward-checkpoint','output']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--calibration',nargs='+',default=['e0002','e0005']);p.add_argument('--assessment',nargs='+',default=['e0008','e0015'])
    p.add_argument('--rays',type=int,default=12);p.add_argument('--stride',type=int,default=2);p.add_argument('--seed',type=int,default=20261106);p.add_argument('--device',default='cpu')
    a=p.parse_args();torch.set_num_threads(2)
    plan=json.loads((a.dataset/'plan.json').read_text());validate_roles(plan,a.calibration,a.assessment)
    if a.stride<1:p.error('Positive stride required')
    a.output.mkdir(parents=True,exist_ok=False);(a.output/'provenance').mkdir()
    for name in [Path(__file__).name,'gaussian_acquisition_sensor.py','gaussian_operator.py','train_gaussian_inverse.py','gaussian_modal_operator.py']:
        shutil.copy2(Path(__file__).with_name(name),a.output/'provenance'/name)
    protocol={k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()}
    protocol.update({'forward_sha256':hashlib.sha256(a.forward_checkpoint.read_bytes()).hexdigest(),
        'source_sha256':{path.name:hashlib.sha256(path.read_bytes()).hexdigest() for path in (a.output/'provenance').iterdir()},
        'dataset_plan_sha256':hashlib.sha256((a.dataset/'plan.json').read_bytes()).hexdigest(),
        'calibration_access':'True-parameter candidate residuals on calibration cases only; no evaluation controls',
        'assessment_access':'After covariance file is exclusively frozen, assessment cases and independent evaluation controls',
        'qualification':'Validation-development calibration, not official TEST; four original layouts, homogeneous1.5kPa material; no posterior variance included'})
    (a.output/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    cases={c['episode_id']:c for c in plan['episodes']};model=load_frozen_operator(a.forward_checkpoint,a.device)
    queries={};menus={};frame_ids=None;times=None
    for eid in a.calibration+a.assessment:
        if not json.loads((a.pool/'private'/eid/'validation.json').read_text())['passed']:raise ValueError('Unaudited pool case')
        menu=json.loads((a.pool/'public'/eid/'action-menu.json').read_text());menus[eid]=menu
        with np.load(a.pool/'private'/eid/'reference.npz') as raw:reference={k:raw[k] for k in raw.files}
        queries[eid]=public_ray_queries(reference,np.asarray(menu['sensor']['camera_to_world_usd_row']),np.asarray(menu['sensor']['K']),a.rays,seed=a.seed)
        current=np.asarray(menu['sensor']['times_s']);ids=np.arange(1,len(current),a.stride)
        if times is not None and not np.array_equal(current[ids],times):raise ValueError('Mixed sensor time schedules')
        frame_ids=ids;times=current[ids]
    calibration=[]
    for eid in a.calibration:
        for action in menus[eid]['candidates']:calibration.append(sequence(model,a.pool,cases[eid],action,queries[eid],frame_ids,a.device))
    parameters=fit_discrepancy(np.stack([s['residual_m'] for s in calibration]),times)
    frozen={'parameters':parameters,'protocol':protocol,'sensor_times_s':times.tolist(),'frozen_before_assessment':True}
    with (a.output/'frozen-covariance.json').open('x') as stream:json.dump(frozen,stream,indent=2)
    covariance=probe_discrepancy_covariance(times,a.rays,parameters)
    rows=[]
    for s in calibration:
        rows.append({**{k:v for k,v in s.items() if k not in ['residual_m','predicted_range_m','measured_range_m']},'role':'calibration_candidate',**diagnostics(s['residual_m'][None],covariance)})
    # This loop is deliberately AFTER the immutable covariance artifact write.
    for eid in a.calibration+a.assessment:
        actions=menus[eid]['candidates'] if eid in a.assessment else []
        for action in actions+[menus[eid]['evaluation_query']]:
            s=sequence(model,a.pool,cases[eid],action,queries[eid],frame_ids,a.device)
            role='assessment_candidate' if action['action_id']!='evaluation' else 'independent_evaluation_control'
            rows.append({**{k:v for k,v in s.items() if k not in ['residual_m','predicted_range_m','measured_range_m']},'role':role,**diagnostics(s['residual_m'][None],covariance)})
            np.savez_compressed(a.output/(eid+'-'+action['action_id']+'.npz'),residual_m=s['residual_m'],predicted_range_m=s['predicted_range_m'],measured_range_m=s['measured_range_m'])
    (a.output/'results.json').write_text(json.dumps({'rows':rows,'frozen_covariance':frozen},indent=2)+'\n')
    print(json.dumps({'parameters':parameters,'rows':rows},indent=2),flush=True)


if __name__=='__main__':main()
