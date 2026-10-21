"""Fit censored probe observation model on RESERVED TRAIN interventions only.

No validation/test response is opened. Teacher must be frozen at the declared
final step. Calibrated candidate controls t0012..t0015 were excluded from its
control fitting, although those cases' original passive TRAIN videos remain
allowed. Four cases provide development calibration, not a coverage guarantee.
"""
import os
os.environ.setdefault('TORCH_DISABLE_NATIVE_JIT','1')
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import shutil
import time
import numpy as np
import torch
from gaussian_probe_roles import load_role_manifest,control_role,known_control_label
from train_gaussian_inverse import load_frozen_operator
from gaussian_acquisition_sensor import public_ray_queries
from calibrate_gaussian_probe_discrepancy import probe_discrepancy_covariance
from censored_range_observation import (CensoredRanges,CensoredRangeModel,
    predict_censored_ranges,measure_censored_depth,concatenate_observations)

ROLE_SHA256='6d2781a71a24ebfa9cb0691ce0dd7ae889c15eae1b38ca3164618523d5552c84'
SENSOR_AMENDMENT_SHA256='b076f1489d05067b3c0dc14c5d6be1dd84bb4a420a4d0fa04b5a39cc8424b95a'


def fit_conservative_parameters(predicted,observed,case_ids,times_s,rays,
                                sensor_floor_m=2e-6,background_std_m=.05):
    """Global TRAIN-only covariance with holes excluded, never zero-filled.

    Worst sequence95% absolute residual gives an isotropic scale. Sequence mean
    and frame-common terms conservatively retain correlated bias on top of that
    scale. Double counting is deliberate; this is not maximum-likelihood fitting.
    Visibility smoothing uses case blocks rather than fake independent pixels.
    """
    if predicted.visible.ndim!=2 or predicted.visible.shape!=observed.visible.shape:
        raise ValueError('Matching [sequences,features] observations required')
    case_ids=np.asarray(case_ids);times=np.asarray(times_s,dtype=float)
    if len(case_ids)!=len(predicted.visible) or predicted.visible.shape[1]!=len(times)*rays or len(times)<2:
        raise ValueError('Case/time/ray dimensions differ')
    if sensor_floor_m<=0 or background_std_m<=0 or np.any(np.diff(times)<=0):raise ValueError('Invalid scales/times')
    joint=predicted.visible&observed.visible
    residual=predicted.residual_m-observed.residual_m
    counts=joint.sum(-1);unsupported=bool((counts<8).any())
    values=residual[joint]
    if not len(values):unsupported=True
    quantiles=[float(np.quantile(abs(residual[i,joint[i]]),.95)) for i in range(len(joint)) if counts[i]]
    quantile=max(quantiles+[float(np.quantile(abs(values),.95)) if len(values) else 0.])
    independent=max(sensor_floor_m,quantile/1.6448536269514722)
    if unsupported:independent=max(independent,background_std_m)
    means=[];frame_deviations=[];lag_a=[];lag_b=[]
    for i in range(len(joint)):
        mean=float(residual[i,joint[i]].mean()) if counts[i] else 0.;means.append(mean)
        grid=residual[i].reshape(len(times),rays);valid=joint[i].reshape(len(times),rays)
        frame_mean=np.full(len(times),np.nan)
        for frame in range(len(times)):
            if valid[frame].any():frame_mean[frame]=grid[frame,valid[frame]].mean()
        centered=frame_mean-mean
        frame_deviations.extend(centered[np.isfinite(centered)].tolist())
        paired=np.isfinite(centered[:-1])&np.isfinite(centered[1:])
        lag_a.extend(centered[:-1][paired].tolist());lag_b.extend(centered[1:][paired].tolist())
    sequence=float(max(abs(np.asarray(means)),default=0.))
    frame=float(np.sqrt(np.mean(np.square(frame_deviations)))) if frame_deviations else 0.
    denominator=float(np.sum(np.square(lag_a)))
    rho=float(np.clip(np.dot(lag_a,lag_b)/denominator,.5,.99)) if denominator>0 else .9
    mismatch=(predicted.visible!=observed.visible).mean(-1)
    unique=np.unique(case_ids);case_rates=[float(mismatch[case_ids==case].mean()) for case in unique]
    # One half pseudo-error case and half pseudo-correct case; 4 cases with zero
    # observed mismatch still give epsilon=.1, avoiding pixel-count confidence.
    visibility=float(np.clip((sum(case_rates)+.5)/(len(unique)+1),.01,.49))
    parameters={'independent_std_m':float(independent),'frame_common_std_m':frame,
                'sequence_common_std_m':sequence,'frame_ar1':rho,
                'reference_dt_s':float(np.median(np.diff(times)))}
    diagnostics={'joint_valid_counts_per_sequence':counts.tolist(),'calibration_sequences':len(joint),
        'effective_calibration_case_count':len(unique),'case_visibility_mismatch_rates':dict(zip(unique.tolist(),case_rates)),
        'unsubtracted_joint_range_bias_m':float(values.mean()) if len(values) else None,
        'joint_range_rmse_m':float(np.sqrt(np.mean(values**2))) if len(values) else None,
        'worst_sequence_absolute_residual_95_m':quantile,
        'range_support_sufficient':not unsupported,'visibility_error_below_half':all(p<.5 for p in case_rates),
        'qualified':not unsupported and all(p<.5 for p in case_rates),
        'method':'Worst-reset95% absolute range quantile /1.645 + correlated second moments; only jointly valid returns; case-block visibility smoothing'}
    return parameters,visibility,diagnostics


def load_generation_amendment(path):
    if path is None:return None
    content=Path(path).read_bytes()
    if hashlib.sha256(content).hexdigest()!=SENSOR_AMENDMENT_SHA256:
        raise ValueError('Unrecognized frozen sensor-only amendment')
    amendment=json.loads(content)
    if set(amendment['case_exceptions'])!={'t0013'} or amendment['tolerance_m']!=2e-6:
        raise ValueError('Only declaredt0013same-tolerance sensor exception allowed')
    return amendment


def verify_generation_source(summary,roles,eid,private,amendment=None):
    sources=summary['source_sha256']
    if sources.get('isaac_calibrated_probes_v02.py')==roles['generation_source_sha256']:
        return
    if amendment is None or eid not in amendment['case_exceptions'] or eid not in roles['calibration_only_ids']:
        raise ValueError('Unamended calibration generator differs from manifest')
    exception=amendment['case_exceptions'][eid]
    if (sources.get(exception['worker_file'])!=exception['worker_sha256'] or
            sources.get('isaac_probe_sensor_guard.py')!=exception['sensor_guard_sha256']):
        raise ValueError('Sensor-only source exception fingerprint differs')
    name=exception['equivalence_file']
    if Path(name).name!=name:raise ValueError('Unsafe equivalence metadata path')
    private=Path(private);equivalence=json.loads((private/name).read_text())
    if not all(equivalence.get(k) for k in ['passed','bit_exact_all_completed_actions','bit_exact_failed_state']):
        raise ValueError('Required unchanged-physics paired proof failed')
    archived=Path(equivalence['original_preserved']).resolve();pool=private.parents[1].resolve()
    if not archived.is_relative_to(pool/'rejected-attempts'):
        raise ValueError('Original failure not preserved inside rejected-attempts')
    old=archived/'private'/eid
    if json.loads((old/'truth/recipe.json').read_text())!=json.loads((private/'truth/recipe.json').read_text()):
        raise ValueError('Sensor recovery changed the physical recipe')
    for action in ['poke_0','poke_1','poke_2']:
        with np.load(old/action/'trajectory.npz') as left,np.load(private/action/'trajectory.npz') as right:
            for key in ['positions_m','velocities_m_s','handle_positions_m','times_s','force_step_commands_N']:
                if not np.array_equal(left[key],right[key]):raise ValueError('Sensor recovery changed paired mechanics')
    failure=json.loads((old/'failure.json').read_text())
    with np.load(old/'failure-state.npz') as left,np.load(private/'poke_3/trajectory.npz') as right:
        indices=np.flatnonzero(abs(right['times_s']-failure['time_s'])<1e-12)
        if len(indices)!=1 or not np.array_equal(left['positions_m'],right['positions_m'][indices[0]]):
            raise ValueError('Sensor recovery differs at exact original failed state')


def calibration_ready(checkpoint,pool,roles):
    return Path(checkpoint).exists() and all((Path(pool)/'private'/eid/'validation.json').exists() for eid in roles['calibration_only_ids'])


def read_sequence(model,dataset,pool,roles,case,action_id,frame_ids,rays,seed,device,amendment=None):
    eid=case['episode_id']
    # Role eligibility precedes ANY private recipe/material/response access.
    if control_role(roles,eid,action_id,case['split'])!='calibration_only':
        raise ValueError('Only reserved TRAIN candidate responses may calibrate')
    private=pool/'private'/eid
    if not json.loads((private/'validation.json').read_text()).get('passed'):raise ValueError('Unaudited calibration response')
    if json.loads((private/'truth/recipe.json').read_text())!=case:raise ValueError('Calibration recipe differs from frozen plan')
    summary=json.loads((private/'summary.json').read_text())
    if not summary.get('completed'):raise ValueError('Incomplete calibration simulation')
    for name,expected in summary['source_sha256'].items():
        if Path(name).name!=name or hashlib.sha256((private/'provenance'/name).read_bytes()).hexdigest()!=expected:
            raise ValueError('Calibration source provenance changed')
    verify_generation_source(summary,roles,eid,private,amendment)
    menu=json.loads((pool/'public'/eid/'action-menu.json').read_text())
    action=next(a for a in menu['candidates'] if a['action_id']==action_id);declared=roles['controls'][action_id]
    if action['force_type_index']!=declared['force_type_index'] or not np.array_equal(action['peak_force_N'],declared['peak_force_N']):
        raise ValueError('Calibration action differs from frozen public control')
    with np.load(private/'reference.npz') as raw:reference={k:raw[k] for k in raw.files}
    spec=json.loads((private/'truth/boundaries.json').read_text());physical,bc,family=known_control_label(case,spec,reference['rest_points_m'],action)
    if not np.allclose(physical[10:16],np.asarray(menu['known_handle_centers_m']).ravel(),atol=1e-7):raise ValueError('Known hardware differs')
    query=public_ray_queries(reference,np.asarray(menu['sensor']['camera_to_world_usd_row']),np.asarray(menu['sensor']['K']),rays,seed=seed)
    times=np.asarray(menu['sensor']['times_s'])
    if len(times)!=31 or not np.allclose(times,np.arange(31)/10,atol=1e-10):raise ValueError('Unexpected actual sensor timestamps')
    with torch.no_grad():
        q=model(torch.tensor(reference['rest_points_m'],device=device,dtype=torch.float32),torch.tensor(times,device=device,dtype=torch.float32),
                torch.tensor(physical[None],device=device),torch.tensor([bc],device=device),torch.tensor([family],device=device)).cpu().numpy()
    predicted=predict_censored_ranges(q,reference,query,frame_ids)
    directory=pool/'outcomes'/eid/action_id/'observations/depth';measured=[]
    for frame in frame_ids:
        with np.load(directory/f'{frame:05d}.npz') as raw:measured.append(measure_censored_depth(raw['depth_m'],query))
    observed=concatenate_observations(measured)
    return CensoredRanges(predicted.visible[0],predicted.residual_m[0]),observed,times[frame_ids],query


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['dataset','pool','roles','forward-checkpoint','output']:p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--rays',type=int,default=12);p.add_argument('--frame-stride',type=int,default=2);p.add_argument('--seed',type=int,default=20261106)
    p.add_argument('--generation-amendment',type=Path)
    p.add_argument('--device',default='cpu');p.add_argument('--wait-for-ready',action='store_true');p.add_argument('--maximum-wait-s',type=float,default=14400)
    a=p.parse_args();torch.set_num_threads(2)
    if (a.rays,a.frame_stride,a.seed)!=(12,2,20261106):p.error('Frozen protocol requires12rays,stride2,seed20261106')
    roles=load_role_manifest(a.roles,ROLE_SHA256);amendment=load_generation_amendment(a.generation_amendment)
    if hashlib.sha256((a.dataset/'plan.json').read_bytes()).hexdigest()!=roles['dataset_plan_sha256']:raise ValueError('Source dataset plan changed')
    started=time.monotonic();last_report=-100
    while not calibration_ready(a.forward_checkpoint,a.pool,roles):
        if not a.wait_for_ready:raise FileNotFoundError('Final teacher/reserved controls not ready')
        if time.monotonic()-started>a.maximum_wait_s:raise TimeoutError('Reserved calibration readiness timeout')
        if time.monotonic()-last_report>30:
            print(json.dumps({'waiting_for':'Frozen final teacher and all4reservedTRAINaudits','elapsed_s':time.monotonic()-started}),flush=True);last_report=time.monotonic()
        time.sleep(5)
    checkpoint_hash=hashlib.sha256(a.forward_checkpoint.read_bytes()).hexdigest()
    checkpoint=torch.load(a.forward_checkpoint,map_location='cpu',weights_only=False)
    if checkpoint.get('step')!=12000 or checkpoint.get('protocol',{}).get('role_sha256')!=ROLE_SHA256:
        raise ValueError('Calibrate only declared12000-step controlled final teacher with frozen roles')
    controls=[r for r in checkpoint['catalog'] if r.get('role')=='teacher_fit_control']
    if len(controls)!=48:raise ValueError('Final teacher must include48declaredcontrolresponses')
    for row in controls:
        eid,action=row['episode_id'].split('/')
        if control_role(roles,eid,action,row['split'])!='teacher_fit':raise ValueError('Teacher used calibration/evaluation response')
    model=load_frozen_operator(a.forward_checkpoint,a.device)
    a.output.mkdir(parents=True,exist_ok=False);(a.output/'provenance').mkdir()
    sources=[Path(__file__).name,'censored_range_observation.py','gaussian_probe_roles.py','gaussian_operator.py','gaussian_acquisition_sensor.py','calibrate_gaussian_probe_discrepancy.py','isaac_calibrated_probes_v03.py','isaac_probe_sensor_guard.py']
    if a.generation_amendment:shutil.copy2(a.generation_amendment,a.output/'generation-amendment.json')
    for name in sources:shutil.copy2(Path(__file__).with_name(name),a.output/'provenance'/name)
    plan=json.loads((a.dataset/'plan.json').read_text());cases={c['episode_id']:c for c in plan['episodes']}
    predicted=[];observed=[];ids=[];queries={};times=None;frame_ids=np.arange(1,31,a.frame_stride)
    for eid in roles['calibration_only_ids']:
        for action_id in roles['eligible_control_ids']:
            pred,obs,current,query=read_sequence(model,a.dataset,a.pool,roles,cases[eid],action_id,frame_ids,a.rays,a.seed,a.device,amendment)
            if times is not None and not np.array_equal(current,times):raise ValueError('Mixed calibration time grids')
            times=current;predicted.append(pred);observed.append(obs);ids.append(eid)
            queries[eid]={'xs':query['xs'].tolist(),'ys':query['ys'].tolist()}
    pred=CensoredRanges(np.stack([s.visible for s in predicted]),np.stack([s.residual_m for s in predicted]))
    obs=CensoredRanges(np.stack([s.visible for s in observed]),np.stack([s.residual_m for s in observed]))
    covariance_parameters,visibility,fit=fit_conservative_parameters(pred,obs,ids,times,a.rays)
    covariance=probe_discrepancy_covariance(times,a.rays,covariance_parameters)
    model_observation=CensoredRangeModel(covariance,visibility,background_std_m=.05,outlier_probability=.05)
    joint=pred.visible&obs.visible;sigma=np.sqrt(np.diag(covariance));residual=pred.residual_m-obs.residual_m
    visibility_mean=.95*np.where(pred.visible,1-visibility,visibility)+.05*.5
    fit.update(nominal_gaussian_90_coverage_joint_hits=float(np.mean((abs(residual)<=1.6448536269514722*sigma)[joint])) if joint.any() else None,
               marginal_visibility_brier=float(np.square(visibility_mean-obs.visible).mean()),
               joint_mixed_log_score_per_query=float(np.mean([model_observation.log_prob(CensoredRanges(pred.visible[i:i+1],pred.residual_m[i:i+1]),CensoredRanges(obs.visible[i],obs.residual_m[i]))[0]/pred.visible.shape[1] for i in range(len(ids))])))
    artifact={'protocol':'reserved-TRAIN-censored-calibration-v0.1','covariance_parameters':covariance_parameters,
        'visibility_error_probability':visibility,'background_std_m':.05,'outlier_probability':.05,
        'forward_checkpoint_sha256':checkpoint_hash,'roles_sha256':ROLE_SHA256,
        'generation_amendment_sha256':hashlib.sha256(a.generation_amendment.read_bytes()).hexdigest() if a.generation_amendment else None,
        'queries':{'frame_stride':a.frame_stride,'rays':a.rays,'seed':a.seed,'frame_ids':frame_ids.tolist()},
        'sensor_times_s':times.tolist(),'calibration_ids':roles['calibration_only_ids'],'calibration_action_ids':roles['eligible_control_ids'],
        'calibration_diagnostics':fit,'public_query_pixels':queries,
        'source_sha256':{name:hashlib.sha256((a.output/'provenance'/name).read_bytes()).hexdigest() for name in sources},
        'frozen_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'qualification':'Global conservative working covariance;4reservedTRAIN intervention cases; original passive videos allowed; noVAL/TEST response fitted; no general coverage guarantee',
        'mixture':'Normal: epsilon-flip visibility+valid-subcovariance ranges+background predicted-miss hits; whole-experiment outlier: fair visibility+independent broad ranges'}
    if hashlib.sha256(a.forward_checkpoint.read_bytes()).hexdigest()!=checkpoint_hash:raise RuntimeError('Teacher changed during calibration')
    with (a.output/'frozen-calibration.json').open('x') as stream:json.dump(artifact,stream,indent=2)
    np.savez_compressed(a.output/'reserved-train-residuals.npz',predicted_visible=pred.visible,observed_visible=obs.visible,predicted_residual_m=pred.residual_m,observed_residual_m=obs.residual_m,case_ids=ids)
    print(json.dumps(artifact,indent=2),flush=True)


if __name__=='__main__':main()
