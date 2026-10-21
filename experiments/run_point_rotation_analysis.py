"""Fresh role-gated rotation-aware inference with persistent joint discrepancy.

Choices precede new views; target distributions precede future/parameter truth.
The old test is never reevaluated. All model-error fitting uses old development
objects, while this runner admits only the new independent cohort.
"""
from __future__ import annotations
import argparse, os, time
from pathlib import Path
import numpy as np
try:
    import run_point_prior_analysis as old
    from campaign_rotation_posterior import load_rotation_libraries, condition_rotation_reference
    from campaign_joint_gaussian_discrepancy import JointGaussianDiscrepancy
    from campaign_point_rotation_forward import PointRotationForward
    from campaign_sparse_depth_queries import make_queries, predict_ranges
    from campaign_support_basis_cache import load_support_basis_cache
    from campaign_exciting_commands import make_exciting_commands, PROGRAM_IDS
    from campaign_simultaneous_bands import object_score
    from censored_range_observation import CensoredRangeModel, CensoredRanges
except ImportError:
    from experiments import run_point_prior_analysis as old
    from experiments.campaign_rotation_posterior import load_rotation_libraries, condition_rotation_reference
    from experiments.campaign_joint_gaussian_discrepancy import JointGaussianDiscrepancy
    from experiments.campaign_point_rotation_forward import PointRotationForward
    from experiments.campaign_sparse_depth_queries import make_queries, predict_ranges
    from experiments.campaign_support_basis_cache import load_support_basis_cache
    from experiments.campaign_exciting_commands import make_exciting_commands, PROGRAM_IDS
    from experiments.campaign_simultaneous_bands import object_score
    from experiments.censored_range_observation import CensoredRangeModel, CensoredRanges

ROOT = old.ROOT
read, write, checked, record, resolve, sha = old.read, old.write, old.checked, old.record, old.resolve, old.sha
STATE_NAMES = old.STATE_NAMES
METHOD_PATH = 'runs/operator-dev/point-rotation-analysis-v02/method-protocol.json'
METHOD_SHA256 = '7c3a4a9fa89398fdf8c77d0208c6bf932efbaa31751128c95d0ce1f7d5d81ea5'
SCHEMA = 'point-rotation-analysis-execution-v0.2'
NEW_SOURCES = ('run_point_rotation_analysis.py', 'summarize_point_rotation_analysis.py',
    'campaign_rotation_posterior.py', 'campaign_joint_gaussian_discrepancy.py',
    'campaign_projected_point_targets.py', 'campaign_point_rotation_forward.py',
    'fit_point_rotation_discrepancy.py')


def source_map(plan, library, fit):
    pins = {}
    for mapping in [plan['source_sha256'], library['source_sha256'], fit['source_sha256']]:
        for name, digest in mapping.items():
            if name in pins and pins[name] != digest: raise ValueError('Conflicting source closures '+name)
            checked(dict(path=name, sha256=digest)); pins[name] = digest
    for name in NEW_SOURCES:
        path = ROOT/'experiments'/name; pins[path.relative_to(ROOT).as_posix()] = sha(path)
    return pins


def eligible_objects(plan, role, *, execution_sha256, method_sha256,
        release=None, calibration_freeze=None, calibration_freeze_record=None):
    objects = plan['objects']
    if (plan['schema'] != 'independent-point-rotation-prior-cohort-v0.2'
        or len(objects) != 48 or [o['index'] for o in objects] != list(range(48))
        or [o['seed'] for o in objects] != list(range(2026120100,2026120148))
        or any(o['object_id'] != f"point_prior_{o['index']:03d}" for o in objects)
        or any(o['role'] != ('CALIBRATION' if o['index']<16 else 'RESERVED_EVALUATION') for o in objects)
        or plan['roles'] != dict(CALIBRATION=list(range(16)),RESERVED_EVALUATION=list(range(16,48)))):
        raise ValueError('Exact NEW independent48 identities/roles required')
    if role == 'CALIBRATION':
        if release is not None or calibration_freeze is not None: raise ValueError('No reserved release in calibration')
        return objects[:16]
    if role != 'RESERVED_EVALUATION' or not isinstance(release,dict) or not isinstance(calibration_freeze,dict):
        raise ValueError('Explicit reserved release and calibration freeze required')
    if (release.get('schema') != 'point-rotation-reserved-release-v0.2'
        or release.get('status') != 'FROZEN_EXPLICIT_RESERVED_EVALUATION_RELEASE'
        or release.get('method_sha256') != method_sha256
        or release.get('execution_protocol_sha256') != execution_sha256
        or release.get('object_indices') != list(range(16,48))
        or release.get('calibration_freeze') != calibration_freeze_record
        or calibration_freeze.get('completed') is not True
        or calibration_freeze.get('method_sha256') != method_sha256
        or calibration_freeze.get('execution_protocol_sha256') != execution_sha256
        or calibration_freeze.get('calibration_object_indices') != list(range(16))):
        raise ValueError('New role/hash release binding failed')
    return objects[16:]


def load_discrepancy(fit_record):
    fit = read(checked(fit_record))
    if not fit['completed'] or fit['objects'] != 16 or fit['new_cohort_response_or_truth_read']:
        raise ValueError('Only complete OLD-development discrepancy fit allowed')
    with np.load(checked(fit['archive']),allow_pickle=False) as f:
        mT, BT, mO, BO = (f[k].copy() for k in ['target_mean_m','target_factor_m','range_mean_m','range_factor_m'])
    if mT.shape != (2*61*455*3,) or BT.shape != (len(mT),16) or mO.shape!=(216,) or BO.shape!=(216,16):
        raise ValueError('Exact shared target/range factor layout required')
    return JointGaussianDiscrepancy(mO, BO, mT, BT)


def prepare_execution(library_summary, library_sha256, output, analysis_output, *, device='cuda:0'):
    method = read(checked(dict(path=METHOD_PATH,sha256=METHOD_SHA256)))
    plan = read(checked(method['cohort_plan'])); library = read(checked(dict(path=library_summary,sha256=library_sha256)))
    if library['plan'] != method['library_plan']: raise ValueError('Library does not bind new method')
    fit = read(checked(method['discrepancy_fit']))
    if resolve(output).exists() or resolve(analysis_output).exists(): raise ValueError('Fresh execution/output required')
    # This reads observation-INDEPENDENT model arrays only, never new captures.
    libraries = load_rotation_libraries(library_summary, library_sha256)
    model = load_discrepancy(method['discrepancy_fit'])
    eligible_objects(plan,'CALIBRATION',execution_sha256='',method_sha256=METHOD_SHA256)
    p = dict(schema=SCHEMA,status='FROZEN_ROTATION_JOINT_GP_EXECUTION',
        method=dict(path=METHOD_PATH,sha256=METHOD_SHA256),cohort_plan=method['cohort_plan'],
        library_summary=dict(path=library_summary,sha256=library_sha256),discrepancy_fit=method['discrepancy_fit'],
        cache=read(checked(library['plan']))['cache'],source_sha256=source_map(plan,library,fit),
        device=device,CUDA_VISIBLE_DEVICES=os.environ.get('CUDA_VISIBLE_DEVICES'),cpu_threads=2,
        forward_particle_chunk=512,particle_count=2048,analysis_output=analysis_output,roles=plan['roles'],
        discrepancy_model_fingerprint=model.fingerprint,raw_artifacts_LOCAL=True,
        new_cohort_values_opened=False,library_particles_per_seed=55*1024)
    path=resolve(output);path.parent.mkdir(parents=True,exist_ok=True);write(path,p);return record(path)


def validate_execution(path,digest,*,check_environment=True):
    p=read(checked(dict(path=path,sha256=digest)))
    if (p['schema']!=SCHEMA or p['status']!='FROZEN_ROTATION_JOINT_GP_EXECUTION'
        or p['method']!=dict(path=METHOD_PATH,sha256=METHOD_SHA256)
        or p['cpu_threads']!=2 or p['forward_particle_chunk']!=512 or p['particle_count']!=2048
        or not p['device'].startswith('cuda:')
        or (check_environment and p['CUDA_VISIBLE_DEVICES']!=os.environ.get('CUDA_VISIBLE_DEVICES'))):
        raise ValueError('Frozen new execution/settings required')
    method=read(checked(p['method']));plan=read(checked(p['cohort_plan']))
    library=read(checked(p['library_summary']));fit=read(checked(p['discrepancy_fit']))
    if (p['cohort_plan']!=method['cohort_plan'] or library['plan']!=method['library_plan']
        or p['discrepancy_fit']!=method['discrepancy_fit'] or p['roles']!=plan['roles']
        or p['source_sha256']!=source_map(plan,library,fit)):
        raise ValueError('Full new method/source/input binding changed')
    return p,method,plan


def weights_from_increment(increment):
    a=np.asarray(increment,dtype=np.float64)
    if a.ndim!=1 or not np.isfinite(a).all():raise ValueError('Finite particle loglikelihood increments required')
    w=np.exp(a-a.max());return w/w.sum()


def select_view(target,base,predictions,model,*,device):
    n=len(base.latent_mean);weights=np.full(n,1/n);all_weights=[];means=[];covariances=[];indices=[]
    for j,pred in enumerate(predictions):
        rows=np.arange(72+36*j,108+36*j)
        observations,_=model.fantasize_view(base,pred,rows,weights,count=32,seed=2026120400)
        for i in range(32):
            obs=CensoredRanges(observations.visible[i],observations.residual_m[i])
            conditional,increment=model.condition_extra_view(base,pred,obs,rows)
            all_weights.append(weights_from_increment(increment));means.append(conditional.latent_mean)
            covariances.append(conditional.covariance_groups);indices.append(conditional.group_index)
    result=target.risk(np.asarray(all_weights),np.asarray(means),covariance_groups=covariances,group_index=indices)
    risks=result['risk'].detach().cpu().numpy().reshape(4,32);expected=risks.mean(1);selected=int(np.argmin(expected))
    return dict(selected_index=selected,expected_target_risk_m2=expected.tolist(),
        target_risk_mc_se_m2=(risks.std(1,ddof=1)/np.sqrt(32)).tolist(),
        selected_advantage_mc_se_m2=((risks-risks[selected]).std(1,ddof=1)/np.sqrt(32)).tolist(),
        fantasies=32,seed=2026120400,latent_discrepancy_shared_between_views=True,
        risk_includes_conditional_discrepancy=True,actual_frame24_opened=False,
        risk_device=device,roundoff_clamps=result['roundoff_clamps'])


def analyze_episode(obj,pair,context,protocol,execution_sha,destination):
    import torch
    try: from campaign_projected_point_targets import ProjectedPointTargets
    except ImportError: from experiments.campaign_projected_point_targets import ProjectedPointTargets
    forward,public,libraries,model,queries,candidates,method=context
    rest=np.asarray(public['rest_points_m']);faces=np.asarray(public['surface_faces']);index=obj['index'];device=protocol['device']
    destination.mkdir(parents=True,exist_ok=False)
    sensor72=CensoredRangeModel(np.eye(72)*.0002**2,.005,.05,0.)
    observation=old._actual_observation(pair['A']['trajectory'],[10,20],rest,faces,queries,sensor72,2026120200+index,device)
    conditioned=condition_rotation_reference(libraries,observation,model)
    empirical=conditioned['empirical_posterior'];pins=empirical['geometry_ids'];theta=empirical['physical_coordinates']
    base=empirical['base_gp_conditional'];n=len(pins);displacements={}
    for label,program in zip('AB',PROGRAM_IDS):
        q=np.empty((n,61,455,3),dtype=np.float64)
        for lo in range(0,n,512):
            with torch.no_grad():
                values=forward.predict(make_exciting_commands(program),torch.tensor(pins[lo:lo+512].copy()),
                    torch.tensor(theta[lo:lo+512].copy()),device=device)['positions_m']
            q[lo:lo+512]=values.cpu().numpy()-rest
        displacements[label]=q
    all_views=predict_ranges(displacements['A']+rest,faces,candidates,device=device,particle_chunk=64)
    predictions=[CensoredRanges(all_views.visible[:,36*j:36*(j+1)],
        all_views.residual_m[:,36*j:36*(j+1)]) for j in range(4)]
    shape=(2,61,455,3);m=model.target_mean_offset_m.reshape(shape);B=model.target_loading_m.reshape(*shape,16)
    risk_target=ProjectedPointTargets(torch.as_tensor(displacements['A'][:,31:].reshape(n,-1),device=device),
        m[0,31:].reshape(-1),B[0,31:].reshape(-1,16),torch.tensor(pins.copy(),device=device),layout=(30,455,3))
    choice=select_view(risk_target,base,predictions,model,device=device)
    random=int(np.random.default_rng(2026120300+index).integers(4))
    choice.update(object_index=index,role=obj['role'],method_sha256=METHOD_SHA256,
        execution_protocol_sha256=execution_sha,source_sha256=protocol['source_sha256'],selected_random_view=random)
    choice_path=destination/'active-choice-frozen.json';write(choice_path,choice)
    del risk_target
    weights={'no_additional_view':np.full(n,1/n)};conditionals={'no_additional_view':base};ESS={}
    sensor36=CensoredRangeModel(np.eye(36)*.0002**2,.005,.05,0.)
    for j,(query,pred) in enumerate(zip(candidates,predictions)):
        actual=old._actual_observation(pair['A']['trajectory'],[24],rest,faces,[query],sensor36,2026120500+4*index+j,device)
        conditional,increment=model.condition_extra_view(base,pred,actual,np.arange(72+36*j,108+36*j))
        name=f'fixed_view_{j}';weights[name]=weights_from_increment(increment);conditionals[name]=conditional
        ESS[name]=float(1/np.square(weights[name]).sum())
    payload=dict(rest_points_m=rest,times_s=np.arange(61)/20);states={name:{} for name in STATE_NAMES}
    for program_index,label in enumerate('AB'):
        targets=ProjectedPointTargets(torch.as_tensor(displacements[label].reshape(n,-1),device=device),
            m[program_index].reshape(-1),B[program_index].reshape(-1,16),
            torch.tensor(pins.copy(),device=device),layout=(61,455,3))
        for name in STATE_NAMES:
            z=conditionals[name]
            result=targets.moments(weights[name],z.latent_mean,z.covariance_groups,z.group_index)
            mean=result['mean_displacement'].detach().cpu().numpy().reshape(61,455,3)
            std=result['std'].detach().cpu().numpy().reshape(61,455,3)
            states[name][label]=(mean,std)
            payload[name+'__'+label+'__mean_displacement_m']=mean
            payload[name+'__'+label+'__latent_std_m']=std
        del targets
    states_path=destination/'predictive-states.npz';np.savez_compressed(states_path,**payload)
    empirical_path=destination/'empirical-posterior.npz'
    np.savez_compressed(empirical_path,geometry_ids=pins,physical_coordinates=theta,
        **{name+'__weights':weights[name] for name in STATE_NAMES})
    freeze=dict(object_index=index,role=obj['role'],method_sha256=METHOD_SHA256,
        execution_protocol_sha256=execution_sha,source_sha256=protocol['source_sha256'],
        active_choice=record(choice_path),states=record(states_path),empirical_posterior=record(empirical_path),
        state_names=list(STATE_NAMES),future_A_or_B_truth_opened=False,material_or_support_truth_opened=False,
        target_discrepancy_projected_at_candidate_pins=True,discrepancy_model_fingerprint=model.fingerprint)
    freeze_path=destination/'posterior-states-frozen.json';write(freeze_path,freeze)
    # FIRST opening of future targets or physical labels is below the distribution freeze.
    truth={label:old._load_positions_subset(pair[label]['trajectory'],list(range(61)),rest)[0]-rest for label in 'AB'}
    results={}
    for name in STATE_NAMES:
        a,sa=states[name]['A'];b,sb=states[name]['B']
        results[name]=old.target_metrics(truth['A'],truth['B'],a,b,sa,sb,after_view=name!='no_additional_view')
        if name in ESS:results[name]['view_update_ESS']=ESS[name]
    spec=read(checked(obj['realized_hypothesis']));true_node=spec['fixture_nodes'][0]
    primary=next(r for r in conditioned['summaries'] if r['seed']==20261112 and r['count_per_mask']==1024)
    support=sorted(primary['support_categories'],key=lambda r:-r['posterior_probability'])
    support_row=next(r for r in support if r['node_id']==true_node)
    material_values=[spec['material'][k] for k in ['youngs_modulus_Pa','density_kg_m3','poissons_ratio','body_linear_drag_s_inverse']]
    checks=[dict(name=r['name'],truth=float(v),in_90_percent_interval=r['quantile05']<=v<=r['quantile95']) for r,v in zip(primary['material'],material_values)]
    result=dict(completed=True,object_index=index,object_id=obj['object_id'],role=obj['role'],
        method_sha256=METHOD_SHA256,execution_protocol_sha256=execution_sha,posterior_frozen_before_future_truth=True,
        posterior_freeze=record(freeze_path),capture_pairs=pair,selected_active_view=choice['selected_index'],
        selected_random_view=random,states=results,joint_band_score=max(v['individual_band_score'] for v in results.values()),
        rawmaterialintervalchecks=checks,supporttruthprobability=support_row['posterior_probability'],
        supporttruthrank=next(i+1 for i,r in enumerate(support) if r['node_id']==true_node),
        reference_stability=conditioned['stability'],discrepancy_model_fingerprint=model.fingerprint)
    write(destination/'reference-summaries.json',conditioned['summaries']);write(destination/'result.json',result)
    return result


def run(execution_path,execution_sha,role,*,release_path=None,release_sha=None):
    p,method,plan=validate_execution(execution_path,execution_sha)
    release=freeze=freeze_record=None
    if role=='RESERVED_EVALUATION':
        if release_path is None or release_sha is None:raise ValueError('Explicit new reserved release record required')
        release=read(checked(dict(path=release_path,sha256=release_sha)));freeze_record=release['calibration_freeze']
        freeze=read(checked(freeze_record))
        for rec in freeze['calibration_results']:checked(rec)
    objects=eligible_objects(plan,role,execution_sha256=execution_sha,method_sha256=METHOD_SHA256,
        release=release,calibration_freeze=freeze,calibration_freeze_record=freeze_record)
    pairs=old.capture_pairs(plan,objects)
    if any(os.environ.get(k)!='2' for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']):raise ValueError('CPU2 environment required')
    import torch
    torch.set_num_threads(2)
    packed,public,_,_=load_support_basis_cache(p['cache']['path'],p['cache']['sha256'])
    if public!=read(checked(plan['public_hardware'])):raise ValueError('Shared public hardware differs')
    libraries=load_rotation_libraries(p['library_summary']['path'],p['library_summary']['sha256'])
    model=load_discrepancy(p['discrepancy_fit'])
    if model.fingerprint!=p['discrepancy_model_fingerprint']:raise ValueError('Discrepancy model changed')
    queries=read(checked(plan['public_acquired_queries']));rest=public['rest_points_m'];faces=public['surface_faces']
    candidates=make_queries(rest,faces,frame_indices=(24,),offsets_deg=((-60.,10.),(-20.,10.),(20.,10.),(60.,10.)),grid_size=6)
    context=(PointRotationForward(packed,np.asarray(rest,dtype=np.float64)),public,libraries,model,queries,candidates,method)
    output=resolve(p['analysis_output']);output.mkdir(parents=True,exist_ok=True)
    status_path=output/('calibration-execution.json' if role=='CALIBRATION' else 'reserved-execution.json')
    if status_path.exists() or any((output/'objects'/o['object_id']).exists() for o in objects):raise ValueError('Retain existing attempt; no overwrite/retry')
    status=dict(pid=os.getpid(),role=role,terminal=False,completed=False,method_sha256=METHOD_SHA256,
        execution_protocol_sha256=execution_sha,planned_object_indices=[o['index'] for o in objects],
        completed_object_indices=[],unlaunched_object_indices=[o['index'] for o in objects],failures=[])
    write(status_path,status);started=time.monotonic()
    try:
        for obj in objects:
            status['active_object_index']=obj['index'];status['unlaunched_object_indices'].remove(obj['index']);write(status_path,status)
            validate_execution(execution_path,execution_sha)
            result=analyze_episode(obj,pairs[obj['index']],context,p,execution_sha,output/'objects'/obj['object_id'])
            status['completed_object_indices'].append(obj['index'])
            print(__import__('json').dumps(dict(object_index=obj['index'],role=role,completed=True)),flush=True)
            write(status_path,status)
        status.update(terminal=True,completed=True,active_object_index=None,elapsed_s=time.monotonic()-started)
    except BaseException as exc:
        status['failures'].append(dict(object_index=status.get('active_object_index'),error=repr(exc)))
        status.update(terminal=True,completed=False,elapsed_s=time.monotonic()-started);write(status_path,status);raise
    write(status_path,status);return status


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='action',required=True)
    p=sub.add_parser('prepare-execution');p.add_argument('--library-summary',required=True);p.add_argument('--library-sha256',required=True);p.add_argument('--output',required=True);p.add_argument('--analysis-output',required=True);p.add_argument('--device',default='cuda:0')
    for role in ['calibration','reserved']:
        p=sub.add_parser('run-'+role);p.add_argument('--execution',required=True);p.add_argument('--execution-sha256',required=True)
        if role=='reserved':p.add_argument('--release',required=True);p.add_argument('--release-sha256',required=True)
    a=parser.parse_args()
    if a.action=='prepare-execution':result=prepare_execution(a.library_summary,a.library_sha256,a.output,a.analysis_output,device=a.device)
    else:result=run(a.execution,a.execution_sha256,'CALIBRATION' if a.action=='run-calibration' else 'RESERVED_EVALUATION',release_path=getattr(a,'release',None),release_sha=getattr(a,'release_sha256',None))
    print(__import__('json').dumps(result))
