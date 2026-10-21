"""Freeze independent-object bands and a fixed baseline before reserved testing.

Consumes compact results of the prospectively specified episode runner, never
raw trajectories, images or material labels. All sixteen calibration objects
are mandatory. Reserved release binds that freeze and the unchanged execution
protocol. Evaluation reports paired object bootstrap intervals and exact
binomial coverage intervals; frames/rays are never statistical replicates.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.stats import binomtest

try:
    from campaign_simultaneous_bands import calibrate
except ImportError:
    from experiments.campaign_simultaneous_bands import calibrate

ROOT=Path(__file__).resolve().parents[1]
STATE_NAMES=('no_additional_view',*[f'fixed_view_{j}' for j in range(4)])
CALIBRATION_SCHEMA='point-rotation-calibration-freeze-v0.2'
RELEASE_SCHEMA='point-rotation-reserved-release-v0.2'


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text())
def write_new(path,value):
    with Path(path).open('x') as stream:
        json.dump(value,stream,indent=2,allow_nan=False);stream.write('\n')


def checked(record):
    name=record['path'];relative=Path(name)
    if relative.is_absolute() or '..' in relative.parts or relative.as_posix()!=name:
        raise ValueError('Canonical repository-relative record required')
    path=(ROOT/relative).resolve()
    if not path.is_relative_to(ROOT) or sha(path)!=record['sha256']:
        raise ValueError('Pinned metadata changed: '+record['path'])
    return path


def record(path):
    return dict(path=Path(path).resolve().relative_to(ROOT).as_posix(),sha256=sha(path))


def validate_execution_inputs(path,digest):
    try:
        from run_point_rotation_analysis import validate_execution
    except ImportError:
        from experiments.run_point_rotation_analysis import validate_execution
    # CPU aggregation does not execute CUDA kernels; source/input closure is
    # identical to the GPU runner, while its device environment need not match.
    canonical=(ROOT/Path(path)).resolve().relative_to(ROOT.resolve()).as_posix()
    return validate_execution(canonical,digest,check_environment=False)


def eligible(plan,role,execution_sha,method_sha,**kwargs):
    try:
        from run_point_rotation_analysis import eligible_objects
    except ImportError:
        from experiments.run_point_rotation_analysis import eligible_objects
    return eligible_objects(plan,role,execution_sha256=execution_sha,method_sha256=method_sha,**kwargs)


def check_execution(path,digest,*,expected_output=None):
    execution,method,plan=validate_execution_inputs(path,digest)
    if execution.get('schema')!='point-rotation-analysis-execution-v0.2':
        raise ValueError('New rotation-aware execution required')
    checked(execution['method'])
    if expected_output is not None and (ROOT/execution['analysis_output']).resolve()!=Path(expected_output).resolve():
        raise ValueError('Execution output directory differs from analyzed results')
    own=Path(__file__).resolve().relative_to(ROOT.resolve()).as_posix()
    if execution['source_sha256'].get(own)!=sha(__file__):
        raise ValueError('Aggregator must be pinned before calibration data use')
    for name,value in execution['source_sha256'].items():checked(dict(path=name,sha256=value))
    return execution,method,plan


def load_results(output,objects,role,execution,execution_sha):
    method_sha=execution['method']['sha256'];paths=[]
    # Validate every state/choice freeze BEFORE reading any object result values.
    for obj in objects:
        index=obj['index'];directory=Path(output)/'objects'/obj['object_id']
        marker_path=directory/'posterior-states-frozen.json';marker=read(marker_path)
        if (marker.get('object_index')!=index or marker.get('role')!=role
            or marker.get('method_sha256')!=method_sha
            or marker.get('execution_protocol_sha256')!=execution_sha
            or marker.get('source_sha256')!=execution['source_sha256']
            or marker.get('state_names')!=list(STATE_NAMES)
            or marker.get('future_A_or_B_truth_opened') is not False
            or marker.get('material_or_support_truth_opened') is not False):
            raise ValueError('All prescribed predictive states must freeze before future truth')
        if checked(marker['states'])!=(directory/'predictive-states.npz').resolve():
            raise ValueError('Predictive states must bind the same object')
        if checked(marker['active_choice'])!=(directory/'active-choice-frozen.json').resolve():
            raise ValueError('Active choice must bind the same object')
        choice=read(checked(marker['active_choice']))
        if (choice.get('object_index')!=index or choice.get('role')!=role
            or choice.get('method_sha256')!=method_sha
            or choice.get('execution_protocol_sha256')!=execution_sha
            or choice.get('source_sha256')!=execution['source_sha256']
            or choice.get('actual_frame24_opened') is not False):
            raise ValueError('Choice must be frozen before acquired-view truth')
        paths.append((obj,directory/'result.json',record(marker_path),choice))
    results=[];records=[]
    for obj,path,marker_rec,choice in paths:
        index=obj['index'];value=read(path)
        if (value.get('object_index')!=index or value.get('object_id')!=obj['object_id']
            or value.get('role')!=role or value.get('completed') is not True
            or value.get('method_sha256')!=method_sha
            or value.get('execution_protocol_sha256')!=execution_sha
            or value.get('posterior_frozen_before_future_truth') is not True
            or value.get('posterior_freeze')!=marker_rec or set(value.get('states',{}))!=set(STATE_NAMES)):
            raise ValueError('All complete prescribed role-bound results required; no filtering')
        for label in STATE_NAMES:
            state=value['states'][label]
            numbers=[state[k] for k in ['A_after1s_MSE_m2','A_after1p5s_MSE_m2','B_all_MSE_m2','individual_band_score']]
            if not np.isfinite(numbers).all() or np.any(np.asarray(numbers)<0):
                raise ValueError('Finite nonnegative complete field metrics required')
            if 'mean_target_scale_m' in state and (not np.isfinite(state['mean_target_scale_m']) or state['mean_target_scale_m']<1e-9):
                raise ValueError('Band width must use the same floored target scale as scores')
        scores=[s['individual_band_score'] for s in value['states'].values()]
        if not np.isfinite(value['joint_band_score']) or not np.isclose(value['joint_band_score'],max(scores),rtol=0,atol=1e-12):
            raise ValueError('One joint five-state family score per object required')
        for key,choice_key in [('selected_active_view','selected_index'),('selected_random_view','selected_random_view')]:
            if type(value[key]) is not int or not 0<=value[key]<4 or value[key]!=choice[choice_key]:
                raise ValueError('One unchanged declared view index per policy required')
        results.append(value);records.append(record(path))
    return results,records


def freeze_calibration(output,execution_path,execution_sha):
    execution,_,plan=check_execution(execution_path,execution_sha,expected_output=output)
    objects=eligible(plan,'CALIBRATION',execution_sha,execution['method']['sha256'])
    values,records=load_results(output,objects,'CALIBRATION',execution,execution_sha)
    band=calibrate([v['joint_band_score'] for v in values],alpha=.1,numerical_floor_m=1e-9)
    fixed_mse=[float(np.mean([v['states'][f'fixed_view_{j}']['A_after1p5s_MSE_m2'] for v in values])) for j in range(4)]
    result=dict(schema=CALIBRATION_SCHEMA,completed=True,
        method_sha256=execution['method']['sha256'],execution_protocol_sha256=execution_sha,
        execution_protocol=record(execution_path),method=execution['method'],calibration_object_indices=list(range(16)),
        calibration_results=records,band_calibration=band.metadata(),object_scores=list(band.scores),
        best_fixed_view_index=int(np.argmin(fixed_mse)),fixed_camera_calibration_mean_MSE_m2=fixed_mse,
        source_sha256={str(Path(__file__).relative_to(ROOT)):sha(__file__),
            'experiments/campaign_simultaneous_bands.py':sha(ROOT/'experiments/campaign_simultaneous_bands.py')},
        reserved_response_or_truth_opened=False,calibration_target='Joint fixed finite mesh/time/potential-view family',
        scope='Calibrated marginal predictive band, not a Bayesian material probability or continuum-field guarantee')
    path=Path(output)/'calibration-frozen.json';write_new(path,result)
    return record(path)


def validate_freeze(calibration_record,execution_sha=None):
    freeze=read(checked(calibration_record))
    if (freeze.get('schema')!=CALIBRATION_SCHEMA
            or freeze.get('completed') is not True or freeze.get('method_sha256')!=freeze.get('method',{}).get('sha256')
            or freeze.get('calibration_object_indices')!=list(range(16))
            or len(freeze.get('calibration_results',[]))!=16 or len(freeze.get('object_scores',[]))!=16
            or freeze.get('reserved_response_or_truth_opened') is not False
            or (execution_sha is not None and freeze.get('execution_protocol_sha256')!=execution_sha)):
        raise ValueError('Exact sixteen-object calibration freeze required')
    for artifact in freeze['calibration_results']:checked(artifact)
    execution,_,plan=check_execution(checked(freeze['execution_protocol']),freeze['execution_protocol_sha256'])
    if freeze['method']!=execution['method']:raise ValueError('Calibration method differs from execution')
    objects=eligible(plan,'CALIBRATION',freeze['execution_protocol_sha256'],freeze['method_sha256'])
    values,records=load_results(ROOT/execution['analysis_output'],objects,'CALIBRATION',execution,freeze['execution_protocol_sha256'])
    if records!=freeze['calibration_results'] or [v['joint_band_score'] for v in values]!=freeze['object_scores']:
        raise ValueError('Calibration object bindings/scores changed')
    band=calibrate(freeze['object_scores'],alpha=.1,numerical_floor_m=1e-9)
    if freeze['band_calibration']!=band.metadata():raise ValueError('Calibration band metadata changed')
    fixed=[float(np.mean([v['states'][f'fixed_view_{j}']['A_after1p5s_MSE_m2'] for v in values])) for j in range(4)]
    if freeze['fixed_camera_calibration_mean_MSE_m2']!=fixed or freeze['best_fixed_view_index']!=int(np.argmin(fixed)):
        raise ValueError('Fixed camera must use calibration only and stable lowest-index tie')
    for name,digest in freeze['source_sha256'].items():
        if sha(ROOT/name)!=digest:raise ValueError('Calibration/aggregation source changed')
    return freeze


def release_reserved(calibration_record,output):
    freeze=validate_freeze(calibration_record)
    result=dict(schema=RELEASE_SCHEMA,status='FROZEN_EXPLICIT_RESERVED_EVALUATION_RELEASE',
        method_sha256=freeze['method_sha256'],execution_protocol_sha256=freeze['execution_protocol_sha256'],
        method=freeze['method'],calibration_freeze=calibration_record,object_indices=list(range(16,48)),
        no_reserved_response_or_truth_used_for_calibration=True)
    write_new(output,result);return record(output)


def paired_bootstrap(difference,*,replicates=10000,seed=2026120600):
    x=np.asarray(difference,dtype=float)
    if x.ndim!=1 or not len(x) or not np.isfinite(x).all():raise ValueError('Finite complete object differences required')
    ids=np.random.default_rng(seed).integers(0,len(x),size=(replicates,len(x)))
    means=x[ids].mean(1)
    return dict(mean=float(x.mean()),percentile_95_interval=np.quantile(means,[.025,.975]).tolist(),
        independent_unit='complete object',replicates=replicates,seed=seed)


def policy_state(value,policy,best):
    if policy=='active':return value['states'][f"fixed_view_{value['selected_active_view']}"]
    if policy=='random':return value['states'][f"fixed_view_{value['selected_random_view']}"]
    if policy=='best_fixed_on_calibration':return value['states'][f'fixed_view_{best}']
    return value['states'][policy]


def evaluate_reserved(output,release_record):
    release=read(checked(release_record))
    if (release.get('schema')!=RELEASE_SCHEMA
            or release.get('status')!='FROZEN_EXPLICIT_RESERVED_EVALUATION_RELEASE'
            or release.get('method_sha256')!=release.get('method',{}).get('sha256')
            or release.get('no_reserved_response_or_truth_used_for_calibration') is not True or release.get('object_indices')!=list(range(16,48))):
        raise ValueError('Exact reserved release required')
    freeze=validate_freeze(release['calibration_freeze'],release['execution_protocol_sha256'])
    execution,_,plan=check_execution(checked(freeze['execution_protocol']),release['execution_protocol_sha256'],expected_output=output)
    if release['method']!=execution['method']:raise ValueError('Reserved release method mismatch')
    objects=eligible(plan,'RESERVED_EVALUATION',release['execution_protocol_sha256'],execution['method']['sha256'],
        release=release,calibration_freeze=freeze,calibration_freeze_record=release['calibration_freeze'])
    values,records=load_results(output,objects,'RESERVED_EVALUATION',execution,release['execution_protocol_sha256'])
    scale=freeze['band_calibration']['scale']
    if scale is None:raise ValueError('Infinite bands require explicit reporting; never replace by finite scale')
    covered=sum(v['joint_band_score']<=scale for v in values)
    ci=binomtest(covered,len(values)).proportion_ci(confidence_level=.95,method='exact')
    best=freeze['best_fixed_view_index'];policies=['no_additional_view',*[f'fixed_view_{j}' for j in range(4)],'best_fixed_on_calibration','random','active']
    arrays={};summaries={}
    for policy in policies:
        rows=[policy_state(v,policy,best) for v in values]
        metrics={k:np.array([r[k] for r in rows]) for k in ['A_after1p5s_MSE_m2','B_all_MSE_m2']}
        arrays[policy]=metrics
        summaries[policy]={key:dict(pooled_coordinate_RMSE_mm=float(1000*np.sqrt(x.mean())),
            mean_object_RMSE_mm=float(1000*np.sqrt(x).mean())) for key,x in metrics.items()}
        if all('mean_target_scale_m' in r for r in rows):
            summaries[policy]['mean_calibrated_full_bandwidth_m']=float(2*scale*np.mean([r['mean_target_scale_m'] for r in rows]))
    paired={}
    for baseline in ['best_fixed_on_calibration','random','no_additional_view']:
        paired[baseline]={}
        for metric in ['A_after1p5s_MSE_m2','B_all_MSE_m2']:
            active,other=arrays['active'][metric],arrays[baseline][metric]
            paired[baseline][metric]=dict(MSE_difference_m2=paired_bootstrap(active-other),
                object_RMSE_difference_mm=paired_bootstrap(1000*(np.sqrt(active)-np.sqrt(other))),
                interpretation='Negative favors active; same target frames and one independent object per replicate')
    result=dict(schema='point-rotation-reserved-view-evaluation-v0.2',completed=True,
        method_sha256=freeze['method_sha256'],execution_protocol_sha256=release['execution_protocol_sha256'],
        release=release_record,calibration_freeze=release['calibration_freeze'],reserved_results=records,
        objects=32,best_fixed_view_index_frozen_on_calibration=best,
        joint_simultaneous_band_coverage=dict(covered=covered,total=32,fraction=covered/32,
            exact_binomial_95_interval=[float(ci.low),float(ci.high)],scale=scale,
            scope='Joint fixed mesh/time/five-potential-state family, one held-out object is one trial'),
        prediction=summaries,paired_active_minus_baseline=paired,
        action_target_comparison='A31..60 for EVERY policy; B0..60 for EVERY policy',
        limitations=['Fixed H0 and known point-pin type','Rotation-aware approximate mechanics and fitted working Gaussian discrepancy',
            'Independent calibration/test objects; discrepancy development objects excluded',
            'No conditional material/support coverage or calibrated Bayesian posterior claim',
            'Virtual view selection, no continuing-state poke or robot camera motion planning',
            'Thirty-two independent evaluation objects give noisy estimates'])
    materials={}
    if all('rawmaterialintervalchecks' in v for v in values):
        for name in [r['name'] for r in values[0]['rawmaterialintervalchecks']]:
            checks=[next(r for r in v['rawmaterialintervalchecks'] if r['name']==name) for v in values]
            count=sum(bool(r['in_90_percent_interval']) for r in checks)
            interval=binomtest(count,32).proportion_ci(confidence_level=.95,method='exact')
            materials[name]=dict(covered=count,total=32,fraction=count/32,
                exact_binomial_95_interval=[float(interval.low),float(interval.high)])
    result['raw_90_percent_material_interval_coverage']=materials
    result['material_coverage_scope']='Unadjusted working posterior intervals; descriptive reserved-case coverage, not a calibrated Bayesian posterior claim'
    if all('supporttruthprobability' in v and 'supporttruthrank' in v for v in values):
        result['support']=dict(mean_true_node_probability=float(np.mean([v['supporttruthprobability'] for v in values])),
            true_node_rank1=sum(v['supporttruthrank']==1 for v in values),objects=32)
    if all('reference_stability' in v for v in values):
        result['numerical_reference_diagnostics']=dict(
            minimum_importance_ESS=min(v['reference_stability']['minimum_largest_run_ESS'] for v in values),
            maximum_cross_seed_support_TV=max(v['reference_stability']['maximum_independent_seed_support_total_variation'] for v in values),
            qualification='Retained diagnostics for all32 cases; no filtering or convergence certificate')
    result['view_empirical_update_ESS']={f'fixed_view_{j}':dict(
        minimum=min(v['states'][f'fixed_view_{j}']['view_update_ESS'] for v in values),
        mean=float(np.mean([v['states'][f'fixed_view_{j}']['view_update_ESS'] for v in values])))
        for j in range(4) if all('view_update_ESS' in v['states'][f'fixed_view_{j}'] for v in values)}
    path=Path(output)/'reserved-evaluation.json';write_new(path,result);return record(path)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='action',required=True)
    p=sub.add_parser('freeze-calibration');p.add_argument('--output',required=True);p.add_argument('--execution',required=True);p.add_argument('--execution-sha256',required=True)
    p=sub.add_parser('release-reserved');p.add_argument('--calibration',required=True);p.add_argument('--calibration-sha256',required=True);p.add_argument('--output',required=True)
    p=sub.add_parser('evaluate-reserved');p.add_argument('--output',required=True);p.add_argument('--release',required=True);p.add_argument('--release-sha256',required=True)
    a=parser.parse_args()
    if a.action=='freeze-calibration':result=freeze_calibration(a.output,a.execution,a.execution_sha256)
    elif a.action=='release-reserved':result=release_reserved(dict(path=a.calibration,sha256=a.calibration_sha256),a.output)
    else:result=evaluate_reserved(a.output,dict(path=a.release,sha256=a.release_sha256))
    print(json.dumps(result))
