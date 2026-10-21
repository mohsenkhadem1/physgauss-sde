"""Post-seal descriptive material/support audit for the v0.2 rotation model.

No model, sensor, inference, resampling or calibration is performed. The same
2048 empirical particles supply all five states. Whole-stage seals and every
state/choice/empirical hash are checked before particle arrays or private labels.
Raw per-object diagnostics remain local; only aggregate summaries are published.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
try:
    import campaign_point_postview_material_audit as math_audit
    import summarize_point_rotation_analysis as aggregate
except ImportError:
    from experiments import campaign_point_postview_material_audit as math_audit
    from experiments import summarize_point_rotation_analysis as aggregate

ROOT=Path(__file__).resolve().parents[1]
STATE_NAMES=math_audit.STATE_NAMES
MATERIAL_KEYS=math_audit.MATERIAL_KEYS
MATERIAL_NAMES=math_audit.MATERIAL_NAMES
RECIPE_SCHEMA='point-rotation-material-audit-recipe-v0.2'
SOURCE_NAMES=('campaign_point_rotation_material_audit.py','campaign_point_postview_material_audit.py',
    'summarize_point_rotation_analysis.py','run_point_rotation_analysis.py','campaign_simultaneous_bands.py')


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text())
def write(path,value):Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
def resolve(name):
    p=Path(name)
    if p.is_absolute() or '..' in p.parts or p.as_posix()!=str(name):raise ValueError('Canonical repository-relative path required')
    path=(ROOT/p).resolve()
    if not path.is_relative_to(ROOT.resolve()):raise ValueError('Repository escape')
    return path

def record(path):return dict(path=Path(path).resolve().relative_to(ROOT.resolve()).as_posix(),sha256=sha(path))
def checked(rec):
    path=resolve(rec['path'])
    if sha(path)!=rec['sha256']:raise ValueError('Frozen audit artifact changed: '+rec['path'])
    return path


def prepare_recipe(path,execution_record,method_record,output_prefix):
    """Write source/choices declaration only; never read execution or case values."""
    target=resolve(path)
    if target.exists():raise ValueError('Existing audit declaration retained')
    resolve(output_prefix)
    sources={'experiments/'+name:sha(ROOT/'experiments'/name) for name in SOURCE_NAMES}
    recipe=dict(schema=RECIPE_SCHEMA,execution_protocol=execution_record,method=method_record,
        roles=['CALIBRATION','RESERVED_EVALUATION'],particles=2048,states=list(STATE_NAMES),
        prior_categories=55,output_prefix=output_prefix,source_sha256=sources,
        require_completed_primary_seal_before_any_values=True,execute_authorization='ROOT_SEPARATE_AFTER_PRIMARY_SEAL',
        choices=dict(interval_probabilities=[.05,.95],quantiles='Inverse empirical CDF, zero weights skipped, no interpolation',
            support_rank='Probability descending then global node ID ascending; all55 including zeros',
            confidence_threshold=.9,lowest_ESS_objects=5,relative_error='Absolute posterior mean error divided by positive physical truth'),
        claim='Descriptive empirical marginal coverage and confidence, not Bayesian or conformal parameter coverage',
        no_model_queries_or_tuning=True,raw_object_diagnostics_LOCAL=True)
    target.parent.mkdir(parents=True,exist_ok=True);write(target,recipe);return record(target)


def validate_recipe(path,digest):
    recipe=read(checked(dict(path=path,sha256=digest)))
    if (recipe.get('schema')!=RECIPE_SCHEMA or recipe.get('roles')!=['CALIBRATION','RESERVED_EVALUATION']
        or recipe.get('particles')!=2048 or recipe.get('states')!=list(STATE_NAMES)
        or recipe.get('prior_categories')!=55 or recipe.get('require_completed_primary_seal_before_any_values') is not True
        or recipe.get('execute_authorization')!='ROOT_SEPARATE_AFTER_PRIMARY_SEAL'
        or recipe.get('choices')!=dict(interval_probabilities=[.05,.95],quantiles='Inverse empirical CDF, zero weights skipped, no interpolation',
            support_rank='Probability descending then global node ID ascending; all55 including zeros',
            confidence_threshold=.9,lowest_ESS_objects=5,relative_error='Absolute posterior mean error divided by positive physical truth')
        or recipe.get('no_model_queries_or_tuning') is not True):raise ValueError('Exact frozen audit recipe required')
    expected={'experiments/'+name:sha(ROOT/'experiments'/name) for name in SOURCE_NAMES}
    if recipe.get('source_sha256')!=expected:raise ValueError('Audit source closure changed')
    for name,digest in expected.items():checked(dict(path=name,sha256=digest))
    return recipe


def preflight_saved_objects(objects,execution,execution_sha,role):
    """ALL markers/hashes before ANY result payload, NPZ member or private label."""
    expected=list(range(16)) if role=='CALIBRATION' else list(range(16,48)) if role=='RESERVED_EVALUATION' else None
    if expected is None or [o['index'] for o in objects]!=expected:raise ValueError('Exact16/32 role objects required')
    method_sha=execution['method']['sha256'];output=resolve(execution['analysis_output'])
    status_path=output/('calibration-execution.json' if role=='CALIBRATION' else 'reserved-execution.json')
    status=read(status_path)
    if (status.get('role')!=role or status.get('terminal') is not True or status.get('completed') is not True
        or status.get('method_sha256')!=method_sha or status.get('execution_protocol_sha256')!=execution_sha
        or status.get('planned_object_indices')!=expected or status.get('completed_object_indices')!=expected
        or status.get('unlaunched_object_indices')!=[] or status.get('failures')!=[]):
        raise ValueError('Entire selected-role primary stage must be terminal and complete')
    records=[]
    for obj in objects:
        directory=output/'objects'/obj['object_id'];marker_path=directory/'posterior-states-frozen.json';marker=read(marker_path)
        identity=dict(object_index=obj['index'],role=role,method_sha256=method_sha,execution_protocol_sha256=execution_sha)
        if (any(marker.get(k)!=v for k,v in identity.items()) or marker.get('source_sha256')!=execution['source_sha256']
            or marker.get('state_names')!=list(STATE_NAMES) or marker.get('future_A_or_B_truth_opened') is not False
            or marker.get('material_or_support_truth_opened') is not False):raise ValueError('Exact frozen-state identity required')
        for key,filename in [('states','predictive-states.npz'),('active_choice','active-choice-frozen.json'),('empirical_posterior','empirical-posterior.npz')]:
            if checked(marker[key])!=(directory/filename).resolve():raise ValueError('Same-object state/choice/particle path required')
        records.append(dict(object=obj,result=record(directory/'result.json'),posterior_freeze=record(marker_path),
            states=marker['states'],choice=marker['active_choice'],empirical=marker['empirical_posterior']))
    # Shared aggregator checks every choice/source/state marker before reading results.
    values,result_records=aggregate.load_results(output,objects,role,execution,execution_sha)
    if result_records!=[r['result'] for r in records]:raise ValueError('Result records changed during preflight')
    for rec,value in zip(records,values):
        rec.update(selected_active_view=value['selected_active_view'],selected_random_view=value['selected_random_view'])
    return records,record(status_path)


def gate_stage(recipe,role,calibration_record,*,release_record=None,primary_record=None):
    """Read only hash-bound metadata/results until complete primary-stage seal."""
    if calibration_record is None:raise ValueError('Completed16 calibration freeze required for either role')
    execution_record=recipe['execution_protocol'];execution_sha=execution_record['sha256']
    execution,_,plan=aggregate.check_execution(checked(execution_record),execution_sha)
    if execution['method']!=recipe['method']:raise ValueError('Audit must use declared new method')
    freeze=aggregate.validate_freeze(calibration_record,execution_sha)
    if freeze['method']!=recipe['method']:raise ValueError('Calibration method mismatch')
    release=primary=None
    if role=='CALIBRATION':
        if release_record is not None or primary_record is not None:raise ValueError('Calibration cannot read reserved release/results')
        objects=aggregate.eligible(plan,role,execution_sha,recipe['method']['sha256'])
        sealed_results=freeze['calibration_results']
    elif role=='RESERVED_EVALUATION':
        if release_record is None or primary_record is None:raise ValueError('Completed32 primary aggregate and explicit reserved release required')
        release=read(checked(release_record))
        if (release.get('schema')!=aggregate.RELEASE_SCHEMA or release.get('status')!='FROZEN_EXPLICIT_RESERVED_EVALUATION_RELEASE'
            or release.get('method')!=recipe['method'] or release.get('method_sha256')!=recipe['method']['sha256']
            or release.get('execution_protocol_sha256')!=execution_sha or release.get('object_indices')!=list(range(16,48))
            or release.get('calibration_freeze')!=calibration_record
            or release.get('no_reserved_response_or_truth_used_for_calibration') is not True):raise ValueError('Exact new reserved release required')
        primary=read(checked(primary_record))
        if (primary.get('schema')!='point-rotation-reserved-view-evaluation-v0.2' or primary.get('completed') is not True
            or primary.get('objects')!=32 or primary.get('method_sha256')!=recipe['method']['sha256']
            or primary.get('execution_protocol_sha256')!=execution_sha or primary.get('release')!=release_record
            or primary.get('calibration_freeze')!=calibration_record):raise ValueError('Completed hash-bound32 primary aggregate required')
        objects=aggregate.eligible(plan,role,execution_sha,recipe['method']['sha256'],release=release,
            calibration_freeze=freeze,calibration_freeze_record=calibration_record)
        sealed_results=primary['reserved_results']
    else:raise ValueError('Explicit CALIBRATION or RESERVED_EVALUATION role required')
    records,status=preflight_saved_objects(objects,execution,execution_sha,role)
    if [r['result'] for r in records]!=sealed_results:raise ValueError('All result records must match completed primary seal')
    # Exact55 catalogue is public, read without loading any cache tensors.
    cache=read(checked(execution['cache']));distribution=read(checked(cache['distribution']))
    catalogue=np.asarray([m['node_id'] for m in distribution['masks']],dtype=np.int64)
    if catalogue.shape!=(55,) or len(np.unique(catalogue))!=55:raise ValueError('All55 prior categories required')
    return records,status,freeze,catalogue


def summarize_comparisons(objects,*,best_fixed):
    """All policies, paired interval losses/errors and retained ESS/support tails."""
    result=math_audit.summarize_policy_comparisons(objects,best_fixed=best_fixed)
    truth=np.asarray([o['material_truth'] for o in objects]);base=np.asarray([[m['mean'] for m in o['diagnostic']['states']['no_additional_view']['material']] for o in objects])
    base_errors=np.abs(base-truth)
    for policy,summary in result.items():
        rows=[]
        for obj in objects:
            name=f"fixed_view_{obj['selected_active_view']}" if policy=='active' else f"fixed_view_{obj['selected_random_view']}" if policy=='random' else f'fixed_view_{best_fixed}' if policy=='best_fixed_on_calibration' else policy
            rows.append(obj['diagnostic']['states'][name])
        for j,m in enumerate(summary['material']):
            means=np.asarray([row['material'][j]['mean'] for row in rows]);error=means-truth[:,j];absolute=np.abs(error)
            m.update(posterior_mean_MAE=float(absolute.mean()),posterior_mean_RMSE=float(np.sqrt(np.mean(error**2))),
                posterior_mean_bias=float(error.mean()),posterior_mean_relative_MAE=float(np.mean(absolute/truth[:,j])),
                paired_mean_absolute_error_change_from_noaction=float(np.mean(absolute-base_errors[:,j])),
                mean_error_improved_objects=int(np.sum(absolute<base_errors[:,j])),mean_error_worse_objects=int(np.sum(absolute>base_errors[:,j])))
        ess=np.asarray([r['empirical_ESS'] for r in rows]);prob=np.asarray([r['support_true_node_probability'] for r in rows]);confidence=np.asarray([r['support_mode_probability'] for r in rows]);ranks=np.asarray([r['support_true_node_rank'] for r in rows]);wrong=ranks!=1
        summary.update(empirical_row_ESS=dict(minimum=float(ess.min()),quantile05=float(np.quantile(ess,.05)),median=float(np.median(ess)),mean=float(ess.mean()),maximum=float(ess.max())),
            true_support_probability=dict(minimum=float(prob.min()),quantile05=float(np.quantile(prob,.05)),median=float(np.median(prob)),mean=float(prob.mean()),maximum=float(prob.max()),exactzero_objects=int(np.sum(prob==0))),
            support_mode_confidence=dict(mean=float(confidence.mean()),maximum=float(confidence.max()),wrong_mode_objects=int(wrong.sum()),
                mean_confidence_when_wrong=float(confidence[wrong].mean()) if wrong.any() else None,
                confidently_wrong_at_least_90_percent_objects=int(np.sum(wrong&(confidence>=.9)))),
            common_empirical_geometry_ids=dict(minimum=min(r['unique_geometry_ids_in_common_particles'] for r in rows),mean=float(np.mean([r['unique_geometry_ids_in_common_particles'] for r in rows]))),
            lowest_ESS_objects=[dict(object_index=objects[int(i)]['object_index'],ESS=float(ess[i]),largest_particle_weight=rows[int(i)]['maximum_particle_weight'],
                true_support_probability=float(prob[i]),true_support_rank=int(ranks[i]),mode_support_probability=float(confidence[i]),positive_weight_geometry_ids=rows[int(i)]['positive_weight_geometry_ids']) for i in np.argsort(ess,kind='stable')[:5]])
    return result


def run_offline_audit(recipe_path,recipe_sha256,role,*,calibration_record,release_record=None,primary_record=None):
    recipe=validate_recipe(recipe_path,recipe_sha256)
    records,status,freeze,catalogue=gate_stage(recipe,role,calibration_record,release_record=release_record,primary_record=primary_record)
    output=resolve(recipe['output_prefix'])/role
    if output.exists():raise ValueError('Existing diagnostic retained; no overwrite/retry')
    output.mkdir(parents=True)
    state=dict(role=role,terminal=False,completed=False,recipe_sha256=recipe_sha256,private_values_opened=False)
    write(output/'execution.json',state);reports=[]
    try:
        for rec in records:
            # EVERY role object/state/choice/primary result is sealed before here.
            state['private_values_opened']=True;write(output/'execution.json',state)
            with np.load(checked(rec['empirical']),allow_pickle=False) as data:
                theta=data['physical_coordinates'];geometry=data['geometry_ids']
                weights={name:data[name+'__weights'] for name in STATE_NAMES}
            spec=read(checked(rec['object']['realized_hypothesis']))
            if spec['boundary_kind']!='point_pin' or len(spec['fixture_nodes'])!=1:raise ValueError('Known point-pin class required')
            truth=[spec['material'][key] for key in MATERIAL_KEYS]
            diagnostic=math_audit.summarize_postview_material(theta,geometry,weights,truth,int(spec['fixture_nodes'][0]),catalogue)
            diagnostic.pop('C28_full_reference_intervals_unchanged',None)
            reports.append(dict(object_index=rec['object']['index'],role=role,selected_active_view=rec['selected_active_view'],selected_random_view=rec['selected_random_view'],
                material_truth=truth,diagnostic=diagnostic,private_hypothesis=rec['object']['realized_hypothesis'],
                source_artifacts={k:rec[k] for k in ['result','posterior_freeze','states','choice','empirical']}))
        write(output/'object-diagnostics-LOCAL.json',reports)
        summary=dict(schema='point-rotation-postview-material-audit-v0.2',completed=True,role=role,objects=len(reports),all_object_indices=[r['object_index'] for r in reports],
            recipe=record(resolve(recipe_path)),execution_protocol=recipe['execution_protocol'],method=recipe['method'],calibration_freeze=calibration_record,
            reserved_release=release_record,sealed_primary_evaluation=primary_record,original_stage_terminal=status,same2048_particle_comparison=True,
            prior_geometry_ids_including_zeros=sorted(map(int,catalogue)),policies=summarize_comparisons(reports,best_fixed=freeze['best_fixed_view_index']),
            source_sha256=recipe['source_sha256'],quantile_convention=reports[0]['diagnostic']['quantile_convention'],
            limitations=['All prescribed16/32 objects retained, all55 categories including zeros; no model/query/tuning/retraction evaluation.',
                'SAME2048 resampled empirical particles; row ESS is not an independent hypothesis count.',
                'Marginal empirical5/95 parameter interval coverage is descriptive; not conformal field coverage or a Bayesian guarantee.',
                'Full-reference material intervals stay separate and unchanged; target discrepancy does not calibrate material probabilities.'],
            raw_object_diagnostics_LOCAL=True,new_models_or_queries=False)
        write(output/'summary.json',summary);state.update(terminal=True,completed=True,objects=len(reports));write(output/'execution.json',state)
        return record(output/'summary.json')
    except BaseException as exc:
        write(output/'failure.json',dict(completed=False,error=type(exc).__name__,message=str(exc),completed_objects=len(reports)))
        state.update(terminal=True,completed=False);write(output/'execution.json',state);raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='action',required=True)
    prepare=sub.add_parser('prepare')
    for key in ['output','output-prefix','execution','execution-sha256','method','method-sha256']:prepare.add_argument('--'+key,required=True)
    run=sub.add_parser('run')
    for key in ['recipe','recipe-sha256','role','calibration','calibration-sha256']:run.add_argument('--'+key,required=True)
    for key in ['release','release-sha256','primary','primary-sha256']:run.add_argument('--'+key)
    a=p.parse_args()
    def optional(path,digest):
        if (path is None)!=(digest is None):raise ValueError('Artifact path AND hash required together')
        return None if path is None else dict(path=path,sha256=digest)
    if a.action=='prepare':r=prepare_recipe(a.output,dict(path=a.execution,sha256=a.execution_sha256),dict(path=a.method,sha256=a.method_sha256),a.output_prefix)
    else:r=run_offline_audit(a.recipe,a.recipe_sha256,a.role,calibration_record=optional(a.calibration,a.calibration_sha256),
        release_record=optional(a.release,a.release_sha256),primary_record=optional(a.primary,a.primary_sha256))
    print(json.dumps(r))
