"""Diagnostic material/support intervals on SAME2048 particles before/afterviews.

Pure summary first; CLI is separately role/hash gated. This does not alter the
C28 full-reference intervals, infer new particles, tune likelihoods or calibrate
Bayesian material probabilities. All declared55 support categories includezeros.
"""
from __future__ import annotations
import numpy as np

STATE_NAMES=('no_additional_view',*(f'fixed_view_{i}' for i in range(4)))
MATERIAL_NAMES=('E_Pa','rho_kg_m3','nu','body_drag_s_inverse')
MATERIAL_KEYS=('youngs_modulus_Pa','density_kg_m3','poissons_ratio','body_linear_drag_s_inverse')


def _normalized(weights,n):
    w=np.asarray(weights,dtype=np.float64)
    if w.shape!=(n,) or not np.isfinite(w).all() or (w<0).any() or abs(w.sum()-1)>64*np.finfo(float).eps:
        raise ValueError('Finite normalized nonnegative empiricalweights required')
    return w/w.sum()


def _quantiles(values,weights,probabilities=(.05,.5,.95)):
    order=np.argsort(values,kind='stable');positive=weights[order]>0
    x=values[order][positive];w=weights[order][positive];cdf=np.cumsum(w);cdf[-1]=1.
    # Left-continuous inverse empiricalCDF, with no invented mass/interpolation.
    return x[np.searchsorted(cdf,np.asarray(probabilities),side='left')]


def summarize_postview_material(theta,geometry_ids,state_weights,material_truth,true_node,prior_geometry_ids):
    """Fiveweighted summaries on a common2048-particle empirical posterior.

    Intervals are marginal5/95% inverse-empirical-CDF intervals, not joint or
    conformal intervals. Zero-weight categories remain in the55-row table.
    """
    x=np.asarray(theta);ids=np.asarray(geometry_ids);truth=np.asarray(material_truth,dtype=np.float64)
    catalogue=np.asarray(prior_geometry_ids)
    if (x.shape!=(2048,4) or x.dtype!=np.dtype('float64') or not np.isfinite(x).all()
        or ids.shape!=(2048,) or ids.dtype.kind not in 'iu' or truth.shape!=(4,) or not np.isfinite(truth).all()
        or catalogue.shape!=(55,) or catalogue.dtype.kind not in 'iu' or len(np.unique(catalogue))!=55
        or not set(np.unique(ids)).issubset(set(catalogue)) or type(true_node) is not int or true_node not in catalogue
        or set(state_weights)!=set(STATE_NAMES)):
        raise ValueError('Samefloat64[2048,4],integerIDs,all55priorIDs,truth and exactfiveweightvectors required')
    if np.any(x[:,[0,1,3]]<=0) or np.any((x[:,2]<=0)|(x[:,2]>=.5)) or np.any(truth[[0,1,3]]<=0) or not 0<truth[2]<.5:
        raise ValueError('Physical E,rho,nu,drag coordinates required')
    catalogue=np.sort(catalogue);states={}
    if not np.allclose(np.asarray(state_weights['no_additional_view']),1/2048,atol=8*np.finfo(float).eps/2048,rtol=0):
        raise ValueError('Noaction must use SAMEequal empiricalparticleweights')
    for name in STATE_NAMES:
        w=_normalized(state_weights[name],2048);anchor=x[0];delta_mean=w@(x-anchor);mean=anchor+delta_mean;centered=(x-anchor)-delta_mean
        std=np.sqrt(np.sum(w[:,None]*centered**2,axis=0));material=[]
        for j,label in enumerate(MATERIAL_NAMES):
            lo,median,hi=map(float,_quantiles(x[:,j],w))
            material.append(dict(name=label,mean=float(mean[j]),std=float(std[j]),quantile05=lo,median=median,quantile95=hi,
                interval_width=hi-lo,truth_in_90_percent_interval=bool(lo<=truth[j]<=hi)))
        probabilities=np.array([w[ids==node].sum() for node in catalogue]);true_index=int(np.flatnonzero(catalogue==true_node)[0]);ranking=np.lexsort((catalogue,-probabilities))
        categories=[dict(node_id=int(node),posterior_probability=float(p),particle_count=int(np.sum(ids==node)),positive_weight_particle_count=int(np.sum((ids==node)&(w>0)))) for node,p in zip(catalogue,probabilities)]
        positive=probabilities>0
        states[name]=dict(material=material,empirical_ESS=float(1/np.square(w).sum()),maximum_particle_weight=float(w.max()),
            unique_geometry_ids_in_common_particles=int(len(np.unique(ids))),positive_weight_geometry_ids=int(positive.sum()),
            support_true_node_probability=float(probabilities[true_index]),support_true_node_rank=int(np.flatnonzero(ranking==true_index)[0])+1,
            support_mode_node=int(catalogue[ranking[0]]),support_mode_probability=float(probabilities[ranking[0]]),
            support_entropy_nats=float(-np.sum(probabilities[positive]*np.log(probabilities[positive]))),support_categories=categories)
    return dict(particles=2048,declared_prior_categories=55,states=states,
        empirical_particle_set_identical_for_all_states=True,
        quantile_convention='Leftinverse empiricalCDF at.05/.5/.95; zero weights skipped, nointerpolation',
        rank_convention='Descending probability, thenascending globalnodeID ties; all55categories includingzeros',
        C28_full_reference_intervals_unchanged=True,
        claim='Descriptive empiricalmarginal interval/support diagnostic, not Bayesian/conformalparametercoverage guarantee')

# Offline file adapters below are never called by the pure summary function.
from pathlib import Path
import argparse
import hashlib
import json
ROOT=Path(__file__).resolve().parents[1]
METHOD_SHA='ecbc02153336b36ddef33f0c6c01370b012f3ddbeb1d76bd5c71e15145dc3bff'


def _resolve(path):
    p=Path(path)
    if p.is_absolute() or '..' in p.parts or p.as_posix()!=str(path):raise ValueError('Canonical repository-relative path required')
    target=(ROOT/p).resolve()
    if not target.is_relative_to(ROOT):raise ValueError('Repository escape')
    return target


def _sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def _read(path):return json.loads(Path(path).read_text())
def _write(path,value):Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
def _record(path):return dict(path=Path(path).resolve().relative_to(ROOT).as_posix(),sha256=_sha(path))
def _checked(record):
    p=_resolve(record['path'])
    if _sha(p)!=record['sha256']:raise ValueError('Frozen auditartifact changed: '+record['path'])
    return p


def preflight_saved_objects(objects,execution,execution_sha,role):
    """Verify ALL selected states/choices before parsing any result/truth values.

    Global completed-stage metadata is required first. NPZs are hashchecked as
    bytes here; no member arrays or private hypothesis files are opened.
    """
    expected=list(range(16)) if role=='CALIBRATION' else list(range(16,48)) if role=='RESERVED_EVALUATION' else None
    if expected is None or [o['index'] for o in objects]!=expected:raise ValueError('Exact16/32 selectedroleobjects required')
    output=_resolve(execution['analysis_output']);status_name='calibration-execution.json' if role=='CALIBRATION' else 'reserved-execution.json'
    status_path=output/status_name;status=_read(status_path)
    if (status.get('role')!=role or status.get('terminal') is not True or status.get('completed') is not True
        or status.get('method_sha256')!=METHOD_SHA or status.get('execution_protocol_sha256')!=execution_sha
        or status.get('planned_object_indices')!=expected or status.get('completed_object_indices')!=expected
        or status.get('unlaunched_object_indices')!=[] or status.get('failures')!=[]):
        raise ValueError('ALL selectedrole states/results must be completed before auditvalues')
    records=[]
    for obj in objects:
        directory=output/'objects'/obj['object_id'];frozen_path=directory/'posterior-states-frozen.json';frozen=_read(frozen_path)
        identity=dict(object_index=obj['index'],role=role,method_sha256=METHOD_SHA,execution_protocol_sha256=execution_sha)
        if (any(frozen.get(k)!=v for k,v in identity.items()) or frozen.get('source_sha256')!=execution['source_sha256']
            or frozen.get('state_names')!=list(STATE_NAMES) or frozen.get('future_A_or_B_truth_opened') is not False
            or frozen.get('material_or_support_truth_opened') is not False):raise ValueError('Exact frozenposterior marker required')
        for key,filename in [('states','predictive-states.npz'),('active_choice','active-choice-frozen.json')]:
            if frozen[key]['path']!=(directory/filename).relative_to(ROOT).as_posix():raise ValueError('State/choice artifact path mismatch')
            _checked(frozen[key])
        choice=_read(_checked(frozen['active_choice']))
        if (any(choice.get(k)!=v for k,v in identity.items()) or choice.get('source_sha256')!=execution['source_sha256']
            or choice.get('actual_frame24_opened') is not False):raise ValueError('Choice-before-A24 marker required')
        for key,filename in [('empirical_posterior','empirical-posterior.npz'),('early_observation','early-observation.npz')]:
            if choice[key]['path']!=(directory/filename).relative_to(ROOT).as_posix():raise ValueError('Empirical/early artifact path mismatch')
            _checked(choice[key])
        # Hash result bytes now, but parse them only after EVERY marker passes.
        records.append(dict(object=obj,result=_record(directory/'result.json'),posterior_freeze=_record(frozen_path),
            states=frozen['states'],choice=frozen['active_choice'],empirical=choice['empirical_posterior']))
    for record in records:
        obj=record['object'];result=_read(_checked(record['result']))
        if (result.get('completed') is not True or result.get('posterior_frozen_before_future_truth') is not True
            or result.get('object_index')!=obj['index'] or result.get('object_id')!=obj['object_id'] or result.get('role')!=role
            or result.get('method_sha256')!=METHOD_SHA or result.get('execution_protocol_sha256')!=execution_sha
            or result.get('posterior_freeze')!=record['posterior_freeze'] or result.get('active_choice')!=record['choice']
            or set(result.get('states',{}))!=set(STATE_NAMES)):
            raise ValueError('ALL complete source-bound resultmetadata required')
        for key in ['selected_active_view','selected_random_view']:
            if type(result.get(key)) is not int or not 0<=result[key]<4:raise ValueError('Frozenviewchoice required')
        choice=_read(_checked(record['choice']))
        if choice['selected_index']!=result['selected_active_view'] or choice['selected_random_view']!=result['selected_random_view']:
            raise ValueError('Result/prefrozenchoice disagreement')
        record['selected_active_view']=result['selected_active_view'];record['selected_random_view']=result['selected_random_view']
    return records,_record(status_path)


def summarize_policy_comparisons(objects,*,best_fixed=None):
    """Paired same-particle descriptive widths/coverage, no model selection."""
    if not objects:raise ValueError('All prescribedobjects required')
    if best_fixed is not None and (type(best_fixed) is not int or not 0<=best_fixed<4):raise ValueError('Frozen bestfixedcamera required')
    policies=['no_additional_view',*[f'fixed_view_{i}' for i in range(4)],'active','random']
    if best_fixed is not None:policies.append('best_fixed_on_calibration')
    summary={}
    for policy in policies:
        rows=[];bases=[]
        for obj in objects:
            name=f"fixed_view_{obj['selected_active_view']}" if policy=='active' else f"fixed_view_{obj['selected_random_view']}" if policy=='random' else f'fixed_view_{best_fixed}' if policy=='best_fixed_on_calibration' else policy
            states=obj['diagnostic']['states'];rows.append(states[name]);bases.append(states['no_additional_view'])
        material=[]
        for j,label in enumerate(MATERIAL_NAMES):
            widths=np.array([r['material'][j]['interval_width'] for r in rows]);before=np.array([r['material'][j]['interval_width'] for r in bases])
            covered=np.array([r['material'][j]['truth_in_90_percent_interval'] for r in rows]);base_covered=np.array([r['material'][j]['truth_in_90_percent_interval'] for r in bases]);narrower=widths<before
            material.append(dict(name=label,objects=len(rows),covered=int(covered.sum()),empirical_coverage_fraction=float(covered.mean()),
                mean_interval_width=float(widths.mean()),paired_mean_width_change_from_noaction=float((widths-before).mean()),
                coverage_gained_objects=int(np.sum(covered&~base_covered)),coverage_lost_objects=int(np.sum(~covered&base_covered)),
                narrower_objects=int(narrower.sum()),narrower_and_misses_truth_objects=int(np.sum(narrower&~covered)),
                narrower_and_loses_previous_coverage_objects=int(np.sum(narrower&~covered&base_covered))))
        summary[policy]=dict(material=material,mean_empirical_ESS=float(np.mean([r['empirical_ESS'] for r in rows])),
            mean_true_support_probability=float(np.mean([r['support_true_node_probability'] for r in rows])),
            paired_mean_true_support_probability_change=float(np.mean([r['support_true_node_probability']-b['support_true_node_probability'] for r,b in zip(rows,bases)])),
            true_support_rank1_objects=sum(r['support_true_node_rank']==1 for r in rows),mean_mode_support_probability=float(np.mean([r['support_mode_probability'] for r in rows])),
            mean_support_entropy_nats=float(np.mean([r['support_entropy_nats'] for r in rows])),mean_positive_weight_geometry_ids=float(np.mean([r['positive_weight_geometry_ids'] for r in rows])))
    return summary


def run_offline_audit(recipe_path,recipe_sha256,execution_path,role,*,release_record=None,calibration_record=None):
    """Requires explicit root authorization outside this no-training offlineCLI."""
    recipe=_read(_checked(dict(path=recipe_path,sha256=recipe_sha256)))
    if (recipe.get('schema')!='point-postview-material-audit-recipe-v0.1' or recipe.get('method_sha256')!=METHOD_SHA
        or recipe.get('roles')!=['CALIBRATION','RESERVED_EVALUATION'] or recipe.get('particles')!=2048
        or recipe.get('states')!=list(STATE_NAMES) or recipe.get('execute_authorization')!='ROOT_EXPLICIT_SEPARATE_REQUIRED'):
        raise ValueError('Frozen exactauditrecipe required')
    for path,digest in recipe['source_sha256'].items():_checked(dict(path=path,sha256=digest))
    own=Path(__file__).resolve().relative_to(ROOT).as_posix()
    if recipe['source_sha256'].get(own)!=_sha(__file__):raise ValueError('Own auditruntime not frozen')
    try:
        from run_point_prior_analysis import validate_execution,eligible_objects
        from summarize_point_prior_analysis import validate_freeze
    except ImportError:
        from experiments.run_point_prior_analysis import validate_execution,eligible_objects
        from experiments.summarize_point_prior_analysis import validate_freeze
    digest=recipe['execution_protocol_sha256'];execution,method,plan=validate_execution(execution_path,digest,check_environment=False)
    release,freeze=None,None
    if role=='RESERVED_EVALUATION':
        if release_record is None:raise ValueError('Reservedaudit needs explicit hash-bound release')
        release=_read(_checked(release_record));from_release=release.get('calibration_freeze')
        if calibration_record is not None and calibration_record!=from_release:raise ValueError('Conflicting calibrationfreeze')
        calibration_record=from_release;freeze=validate_freeze(calibration_record,digest)
    elif role=='CALIBRATION':
        if release_record is not None:raise ValueError('Calibrationaudit cannot consume reservedrelease')
        if calibration_record is not None:freeze=validate_freeze(calibration_record,digest)
    else:raise ValueError('Explicit16/32 role required')
    objects=eligible_objects(plan,role,execution_sha256=digest,method_sha256=METHOD_SHA,
        release=release,calibration_freeze=freeze if role=='RESERVED_EVALUATION' else None,
        calibration_freeze_record=calibration_record if role=='RESERVED_EVALUATION' else None)
    records,status_record=preflight_saved_objects(objects,execution,digest,role)
    output=_resolve(recipe['output_prefix'])/role
    if output.exists():raise ValueError('Existingaudit retained; no overwrite/retry')
    prior=_read(_checked(execution['support_prior']));catalogue=np.array([m['node_id'] for m in prior['masks']],dtype=np.int64)
    output.mkdir(parents=True);_write(output/'execution.json',dict(terminal=False,completed=False,role=role,recipe_sha256=recipe_sha256,private_values_opened=False))
    reports=[]
    try:
        for rec in records:
            # ALL role results/markers passed before ANYparticle/weight/privateθ read.
            with np.load(_checked(rec['empirical']),allow_pickle=False) as data:
                theta=data['physical_coordinates'];geometry=data['geometry_ids']
            with np.load(_checked(rec['states']),allow_pickle=False) as data:
                weights={name:data[name+'__weights'] for name in STATE_NAMES}
            spec=_read(_checked(rec['object']['realized_hypothesis']))
            if spec['boundary_kind']!='point_pin' or len(spec['fixture_nodes'])!=1:raise ValueError('Declaredpoint prior truth required')
            truth=[spec['material'][k] for k in MATERIAL_KEYS];node=int(spec['fixture_nodes'][0])
            diagnostic=summarize_postview_material(theta,geometry,weights,truth,node,catalogue)
            reports.append(dict(object_index=rec['object']['index'],role=role,selected_active_view=rec['selected_active_view'],
                selected_random_view=rec['selected_random_view'],diagnostic=diagnostic,
                source_artifacts={k:rec[k] for k in ['result','posterior_freeze','states','choice','empirical']},private_hypothesis=rec['object']['realized_hypothesis']))
        best=freeze['best_fixed_view_index'] if freeze is not None else None
        _write(output/'object-diagnostics-LOCAL.json',reports)
        summary=dict(completed=True,role=role,objects=len(reports),recipe_sha256=recipe_sha256,execution_protocol_sha256=digest,
            calibration_freeze=calibration_record,reserved_release=release_record,original_stage_terminal=status_record,
            same2048_particle_comparison=True,C28_full_reference_intervals_unchanged=True,
            prior_geometry_ids_including_zeros=catalogue.tolist(),quantile_convention=reports[0]['diagnostic']['quantile_convention'],
            policies=summarize_policy_comparisons(reports,best_fixed=best),
            source_sha256=recipe['source_sha256'],claim='Descriptive empirical interval/support confidence comparison only; no Bayesiancoverage guarantee or tuning.',
            new_models_inference_queries_resampling_or_sensor_changes=False,raw_object_diagnostics_LOCAL=True)
        _write(output/'summary.json',summary);_write(output/'execution.json',dict(terminal=True,completed=True,role=role,objects=len(reports),recipe_sha256=recipe_sha256,private_values_opened=True))
        return _record(output/'summary.json')
    except BaseException as error:
        _write(output/'failure.json',dict(completed=False,error=type(error).__name__,message=str(error),completed_objects=len(reports)))
        _write(output/'execution.json',dict(terminal=True,completed=False,role=role,recipe_sha256=recipe_sha256,private_values_opened=True));raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for flag in ['recipe','recipe-sha256','execution','role']:p.add_argument('--'+flag,required=True)
    for flag in ['release','release-sha256','calibration','calibration-sha256']:p.add_argument('--'+flag)
    a=p.parse_args()
    def optional(path,digest):
        if (path is None)!=(digest is None):raise ValueError('Artifact path ANDhash required together')
        return None if path is None else dict(path=path,sha256=digest)
    result=run_offline_audit(a.recipe,a.recipe_sha256,a.execution,a.role,
        release_record=optional(a.release,a.release_sha256),calibration_record=optional(a.calibration,a.calibration_sha256))
    print(json.dumps(result))
