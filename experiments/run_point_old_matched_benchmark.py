"""Prospective auxiliary OLD-model predictions on the NEW reserved32 cohort.

No primary pipeline edits or old-policy optimization. Predictions read only
declared acquired frames after primary reserved release. ALL32 baseline state
families freeze before future targets or primary result values are opened.
Comparison then uses the primary's frozen choices for matched-view errors.
This benchmark does not calibrate old-model uncertainty.
"""
from __future__ import annotations
import argparse,json,os,time
from dataclasses import replace
from pathlib import Path
import numpy as np
try:
    import run_point_prior_analysis as legacy
    import run_point_rotation_analysis as primary
    from campaign_point_prior_posterior import load_reference_libraries,condition_reference,METHOD_PATH as OLD_METHOD,METHOD_SHA256 as OLD_METHOD_SHA
    from campaign_point_active_views import update_after_view
    from campaign_support_basis_cache import load_support_basis_cache
    from campaign_sparse_depth_queries import make_queries,predict_ranges
    from campaign_exciting_commands import PROGRAM_IDS,make_exciting_commands
    from campaign_hypothesis_inputs import plain
    from censored_range_observation import CensoredRangeModel,CensoredRanges
except ImportError:
    from experiments import run_point_prior_analysis as legacy
    from experiments import run_point_rotation_analysis as primary
    from experiments.campaign_point_prior_posterior import load_reference_libraries,condition_reference,METHOD_PATH as OLD_METHOD,METHOD_SHA256 as OLD_METHOD_SHA
    from experiments.campaign_point_active_views import update_after_view
    from experiments.campaign_support_basis_cache import load_support_basis_cache
    from experiments.campaign_sparse_depth_queries import make_queries,predict_ranges
    from experiments.campaign_exciting_commands import PROGRAM_IDS,make_exciting_commands
    from experiments.campaign_hypothesis_inputs import plain
    from experiments.censored_range_observation import CensoredRangeModel,CensoredRanges

ROOT=legacy.ROOT
read,checked,record,resolve,sha=legacy.read,legacy.checked,legacy.record,legacy.resolve,legacy.sha
PRIMARY=dict(path='runs/operator-dev/point-rotation-analysis-v02/execution-protocol.json',sha256='0bac66d3701bf3661791859ea2ca6f6009a093c9fb2fd39ca5693b4713dd7fb3')
PRIMARY_METHOD_SHA='7c3a4a9fa89398fdf8c77d0208c6bf932efbaa31751128c95d0ce1f7d5d81ea5'
SCHEMA='old-model-new32-matched-benchmark-v0.1'
STATE_NAMES=legacy.STATE_NAMES


def write_new(path,value):
    with Path(path).open('x') as stream:json.dump(plain(value),stream,indent=2,allow_nan=False);stream.write('\n')


def source_map(execution):
    path='experiments/run_point_old_matched_benchmark.py'
    return {**execution['source_sha256'],path:sha(ROOT/path)}


def source_check(protocol):
    for path,digest in protocol['source_sha256'].items():checked(dict(path=path,sha256=digest))


def sensor(n):return CensoredRangeModel(np.eye(n)*.0002**2,.005,.05,0.)
def matched_libraries(libraries):
    if (libraries.predictive_particles!=2048 or libraries.counts_per_mask!=(256,1024)
            or [r.seed for r in libraries.references]!=[20261112,20261113,20261114]):raise ValueError('Matched2048budget/all3oldpriorlibraries/nestedcounts required')
    return replace(libraries,resampling_seed=2026120700)
def candidates(rest,faces):return make_queries(rest,faces,frame_indices=(24,),offsets_deg=((-60.,10.),(-20.,10.),(20.,10.),(60.,10.)),grid_size=6)
def query_hash(queries):
    import hashlib
    return hashlib.sha256(json.dumps(plain(queries),sort_keys=True,separators=(',',':')).encode()).hexdigest()


def prepare(destination,output):
    folder=resolve(destination)
    if folder.exists() or resolve(output).exists():raise ValueError('Fresh auxiliary plan/output required')
    execution,method,plan=primary.validate_execution(PRIMARY['path'],PRIMARY['sha256'],check_environment=False)
    if execution['method']['sha256']!=PRIMARY_METHOD_SHA:raise ValueError('Exact frozen newprimary method required')
    old_method=read(checked(dict(path=OLD_METHOD,sha256=OLD_METHOD_SHA)))
    inference=old_method['base_inference'];public=read(checked(plan['public_hardware']))
    queries=read(checked(plan['public_acquired_queries']))
    if [q['frame_index'] for q in queries]!=[10,20] or sum(len(q['xs']) for q in queries)!=72:raise ValueError('Same72basequeries required')
    protocol=dict(schema=SCHEMA,status='PROSPECTIVE_AUXILIARY_SOURCE_FROZEN_BEFORE_NEW_TEST_RELEASE',
        primary_execution=PRIMARY,primary_method=execution['method'],cohort_plan=execution['cohort_plan'],cache=execution['cache'],
        old_library_method=dict(path=OLD_METHOD,sha256=OLD_METHOD_SHA),old_library_records=inference['library_references'],
        support_prior=dict(path='runs/operator-dev/unknown-point-oracle-inputs-v01/public/point-mask-prior.json',sha256='7c919bdf7718592c3dacef2402ceb94218a8897348d3a2dc90a183b967526bd0'),
        public_hardware=plan['public_hardware'],public_queries=plan['public_acquired_queries'],
        candidate_query_sha256=query_hash(candidates(public['rest_points_m'],public['surface_faces'])),
        object_indices=list(range(16,48)),role='RESERVED_EVALUATION',states=list(STATE_NAMES),
        prior_seeds=[20261112,20261113,20261114],counts_per_mask=[256,1024],geometry_categories=55,
        particles=2048,resampling_seed=2026120700,initial_noise_seed_base=2026120200,view_noise_seed_base=2026120500,
        sensor=dict(range_std_m=.0002,visibility_error_probability=.005,background_std_m=.05,outlier_probability=0.),
        old_archive_members_used=['geometry_ids','physical_coordinates','predicted_visible','predicted_residual_m'],
        old_likelihood_or_posterior_weights_reused=False,forward='Uncorrected frozen PackedBoundaryMaterialForward; no rotation retraction or discrepancy GP.',
        program_manifests={p:make_exciting_commands(p).public_manifest() for p in PROGRAM_IDS},
        source_sha256=source_map(execution),output=output,device='cuda:2',CPU_threads=2,particle_chunk=512,
        prediction_gate='Exact newprimaryreservedrelease before anyactualsensorframe; all32oldstatesfreeze before futureA/B or primaryresult values. No physicalparameter labels are needed.',
        final_comparison='After BOTHcomplete sealed32, compare everyold/newpotentialstate and primary-selected SAMEview onA31..60/B0..60; primarychoice mustfreeze beforeactualview. Nooldactionoptimization.',
        uncertainty_calibration_claim=False,primary_view_policy_test_redefined=False,
        failure_policy='Stop andpreserve anyfailed/incompleteobject; no retry/redraw/filtering or selectedsubsetaggregate.',
        bootstrap_replicates=10000,bootstrap_seed=2026120600,raw_artifacts_LOCAL=True)
    for item in protocol['old_library_records']:checked(item)
    source_check(protocol);folder.mkdir(parents=True);path=folder/'protocol.json';write_new(path,protocol);return record(path)


def validate_protocol(protocol_record):
    p=read(checked(protocol_record))
    if (p.get('schema')!=SCHEMA or p.get('primary_execution')!=PRIMARY or p.get('primary_method',{}).get('sha256')!=PRIMARY_METHOD_SHA
            or p.get('object_indices')!=list(range(16,48)) or p.get('role')!='RESERVED_EVALUATION'
            or p.get('states')!=list(STATE_NAMES) or p.get('particles')!=2048 or p.get('resampling_seed')!=2026120700
            or p.get('prior_seeds')!=[20261112,20261113,20261114] or p.get('counts_per_mask')!=[256,1024]
            or p.get('sensor')!=dict(range_std_m=.0002,visibility_error_probability=.005,background_std_m=.05,outlier_probability=0.)
            or p.get('device')!='cuda:2' or p.get('CPU_threads')!=2 or p.get('particle_chunk')!=512):raise ValueError('Exact frozen auxiliary matched benchmark required')
    execution,method,plan=primary.validate_execution(PRIMARY['path'],PRIMARY['sha256'],check_environment=False)
    if p['cohort_plan']!=execution['cohort_plan'] or p['source_sha256']!=source_map(execution):raise ValueError('Source/primary/input closure changed')
    source_check(p)
    for key in ['public_hardware','public_queries','old_library_method','support_prior']:checked(p[key])
    if p['old_library_method']!=dict(path=OLD_METHOD,sha256=OLD_METHOD_SHA):raise ValueError('Original C26 libraries required')
    old=read(checked(p['old_library_method']))
    if p['old_library_records']!=old['base_inference']['library_references']:raise ValueError('Original all3 seed libraries required')
    return p,execution,method,plan


def authorize_reserved(p,plan,release_record):
    release=read(checked(release_record));freeze_record=release.get('calibration_freeze')
    if release.get('execution_protocol_sha256')!=PRIMARY['sha256'] or release.get('method_sha256')!=PRIMARY_METHOD_SHA:raise ValueError('Newprimary explicit reservedrelease required')
    freeze=read(checked(freeze_record))
    for r in freeze['calibration_results']:checked(r) # Hashes, never calibration result values.
    return primary.eligible_objects(plan,'RESERVED_EVALUATION',execution_sha256=PRIMARY['sha256'],method_sha256=PRIMARY_METHOD_SHA,
        release=release,calibration_freeze=freeze,calibration_freeze_record=freeze_record)


def _save_observation(path,observation):
    np.savez_compressed(path,visible=observation.visible,residual_m=observation.residual_m)


def predict_object(obj,pair,context,p,protocol_record,release_record,destination):
    import torch
    packed,public,libraries,queries,views=context;rest=np.asarray(public['rest_points_m']);faces=np.asarray(public['surface_faces'])
    index=obj['index'];device=p['device'];destination.mkdir(parents=True,exist_ok=False)
    observation=legacy._actual_observation(pair['A']['trajectory'],[10,20],rest,faces,queries,sensor(72),2026120200+index,device)
    _save_observation(destination/'base-observation.npz',observation)
    posterior=condition_reference(libraries,observation);empirical=posterior['empirical_posterior']
    write_new(destination/'reference-diagnostics.json',dict(summaries=posterior['summaries'],stability=posterior['stability'],old_LL_or_weights_reused=False))
    np.savez_compressed(destination/'empirical-posterior.npz',**{k:empirical[k] for k in ['geometry_ids','physical_coordinates','reference_indices','weights']})
    displacements={}
    for label,program in zip('AB',PROGRAM_IDS):
        q=np.empty((2048,61,len(rest),3),dtype=np.float64)
        if make_exciting_commands(program).public_manifest()!=p['program_manifests'][program]:raise ValueError('Known controls changed')
        for first in range(0,2048,512):
            with torch.no_grad():values=packed.predict(make_exciting_commands(program),torch.tensor(empirical['geometry_ids'][first:first+512].copy()),
                torch.tensor(empirical['physical_coordinates'][first:first+512].copy()),device=device)['positions_m']
            q[first:first+512]=values.cpu().numpy()-rest
        displacements[label]=q
    # All4 fixed counterfactuals are declared; there is deliberately NO old selector.
    predicted=predict_ranges(displacements['A']+rest,faces,views,device=device,particle_chunk=64)
    weights=[np.full(2048,1/2048)];diagnostics=[]
    for j,query in enumerate(views):
        actual=legacy._actual_observation(pair['A']['trajectory'],[24],rest,faces,[query],sensor(36),2026120500+4*index+j,device)
        _save_observation(destination/f'view{j}-observation.npz',actual)
        predicted_view=CensoredRanges(predicted.visible[:,36*j:36*(j+1)],predicted.residual_m[:,36*j:36*(j+1)])
        w,diagnostic=update_after_view(predicted_view,actual,sensor(36));weights.append(w);diagnostics.append(diagnostic)
    payload=dict(rest_points_m=rest,times_s=np.arange(61)/20)
    for name,w in zip(STATE_NAMES,weights):
        for label in 'AB':
            mean,std=legacy.weighted_displacement_moments(displacements[label],w)
            payload[name+'__'+label+'__mean_displacement_m']=mean;payload[name+'__'+label+'__latent_std_m']=std
        payload[name+'__weights']=w
    path=destination/'predictive-states.npz';np.savez_compressed(path,**payload);source_check(p)
    freeze=dict(schema='old-model-new32-object-states-v0.1',completed=True,object_index=index,object_id=obj['object_id'],role='RESERVED_EVALUATION',
        protocol=protocol_record,release=release_record,states=record(path),state_names=list(STATE_NAMES),capture_pairs=pair,
        source_sha256=p['source_sha256'],future_A_or_B_truth_opened=False,material_or_support_truth_opened=False,
        primary_result_values_opened=False,old_action_selection_performed=False,view_update_diagnostics=diagnostics,
        reference_diagnostics=record(destination/'reference-diagnostics.json'))
    path=destination/'states-frozen.json';write_new(path,freeze);return record(path)


def predict(protocol_record,release_record):
    p,execution,method,plan=validate_protocol(protocol_record);objects=authorize_reserved(p,plan,release_record)
    # Release precedes preflight and ALL sensor/finite-array access.
    pairs=legacy.capture_pairs(plan,objects)
    if any(os.environ.get(k)!='2' for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']):raise ValueError('CPU2 environment required')
    import torch
    torch.set_num_threads(2);output=resolve(p['output']);output.mkdir(parents=True,exist_ok=False)
    status=dict(pid=os.getpid(),terminal=False,completed=False,protocol=protocol_record,release=release_record,
        planned_indices=list(range(16,48)),completed_indices=[],queued_indices=list(range(16,48)),failures=[])
    write_new(output/'execution.json',status);started=time.monotonic();records=[]
    try:
        packed,public,ids,_=load_support_basis_cache(p['cache']['path'],p['cache']['sha256'])
        if len(ids)!=55 or public!=read(checked(p['public_hardware'])):raise ValueError('Same publicH0/all55 categories required')
        libraries=matched_libraries(load_reference_libraries(p['old_library_method']['path'],p['old_library_method']['sha256']))
        if libraries.predictive_particles!=2048 or [r.seed for r in libraries.references]!=p['prior_seeds']:raise ValueError('Matched budget/all3 oldpriorlibraries required')
        queries=read(checked(p['public_queries']));views=candidates(public['rest_points_m'],public['surface_faces'])
        if query_hash(views)!=p['candidate_query_sha256']:raise ValueError('Fixed virtualcamera queries changed')
        context=(packed,public,libraries,queries,views)
        for obj in objects:
            source_check(p);status['active_index']=obj['index'];status['queued_indices'].remove(obj['index']);legacy.write(output/'execution.json',status)
            records.append(predict_object(obj,pairs[obj['index']],context,p,protocol_record,release_record,output/'objects'/obj['object_id']))
            status['completed_indices'].append(obj['index']);legacy.write(output/'execution.json',status)
            print(json.dumps(dict(completed_old_states=obj['index'])),flush=True)
        global_freeze=dict(schema='old-model-new32-all-states-frozen-v0.1',completed=True,protocol=protocol_record,release=release_record,
            object_indices=list(range(16,48)),object_states=records,source_sha256=p['source_sha256'],
            future_A_or_B_truth_opened=False,material_or_support_truth_opened=False,primary_result_values_opened=False)
        path=output/'all-states-frozen.json';write_new(path,global_freeze);status.update(terminal=True,completed=True,all_states_freeze=record(path),elapsed_s=time.monotonic()-started)
    except BaseException as error:
        status['failures'].append(dict(object_index=status.get('active_index'),error=repr(error)));status.update(terminal=True,completed=False,elapsed_s=time.monotonic()-started);legacy.write(output/'execution.json',status);raise
    legacy.write(output/'execution.json',status);return status


def verify_all_states(protocol_record,freeze_record):
    p,_,_,_=validate_protocol(protocol_record);freeze=read(checked(freeze_record))
    if (freeze.get('schema')!='old-model-new32-all-states-frozen-v0.1' or freeze.get('completed') is not True
            or freeze.get('protocol')!=protocol_record or freeze.get('object_indices')!=list(range(16,48))
            or len(freeze.get('object_states',[]))!=32 or freeze.get('source_sha256')!=p['source_sha256']
            or any(freeze.get(k) is not False for k in ['future_A_or_B_truth_opened','material_or_support_truth_opened','primary_result_values_opened'])):raise ValueError('All32 baseline states mustfreeze before paired truth/comparison')
    values=[]
    for i,r in zip(range(16,48),freeze['object_states']):
        v=read(checked(r))
        if (v.get('object_index')!=i or v.get('role')!='RESERVED_EVALUATION' or v.get('completed') is not True
                or v.get('protocol')!=protocol_record or v.get('release')!=freeze['release']
                or v.get('state_names')!=list(STATE_NAMES) or v.get('source_sha256')!=p['source_sha256']
                or any(v.get(k) is not False for k in ['future_A_or_B_truth_opened','material_or_support_truth_opened','primary_result_values_opened'])):raise ValueError('Complete exact ordered baseline states required')
        checked(v['states']);values.append(v)
    return p,freeze,values


def compare(protocol_record,freeze_record,primary_evaluation_record):
    # This order blocks ALL primary result values/targets until baseline32freeze.
    p,freeze,old_values=verify_all_states(protocol_record,freeze_record)
    evaluation=read(checked(primary_evaluation_record))
    if (evaluation.get('schema')!='point-rotation-reserved-view-evaluation-v0.2' or evaluation.get('completed') is not True
            or evaluation.get('objects')!=32 or evaluation.get('release')!=freeze['release']
            or evaluation.get('execution_protocol_sha256')!=PRIMARY['sha256'] or evaluation.get('method_sha256')!=PRIMARY_METHOD_SHA
            or len(evaluation.get('reserved_results',[]))!=32):raise ValueError('BOTH complete primary/auxiliary32 seals required')
    paired=[]
    for i,old_value,new_record in zip(range(16,48),old_values,evaluation['reserved_results']):
        new=read(checked(new_record))
        if (new.get('object_index')!=i or new.get('role')!='RESERVED_EVALUATION' or new.get('completed') is not True
                or new.get('posterior_frozen_before_future_truth') is not True or new['capture_pairs']!=old_value['capture_pairs']):raise ValueError('Exact same complete pairedobject/captures required')
        marker=read(checked(new['posterior_freeze']));choice=read(checked(marker['active_choice']))
        if (marker.get('future_A_or_B_truth_opened') is not False or marker.get('material_or_support_truth_opened') is not False
                or marker.get('execution_protocol_sha256')!=PRIMARY['sha256']
                or choice.get('actual_frame24_opened') is not False or choice.get('selected_index')!=new['selected_active_view']
                or choice.get('execution_protocol_sha256')!=PRIMARY['sha256']):raise ValueError('Primary SAMEviewchoice mustprecede actualview/targets')
        with np.load(checked(old_value['states']),allow_pickle=False) as z:
            rest=z['rest_points_m'];old_metrics={}
            # First futuretarget access in this entire auxiliarypipeline is here.
            truth={label:legacy._load_positions_subset(old_value['capture_pairs'][label]['trajectory'],list(range(61)),rest)[0]-rest for label in 'AB'}
            for name in STATE_NAMES:
                ma=z[name+'__A__mean_displacement_m'];mb=z[name+'__B__mean_displacement_m']
                old_metrics[name]=dict(A_after1p5s_MSE_m2=float(np.mean((ma[31:]-truth['A'][31:])**2)),B_all_MSE_m2=float(np.mean((mb-truth['B'])**2)))
        paired.append(dict(object_index=i,old=old_metrics,new={n:{k:new['states'][n][k] for k in ['A_after1p5s_MSE_m2','B_all_MSE_m2']} for n in STATE_NAMES},
            primary_active_view_index=new['selected_active_view'],primary_random_view_index=new['selected_random_view']))
    try:from summarize_point_rotation_analysis import paired_bootstrap
    except ImportError:from experiments.summarize_point_rotation_analysis import paired_bootstrap
    summaries={};best=evaluation['best_fixed_view_index_frozen_on_calibration']
    for policy in [*STATE_NAMES,'primary_active_same_view','primary_random_same_view','primary_bestfixed_same_view']:
        metrics={}
        for key in ['A_after1p5s_MSE_m2','B_all_MSE_m2']:
            def name(row):
                if policy=='primary_active_same_view':return f"fixed_view_{row['primary_active_view_index']}"
                if policy=='primary_random_same_view':return f"fixed_view_{row['primary_random_view_index']}"
                if policy=='primary_bestfixed_same_view':return f'fixed_view_{best}'
                return policy
            old=np.array([r['old'][name(r)][key] for r in paired]);new=np.array([r['new'][name(r)][key] for r in paired])
            metrics[key]=dict(old_pooled_coordinate_RMSE_mm=float(1000*np.sqrt(old.mean())),new_pooled_coordinate_RMSE_mm=float(1000*np.sqrt(new.mean())),
                new_minus_old_MSE_m2=paired_bootstrap(new-old),new_minus_old_object_RMSE_mm=paired_bootstrap(1000*(np.sqrt(new)-np.sqrt(old))))
        summaries[policy]=metrics
    report=dict(schema='old-new-matched-32-auxiliary-comparison-v0.1',completed=True,objects=32,protocol=protocol_record,
        auxiliary_all_states=freeze_record,primary_evaluation=primary_evaluation_record,results=summaries,paired_objects=paired,
        no_primary_results_before_auxiliary_all_states=True,old_uncertainty_calibration_claim=False,
        interpretation='Prospective auxiliary combined-model benchmark, not a revised primaryviewpolicy test or a rotation-vs-GP ablation. Allsame-viewcases retained. Negative paired differences favornew.',
        source_sha256=p['source_sha256'])
    path=resolve(p['output'])/'matched-comparison.json';write_new(path,report);return record(path)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='action',required=True)
    q=sub.add_parser('prepare');q.add_argument('--destination',required=True);q.add_argument('--output',required=True)
    for action in ['predict','compare']:
        q=sub.add_parser(action);q.add_argument('--protocol',required=True);q.add_argument('--protocol-sha256',required=True)
        if action=='predict':q.add_argument('--release',required=True);q.add_argument('--release-sha256',required=True)
        else:q.add_argument('--all-states',required=True);q.add_argument('--all-states-sha256',required=True);q.add_argument('--primary-evaluation',required=True);q.add_argument('--primary-evaluation-sha256',required=True)
    a=parser.parse_args()
    if a.action=='prepare':result=prepare(a.destination,a.output)
    elif a.action=='predict':result=predict(dict(path=a.protocol,sha256=a.protocol_sha256),dict(path=a.release,sha256=a.release_sha256))
    else:result=compare(dict(path=a.protocol,sha256=a.protocol_sha256),dict(path=a.all_states,sha256=a.all_states_sha256),dict(path=a.primary_evaluation,sha256=a.primary_evaluation_sha256))
    print(json.dumps(result))
