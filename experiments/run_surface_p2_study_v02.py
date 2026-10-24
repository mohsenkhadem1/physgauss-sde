"""Crash-resumable source-only prospective static study runner.

Actual execution requires hash-bound full-catalog admission, frozen protocol,
and an independent cost/review release. No automatic execution at import.
All raw parameters, observations, posterior arrays and choices remain LOCAL.
"""
from __future__ import annotations
import argparse,ast,hashlib,json,os,resource,shutil,signal,subprocess,sys,time
from dataclasses import asdict
from pathlib import Path
import numpy as np
from experiments import campaign_surface_p2_study_v02 as study
from experiments.campaign_surface_p2_inference_v01 import LabelledModal,IntegrationPlan,infer,physical_parameters,qualify_integrals
from experiments.campaign_surface_response_cache_v01 import build_cache,CachedLabelledModal,model_fingerprint
from experiments.campaign_surface_conditional_amplitude_v01 import ConditionalAmplitudeReference,AmplitudePlan,RatioModal,ExactRatioMemo

ROOT=Path(__file__).resolve().parents[1]
LOCAL=ROOT/'runs/operator-dev/surface-p2-study-v02'
RESOURCES=dict(worker_CPU_sets=[[8,9],[10,11],[12,13],[14,15]],controller_CPU=8,
    threads_per_worker=1,workers=4,GPU=False,worker_RSS_cap_bytes=4*1024**3,
    aggregate_RSS_cap_bytes=16*1024**3,worker_wall_cap_s=7200,global_wall_cap_s=7200,
    disk_reserve_bytes=30*1024**3,maximum_writes_bytes=12*1024**3)
LAWS=('matching_linear','NH80_stress')
GPU_UUIDS=('GPU-5b432ce5-a24a-68ba-150f-682265260a9b','GPU-c49707f6-24fd-38e4-3973-3543ec644ceb','GPU-571378aa-c44e-ffb8-b346-1e409f9821f9')


def resources_for(device_mode='cpu'):
    if device_mode not in ('cpu','gpu'):raise ValueError('Explicit cpu or gpu device mode')
    base=dict(RESOURCES)
    if device_mode=='gpu':base.update(worker_CPU_sets=[[i]for i in range(16)],controller_CPU=16,workers=16,
        aggregate_RSS_cap_bytes=64*1024**3,host_preflight_available_bytes=94*1024**3,RAM_reserve_bytes=30*1024**3,worker_wall_cap_s=9000,global_wall_cap_s=10800)
    return dict(base,device_mode=device_mode,GPU=device_mode=='gpu',
        allowed_GPU_UUIDs=list(GPU_UUIDS)if device_mode=='gpu'else[],
        worker_GPU_UUIDs=[GPU_UUIDS[i%3]for i in range(base['workers'])]if device_mode=='gpu'else[],
        allocated_GPU_cap_bytes_per_worker=512*1024**2 if device_mode=='gpu'else 0,
        study_VRAM_cap_bytes_per_physical_GPU=8*1024**3 if device_mode=='gpu'else 0,
        GPU_maximum_batch=4096 if device_mode=='gpu'else 0)



def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb')as f:
        for b in iter(lambda:f.read(1024**2),b''):h.update(b)
    return h.hexdigest()


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');os.replace(tmp,path)


def array_sha(a):return hashlib.sha256(np.ascontiguousarray(a,dtype='<f8').tobytes()).hexdigest()


def sources():
    pending=['experiments/run_surface_p2_study_v02.py','tests/test_surface_p2_study_v02.py',
        'tests/test_surface_p2_inference_v01.py','tests/test_surface_response_cache_v01.py','tests/test_surface_conditional_amplitude_v01.py','tests/test_amplitude_adaptive_oracle_v01.py'];seen=set()
    while pending:
        name=pending.pop()
        if name in seen:continue
        path=ROOT/name
        if not path.is_file():raise ValueError('Missing source '+name)
        seen.add(name)
        for node in ast.walk(ast.parse(path.read_text())):
            modules=[]
            if isinstance(node,ast.Import):modules=[x.name for x in node.names]
            elif isinstance(node,ast.ImportFrom)and node.level==0:
                modules=[node.module or '']+[(node.module+'.'+x.name)for x in node.names if node.module]
            for mod in modules:
                rel=mod.replace('.','/')+'.py'
                if (ROOT/rel).is_file()and rel not in seen:pending.append(rel)
    return {name:sha(ROOT/name)for name in sorted(seen)}


def catalog_gate(path,digest,*,force_hashes=None):
    path=Path(path)
    if sha(path)!=digest:raise ValueError('Catalog admission SHA mismatch')
    a=json.loads(path.read_text())
    keys=('qualified','all50_cases_retained','all50_load_cases_qualified')
    if not all(a.get(k)is True for k in keys)or a.get('target_mesh_n')!=4:
        raise ValueError('Full50 retained catalog cases must qualify for n4')
    for key in ('source_SHA256','protocol_SHA256','force_SHA256'):
        if not a.get(key):raise ValueError('Catalog provenance missing '+key)
    for name,d in a['source_SHA256'].items():
        p=Path(name)if Path(name).is_absolute()else ROOT/name
        if sha(p)!=d:raise ValueError('Catalog source changed '+name)
    actual={name:array_sha(f)for name,f in study.load_catalog(4).items()}if force_hashes is None else force_hashes
    if a['force_SHA256']!=actual:raise ValueError('Exact v02 ten-load catalog mismatch')
    return a


def environment_guard(worker=None,*,device_mode='cpu'):
    limits=resources_for(device_mode)
    allowed={limits['controller_CPU']}if worker is None else set(limits['worker_CPU_sets'][worker])
    expected_device=''if device_mode=='cpu'else GPU_UUIDS[0 if worker is None else worker%3]
    if os.environ.get('CUDA_VISIBLE_DEVICES')!=expected_device or(device_mode=='gpu'and os.environ.get('CUBLAS_WORKSPACE_CONFIG')!=':4096:8')or any(os.environ.get(k)!='1'for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'))or not set(os.sched_getaffinity(0))<=allowed:
        raise ValueError('Frozen CPU/one-thread/noGPU guard')
    if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024>RESOURCES['worker_RSS_cap_bytes']:raise MemoryError('RSS cap exceeded')
    if shutil.disk_usage(ROOT).free<RESOURCES['disk_reserve_bytes']:raise OSError('30GiB disk reserve failed')
    if LOCAL.exists()and sum(p.stat().st_size for p in LOCAL.rglob('*')if p.is_file())>RESOURCES['maximum_writes_bytes']:raise OSError('Frozen local-write budget exceeded')


def available_memory():
    return int(next(l for l in Path('/proc/meminfo').read_text().splitlines()if l.startswith('MemAvailable:')).split()[1])*1024


def host_preflight(device_mode):
    limits=resources_for(device_mode)
    if device_mode=='gpu':
        available=available_memory()
        if available<limits['host_preflight_available_bytes']:raise MemoryError('16-worker preflight requires94GiB available, including30GiB reserve')


def prepare(admission_path,admission_digest,*,device_mode='cpu',numerical_engine='tensor'):
    if numerical_engine not in('tensor','conditional_amplitude'):raise ValueError('Explicit tensor baseline or conditional_amplitude')
    environment_guard(device_mode=device_mode);host_preflight(device_mode);admission=catalog_gate(admission_path,admission_digest)
    if(LOCAL/'protocol.json').exists():raise ValueError('No refreeze')
    plan=IntegrationPlan(chunk=4096 if device_mode=='gpu'else 1024)
    protocol=dict(schema='surface-p2-study-v02',status='FROZEN_BEFORE_ACTUAL_STUDY',design=study.design(),
        catalog_admission=dict(path=str(Path(admission_path).resolve()),SHA256=admission_digest),
        force_SHA256=admission['force_SHA256'],source_SHA256=sources(),resources=resources_for(device_mode),device_mode=device_mode,
        integration_plan=dict(simpson=list(plan.simpson),gauss=list(plan.gauss),chunk=plan.chunk),
        numerical_engine=numerical_engine,
        selected_numerical_plan=dict(engine=numerical_engine,**(asdict(plan)if numerical_engine=='tensor'else dict(**asdict(AmplitudePlan()),adaptive_epsrel=[1e-6,1e-8],adaptive_error_estimate_limit=1e-5))),
        amplitude_plan=asdict(AmplitudePlan()),ratio_memo_bytes=256*1024**2,
        ratio_evaluator='CPU allmode float64 for<16rows; optional pinnedGPU float64 forbatched logratio; exact-byte bounded memo',
        resource_projection='Measured3independentposteriors ononeGPU:89.50s includingbuild,67.10–67.96s integration; conservative~1.7h608posteriorprojection across3GPUs;6vs3contexts unmeasured, worker9000s/global10800s are caps not proven runtime bounds',
        expected_objects=16,expected_prefix_posteriors=32,expected_joint_posteriors=576,expected_total_posteriors=608,
        cache='Tensor baseline: exact-node LOCAL memmaps. Conditional-amplitude: bounded exact-byte ratio memo only, no giant tensor cache.',
        continuation='Checkpoint completed numerical posteriors and sealed choices exactly; do not rerun unresolved or failed completed posteriors; no object redraw/filter; tensor baseline source preserved, a later execution requires a separate namespace and new protocol',
        source_only_until_reviewed_cost_release=True,retries=0)
    write(LOCAL/'protocol.json',protocol)
    return dict(protocol=str(LOCAL/'protocol.json'),SHA256=sha(LOCAL/'protocol.json'))


def validate(digest):
    if sha(LOCAL/'protocol.json')!=digest:raise ValueError('Frozen protocol SHA mismatch')
    p=json.loads((LOCAL/'protocol.json').read_text())
    if p['source_SHA256']!=sources()or p['resources']!=resources_for(p['device_mode']):raise ValueError('Source/resource changed')
    admission=p['catalog_admission'];catalog_gate(admission['path'],admission['SHA256'])
    return p


def release_gate(digest):
    """Root review/cost receipt is explicit; metadata alone never launches."""
    release=json.loads((LOCAL/'reviewed-release.json').read_text())
    if release.get('approved')is not True or release.get('protocol_SHA256')!=digest or release.get('source_SHA256')!=sources()or release.get('resources')!=resources_for(json.loads((LOCAL/'protocol.json').read_text())['device_mode']):
        raise ValueError('Root-reviewed frozen source/resource release required')
    cost=release.get('cost_control',{})
    if not cost.get('path')or not cost.get('SHA256')or sha(cost['path'])!=cost['SHA256']:
        raise ValueError('Hash-bound actual tiny cost-control receipt required')
    cost_result=json.loads(Path(cost['path']).read_text())
    closure=release.get('cost_ownership',{})
    if cost_result.get('schema')!='surface-amplitude-cost-v01' or not cost_result.get('completed') or not cost_result.get('all_numerical_controls_qualified'):raise ValueError('Completed qualified synthetic cost required')
    if not closure.get('path') or sha(closure['path'])!=closure.get('SHA256'):raise ValueError('Hash-bound cost ownership required')
    closed=json.loads(Path(closure['path']).read_text())
    if not(closed.get('completed') and closed.get('all_reaped') and closed.get('all_worker_exit_codes_zero')) or closed.get('controller_exit_code')!=0 or closed.get('result_SHA256')!=cost['SHA256'] or closed.get('recipe_SHA256')!=cost_result.get('recipe_SHA256'):raise ValueError('Successful reaped cost ownership bound to result required')
    if json.loads((LOCAL/'protocol.json').read_text())['device_mode']=='gpu':
        concurrent=release.get('concurrency_control',{})
        if not concurrent.get('path')or not concurrent.get('SHA256')or sha(concurrent['path'])!=concurrent['SHA256']:raise ValueError('Hash-bound shared-device concurrency control required')
        outcome=json.loads(Path(concurrent['path']).read_text())
        if not outcome.get('completed')or not outcome.get('all_qualified')or not outcome.get('all_reaped'):raise ValueError('Concurrency control must complete/qualify/reap all3')
    return release


def make_model(*,device_mode='cpu',slot=None):
    from experiments import campaign_surface_p2_static_v02 as physics
    fwd=physics.SparseStaticForward(4)
    model=LabelledModal.from_forward(fwd,loads=study.load_catalog(4),points=study.observation_points(),
        condensed=True,query_matrix=physics.point_query_matrix)
    if device_mode=='gpu':
        from experiments.campaign_surface_gpu_modal_v01 import Float64GPUModal
        model=Float64GPUModal(model.modal,model.labels,physical_UUID=GPU_UUIDS[0 if slot is None else slot%3],
            allowed_UUIDs=GPU_UUIDS,maximum_batch=4096,allocated_cap_bytes=512*1024**2)
    return fwd,model


def plan_from(p):
    x=p['integration_plan'];return IntegrationPlan(tuple(x['simpson']),tuple(x['gauss']),x['chunk'])


def ratio_callback(model):
    """One model-bound memo per worker, with exact CPU scalar evaluation."""
    CPU=RatioModal(model.modal,model.labels)
    def exact(z,labels):
        return model.ratio_select(labels)(z)if hasattr(model,'ratio_select')and len(z)>=16 else CPU(np.exp(z),labels)
    return ExactRatioMemo(exact,model_fingerprint=model_fingerprint(model),maximum_bytes=256*1024**2)


def create_shared_cache(digest):
    """Explicit separately released stage, before any object observations."""
    p=validate(digest);environment_guard(device_mode=p['device_mode']);release_gate(digest);fwd,model=make_model(device_mode=p['device_mode']);plan=plan_from(p)
    if p['numerical_engine']=='tensor':
        receipt=LOCAL/'cache-admission-LOCAL.json'
        write(receipt,dict(cache_admission=True,scope='qualified_physics_catalog',
            model_fingerprint=model_fingerprint(model),protocol_SHA256=digest,catalog_admission=p['catalog_admission']))
        build_cache(model,LOCAL/'cache-LOCAL',admission_receipt=receipt,admission_SHA256=sha(receipt),plan=plan)
        manifest=LOCAL/'cache-LOCAL/manifest.json'
        cached=CachedLabelledModal(model,manifest.parent,manifest_SHA256=sha(manifest))
        prior=qualify_integrals(lambda u:np.zeros(len(u)),cached.select(study.TARGETS),plan=plan)
        manifestSHA=sha(manifest)
    else:
        # A constant dummy zero-response likelihood integrates the original
        # normalized prior, not an acquired or synthetic physical observation.
        callback=ratio_callback(model)
        def prior_ratio(z,labels):
            return np.stack([np.zeros((len(z),24))if label=='__PRIOR_ZERO__'else callback(z,[label])[:,0]for label in labels],axis=1)
        reference=ConditionalAmplitudeReference(None,[('__PRIOR_ZERO__',np.zeros(24))],study.TARGETS,log_ratio_predict=prior_ratio)
        prior=reference.qualify(plan=AmplitudePlan());prior['predictive_mean']=prior['local_final']['predictive_mean']if prior['local_final']is not None else None
        prior['predictive_SD']=prior['local_final']['predictive_SD']if prior['local_final']is not None else None
        manifestSHA=None
    if not prior['qualified']:raise FloatingPointError('Prior predictive quadrature unresolved; no object study')
    np.savez_compressed(LOCAL/'prior-predictive-LOCAL.npz',mean=prior['predictive_mean'],SD=prior['predictive_SD'])
    write(LOCAL/'cache-ready.json',dict(protocol_SHA256=digest,manifest_SHA256=manifestSHA,numerical_engine=p['numerical_engine'],
        prior_SHA256=sha(LOCAL/'prior-predictive-LOCAL.npz'),model_fingerprint=model_fingerprint(model),
        all_modes=model.modal.diagnostics,qualified=True,tensor_cache_created=p['numerical_engine']=='tensor'))
    return dict(reference_ready_SHA256=sha(LOCAL/'cache-ready.json'))


def cached_model(digest,*,slot=None):
    p=validate(digest);r=json.loads((LOCAL/'cache-ready.json').read_text())
    if r['numerical_engine']!=p['numerical_engine']or r['protocol_SHA256']!=digest or not r['qualified']or sha(LOCAL/'prior-predictive-LOCAL.npz')!=r['prior_SHA256']:raise ValueError('Shared cache-ready binding mismatch')
    fwd,model=make_model(device_mode=p['device_mode'],slot=slot)
    if model_fingerprint(model)!=r['model_fingerprint']:raise ValueError('Reconstructed exact model differs')
    cached=CachedLabelledModal(model,LOCAL/'cache-LOCAL',manifest_SHA256=r['manifest_SHA256'])if p['numerical_engine']=='tensor'else model
    return p,fwd,cached


def private_streams(index):
    if index not in range(16):raise ValueError('Original16object index required')
    return dict(unit=np.random.default_rng(study.COHORT_SEED).uniform(size=(16,2))[index],
        prefix_noise=study.SIGMA_M*np.random.default_rng(study.PREFIX_NOISE_SEED).normal(size=(16,24))[index],
        probe_noise=study.SIGMA_M*np.random.default_rng(study.PROBE_NOISE_SEED).normal(size=(16,3,24))[index],
        random_realization=np.random.default_rng(study.RANDOM_ACTION_SEED).integers(3,size=(16,2))[index])


def physical_metric(fwd,u,force,theta,*,label,nonlinear_model=None):
    from experiments.run_clamped_surface_p2_qualification_v02 import deformation
    from experiments.run_clamped_surface_p2_qualification_v01 import reaction_diagnostics,stability_diagnostic
    metric=deformation(fwd,u)
    if nonlinear_model is None:
        eq=reaction_diagnostics(fwd,u,force,*theta)
        reduced=theta[0]*fwd.Kunit_free+theta[1]*fwd.W_free
    else:
        state=nonlinear_model.evaluate(u,force=force,tangent=True)
        eq=reaction_diagnostics(fwd,u,force,*theta,nonlinear_residual=state['residual'])
        reduced=state['tangent'][fwd.free][:,fwd.free]
    stable=stability_diagnostic(reduced,4)
    valid=metric['minimum_J']>=.8 and metric['maximum_absolute_symmetric_strain_eigenvalue']<=.05 and eq['passed']and stable['passed']
    excitation=metric['nonrigid_RMS_m']>=100e-6 and metric['nonrigid_fraction']>=.3
    return dict(valid=bool(valid and(label!=study.PREFIX or excitation)),material_valid=bool(valid),
        prefix_excitation_required=label==study.PREFIX,excitation_passed=bool(excitation),
        nonrigid_RMS_m=metric['nonrigid_RMS_m'],nonrigid_fraction=metric['nonrigid_fraction'],
        minimum_J=metric['minimum_J'],maximum_absolute_symmetric_strain_eigenvalue=metric['maximum_absolute_symmetric_strain_eigenvalue'],equilibrium=eq,stability=stable)


def checkpoint_present(path):
    path=Path(path);metadata=path.with_suffix('.commit.json')
    if path.exists()!=metadata.exists():raise ValueError('Unbound partial checkpoint requires explicit recovery before any solve')
    return path.exists()


def checkpoint_npz(path,context,**arrays):
    path=Path(path);metadata=path.with_suffix('.commit.json')
    if path.exists() or metadata.exists():raise ValueError('Existing checkpoint cannot be overwritten')
    temporary=path.with_suffix('.tmp.npz');np.savez_compressed(temporary,**arrays);os.replace(temporary,path)
    write(metadata,dict(context=context,raw_SHA256=sha(path)))


def verify_npz(path,context):
    path=Path(path);metadata=path.with_suffix('.commit.json')
    if not path.is_file() or not metadata.is_file():raise ValueError('Unbound partial checkpoint requires explicit recovery')
    commit=json.loads(metadata.read_text())
    if commit.get('context')!=context or commit.get('raw_SHA256')!=sha(path):raise ValueError('Partial checkpoint context/checksum mismatch')


def generate_truth_fields(index,fwd,model,directory,protocol_SHA256):
    """Both laws, all ten loads, original draw; completed loads checkpoint once."""
    from experiments import campaign_surface_p2_static_v02 as physics
    stream=private_streams(index);theta=physical_parameters(stream['unit'][None])[0]
    loads=study.load_catalog(4);labels=tuple(loads);force=np.asarray(list(loads.values()))
    linearpath=directory/'linear-fields-LOCAL.npz'
    context=dict(protocol_SHA256=protocol_SHA256,index=index,unit_SHA256=array_sha(stream['unit']),force_SHA256={label:array_sha(f)for label,f in loads.items()})
    if not checkpoint_present(linearpath):
        fields=fwd.staticforward(theta[None],force)[0]
        checkpoint_npz(linearpath,context,fields=fields,unit=stream['unit'])
    verify_npz(linearpath,context)
    with np.load(linearpath)as z:
        if not np.array_equal(z['unit'],stream['unit']):raise ValueError('Original unit draw checkpoint changed')
        linear=z['fields']
    diagnostics={law:{}for law in LAWS};nonlinear=[]
    nh=physics.P2NonlinearReference.from_forward(fwd,*theta,quadrature_order=80)
    for j,(label,f)in enumerate(loads.items()):
        snapshot=directory/f'truth-load-{label}-LOCAL.npz'
        load_context=dict(context,label=label)
        if checkpoint_present(snapshot):
            verify_npz(snapshot,load_context)
            with np.load(snapshot)as z:
                if not np.array_equal(z['unit'],stream['unit'])or str(z['force_SHA256'])!=array_sha(f):raise ValueError('Completed load checkpoint mismatch')
                un=z['NH80_field'];diag=json.loads(str(z['diagnostics_json']))
        else:
            linear_diag=physical_metric(fwd,linear[j],f,theta,label=label)
            un,newton=nh.solve(f)
            nonlinear_diag=physical_metric(fwd,un,f,theta,label=label,nonlinear_model=nh)
            nonlinear_diag['Newton']=newton;diag=dict(matching_linear=linear_diag,NH80_stress=nonlinear_diag)
            checkpoint_npz(snapshot,load_context,NH80_field=un,unit=stream['unit'],force_SHA256=array_sha(f),diagnostics_json=json.dumps(diag))
        for law in LAWS:diagnostics[law][label]=diag[law]
        nonlinear.append(un)
        write(directory/'truth-progress-LOCAL.json',dict(completed_labels=labels[:j+1],last_snapshot_SHA256=sha(snapshot),index=index))
    matching=model.select(labels)(stream['unit'][None])[0]
    sparse_response=physics.field_queries(4,linear,study.observation_points()).reshape(len(labels),24)
    if np.max(abs(matching-sparse_response))>2e-12:raise FloatingPointError('Truth modal/direct sparse parity failed')
    nh_response=physics.field_queries(4,np.asarray(nonlinear),study.observation_points()).reshape(len(labels),24)
    np.savez_compressed(directory/'truth-LOCAL.npz',unit=stream['unit'],theta=theta,
        prefix_noise=stream['prefix_noise'],probe_noise=stream['probe_noise'],random_realization=stream['random_realization'],
        matching_linear=matching,NH80_stress=nh_response)
    record=dict(labels=list(labels),physics=diagnostics,all_physical_valid=all(v['valid']for rows in diagnostics.values()for v in rows.values()),
        raw_SHA256=sha(directory/'truth-LOCAL.npz'),completed_loads_not_resimulated=True)
    write(directory/'truth-checkpoint-LOCAL.json',record);return record


def observation_fingerprint(observations):
    h=hashlib.sha256()
    for label,value in observations:h.update(label.encode());h.update(np.ascontiguousarray(value,dtype='<f8').tobytes())
    return h.hexdigest()


def save_posterior(path,result,*,model,observations,targets,numerical_engine,recipe_SHA256):
    final=result.get('local_final_simpson',result.get('local_final'));record=dict(qualified=result['qualified'],diagnostics=result['diagnostics'],
        observations_SHA256=observation_fingerprint(observations),model_fingerprint=model_fingerprint(model),
        numerical_engine=numerical_engine,recipe_SHA256=recipe_SHA256)
    if final is not None:
        unitMAP=final['unit_grid'][int(np.argmax(final['log_likelihood']))]
        map_prediction=model.select(targets)(unitMAP[None]).ravel()
        np.savez_compressed(path.with_suffix('.npz'),quadrature_probability=final['quadrature_probability'],
            mean=final['predictive_mean'],SD=final['predictive_SD'],unit_mean=final['unit_mean'],
            unit_grid=final['unit_grid']if numerical_engine=='conditional_amplitude'else np.empty((0,2)),
            inverse_amplitude=final.get('inverse_amplitude',np.empty(0)),log_ratio=final.get('log_ratio',np.empty(0)),
            unit_CDF=final['unit_CDF'],unit_quantiles=final['unit_quantiles'],MAP_unit=unitMAP,MAP_prediction=map_prediction)
        record['raw_SHA256']=sha(path.with_suffix('.npz'))
    write(path,record)


def one_posterior(path,model,observations,plan,*,numerical_engine='tensor',amplitude_plan=AmplitudePlan(),ratio_predict=None):
    """Checkpoint is final even if unresolved: no repeated numerical attempts."""
    recipe_SHA=hashlib.sha256(json.dumps(dict(numerical_engine=numerical_engine,tensor=asdict(plan),amplitude=asdict(amplitude_plan)),sort_keys=True).encode()).hexdigest()
    if path.exists():
        record=json.loads(path.read_text())
        if record['recipe_SHA256']!=recipe_SHA or record['numerical_engine']!=numerical_engine:raise ValueError('Checkpoint numerical recipe changed')
        if record['observations_SHA256']!=observation_fingerprint(observations)or record['model_fingerprint']!=model_fingerprint(model):raise ValueError('Checkpoint model/data changed')
        if record.get('raw_SHA256')and sha(path.with_suffix('.npz'))!=record['raw_SHA256']:raise ValueError('Posterior checkpoint hash mismatch')
        return record
    if numerical_engine=='tensor':result=infer(model,observations,study.TARGETS,plan=plan)
    elif numerical_engine=='conditional_amplitude':
        callback=ratio_callback(model)if ratio_predict is None else ratio_predict
        result=ConditionalAmplitudeReference(None,observations,study.TARGETS,log_ratio_predict=callback).qualify(plan=amplitude_plan)
    else:raise ValueError('Unknown numerical engine')
    save_posterior(path,result,model=model,observations=observations,targets=study.TARGETS,numerical_engine=numerical_engine,recipe_SHA256=recipe_SHA)
    return json.loads(path.read_text())


def posterior_score(path,target,unit):
    record=json.loads(path.read_text())
    if not record['qualified']:return dict(resolved=False)
    with np.load(path.with_suffix('.npz'))as z:
        rmse=lambda x:float(np.sqrt(np.mean((x-target)**2)))
        q=z['unit_quantiles'];covered=(unit>=q[:,0])&(unit<=q[:,2])
        return dict(resolved=True,posterior_RMS_m=rmse(z['mean']),MAP_RMS_m=rmse(z['MAP_prediction']),
            mean_predictive_SD_m=float(np.sqrt(np.mean(z['SD']**2))),
            E_90percent_covered=float(covered[0]),K_90percent_covered=float(covered[1]),
            E_unit_interval_width=float(q[0,2]-q[0,0]),K_unit_interval_width=float(q[1,2]-q[1,0]))


def policy_scores(choices,joints,*,random_realization):
    result={}
    for family,choice in choices.items():
        rows=joints[family]
        def average(indexes):
            selected=[rows[i][rep]for i in indexes for rep in range(3)]
            if not all(s['resolved']for s in selected):return dict(resolved=False)
            keys=[k for k in selected[0]if k!='resolved']
            return dict(resolved=True,**{k:float(np.mean([s[k]for s in selected]))for k in keys})
        result[family]=dict(proposed=average([choice['proposed']]),information=average([choice['information']]),
            uniform_random=average(range(3)),ancillary_random_realization=average([int(random_realization[list(study.ACTIONS).index(family)])]),
            **{f'fixed_{i}':average([i])for i in range(3)})
    return result


def process_object(index,fwd,model,p,digest,slot,*,ratio_predict=None):
    directory=LOCAL/f'object-{index:02d}-LOCAL';directory.mkdir(parents=True,exist_ok=True)
    terminal=directory/'terminal-LOCAL.json'
    if terminal.exists():return json.loads(terminal.read_text())
    write(directory/'checkpoint.json',dict(stage='truth',index=index,protocol_SHA256=digest))
    if not(directory/'truth-checkpoint-LOCAL.json').exists():generate_truth_fields(index,fwd,model,directory,digest)
    truthmeta=json.loads((directory/'truth-checkpoint-LOCAL.json').read_text())
    if sha(directory/'truth-LOCAL.npz')!=truthmeta['raw_SHA256']:raise ValueError('Truth checkpoint mismatch')
    numerical=dict(numerical_engine=p['numerical_engine'],amplitude_plan=AmplitudePlan(tuple(p['amplitude_plan']['inner']),tuple(p['amplitude_plan']['outer']),p['amplitude_plan']['moment_chunk']),ratio_predict=ratio_predict)
    labels=truthmeta['labels'];plan=plan_from(p);result=dict(index=index,retained=True,protocol_SHA256=digest,physical_valid=truthmeta['all_physical_valid'],laws={})
    prior=np.load(LOCAL/'prior-predictive-LOCAL.npz')['mean']
    for law in LAWS:
        environment_guard(slot,device_mode=p['device_mode'])
        with np.load(directory/'truth-LOCAL.npz')as z:prefix=z[law][labels.index(study.PREFIX)]+z['prefix_noise']
        prefixpath=directory/f'{law}-prefix-LOCAL.json'
        record=one_posterior(prefixpath,model,[(study.PREFIX,prefix)],plan,**numerical)
        seal=directory/f'{law}-choices-seal-LOCAL.json'
        if not seal.exists():
            choices=None
            if record['qualified']:
                with np.load(prefixpath.with_suffix('.npz'))as z:
                    prob=z['quadrature_probability'];unit=z['unit_grid'];amplitude=z['inverse_amplitude'];logratio=z['log_ratio']
                predictor=lambda u,l:model.select(l)(u)
                if p['numerical_engine']=='conditional_amplitude':
                    predictor=study.weighted_hypothesis_predictor(unit,amplitude,logratio,ratio_predict,predictor)
                else:
                    n=plan.simpson[-1];axis=np.linspace(0,1,n);unit=np.stack(np.meshgrid(axis,axis,indexing='ij'),-1).reshape(-1,2)
                choices=study.choices_from_prefix(unit,prob,predictor,chunk=512)
                del unit,prob
            write(seal,dict(protocol_SHA256=digest,prefix_checkpoint_SHA256=sha(prefixpath),
                choices=choices,post_probe_outcomes_read=False,truth_used_for_selection=False))
        sealed=json.loads(seal.read_text())
        if sealed['protocol_SHA256']!=digest or sealed['prefix_checkpoint_SHA256']!=sha(prefixpath)or sealed['post_probe_outcomes_read']or sealed['truth_used_for_selection']:raise ValueError('Causal choices seal mismatch')
        sealSHA=sha(seal)
        # Only after this seal is committed are latent targets and actual probe
        # observations read for scoring/joint inference.
        with np.load(directory/'truth-LOCAL.npz')as z:
            target=z[law][[labels.index(l)for l in study.TARGETS]].ravel();unit=z['unit'];random_realization=z['random_realization']
            oracle=z['matching_linear'][[labels.index(l)for l in study.TARGETS]].ravel()
            prefix_discrepancy=(z[law][labels.index(study.PREFIX)]-z['matching_linear'][labels.index(study.PREFIX)])/study.SIGMA_M
        lawresult=dict(prefix=posterior_score(prefixpath,target,unit),prior_RMS_m=float(np.sqrt(np.mean((prior-target)**2))),
            known_true_parameter_linear_oracle_RMS_m=float(np.sqrt(np.mean((oracle-target)**2))),
            prefix_whitened_oracle_RMS=float(np.sqrt(np.mean(prefix_discrepancy**2))),
            target_whitened_oracle_RMS=float(np.sqrt(np.mean(((oracle-target)/study.SIGMA_M)**2))),
            policies={},all_joint_qualified=True,posteriors=1)
        if not record['qualified']:
            lawresult.update(all_joint_qualified=False,unresolved_reason='Prefix numerical failure; no action chosen')
            result['laws'][law]=lawresult;continue
        joints={}
        for family,actions in study.ACTIONS.items():
            joints[family]=[]
            for action in actions:
                rows=[]
                for rep in range(3):
                    if sha(seal)!=sealSHA:raise ValueError('Sealed choices mutated')
                    with np.load(directory/'truth-LOCAL.npz')as z:y=z[law][labels.index(action)]+z['probe_noise'][rep]
                    path=directory/f'{law}-{action}-r{rep}-LOCAL.json'
                    rec=one_posterior(path,model,[(study.PREFIX,prefix),(action,y)],plan,**numerical)
                    rows.append(posterior_score(path,target,unit));lawresult['all_joint_qualified']&=rec['qualified'];lawresult['posteriors']+=1
                    write(directory/'checkpoint.json',dict(stage='joint',law=law,action=action,replicate=rep,index=index,protocol_SHA256=digest,sealed_choices_SHA256=sealSHA))
                    environment_guard(slot,device_mode=p['device_mode'])
                joints[family].append(rows)
        lawresult['policies']=policy_scores(sealed['choices'],joints,random_realization=random_realization)
        result['laws'][law]=lawresult
    result['numerically_resolved']=all(r['prefix']['resolved']and r['all_joint_qualified']for r in result['laws'].values())
    result['completed']=True;write(terminal,result)
    write(directory/'checkpoint.json',dict(stage='terminal',index=index,protocol_SHA256=digest,terminal_SHA256=sha(terminal)))
    return result


def worker(slot,digest):
    p=validate(digest);environment_guard(slot,device_mode=p['device_mode']);release_gate(digest);p,fwd,model=cached_model(digest,slot=slot);records=[];ratio_predict=ratio_callback(model)if p['numerical_engine']=='conditional_amplitude'else None
    for index in range(slot,16,p['resources']['workers']):
        try:record=process_object(index,fwd,model,p,digest,slot,ratio_predict=ratio_predict)
        except Exception as exc:
            record=dict(index=index,protocol_SHA256=digest,retained=True,completed=False,physical_valid=False,numerically_resolved=False,error=type(exc).__name__+': '+str(exc))
            write(LOCAL/f'object-{index:02d}-LOCAL/terminal-LOCAL.json',record)
        records.append(dict(index=index,completed=record['completed']))
    write(LOCAL/f'worker-{slot}-terminal.json',dict(slot=slot,completed=True,pid=os.getpid(),protocol_SHA256=digest,records=records))
    return dict(slot=slot,objects=len(records))


def worker_command(slot,digest,*,guard_only=False,device_mode='cpu'):
    cpu=','.join(map(str,resources_for(device_mode)['worker_CPU_sets'][slot]))
    cmd=['taskset','-c',cpu,sys.executable,'-m','experiments.run_surface_p2_study_v02','guard'if guard_only else'worker','--slot',str(slot),'--protocol-sha256',digest,'--device-mode',device_mode]
    return cmd


def _processes(pid):
    """Owned process group RSS; unrelated machine work is never included."""
    rows=[]
    for path in Path('/proc').iterdir():
        if not path.name.isdigit():continue
        try:
            text=(path/'stat').read_text();fields=text[text.rfind(')')+2:].split()
            if int(fields[2])==pid:
                rss=int(fields[21])*os.sysconf('SC_PAGE_SIZE');rows.append((int(path.name),rss))
        except(FileNotFoundError,ProcessLookupError,PermissionError):pass
    return rows


def GPU_usage(worker_pids,*,cap_bytes=8*1024**3):
    """Count only owned worker groups; unrelated jobs and desktopGPU3 are never managed."""
    owned={p for pid in worker_pids for p,_ in _processes(pid)}
    query=subprocess.run(['nvidia-smi','--query-compute-apps=pid,used_gpu_memory,gpu_uuid','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True,timeout=5)
    totals={uuid:0 for uuid in GPU_UUIDS}
    for line in query.stdout.splitlines():
        cells=[c.strip()for c in line.split(',')]
        if len(cells)!=3:raise ValueError('Unexpected GPU memory telemetry')
        pid,memory,uuid=cells
        if int(pid)not in owned:continue
        if uuid not in GPU_UUIDS:raise RuntimeError('Owned process on unapproved physical GPU')
        try:totals[uuid]+=int(memory)*1024**2
        except ValueError as error:raise RuntimeError('Owned GPU VRAM telemetry unavailable')from error
    if any(v>cap_bytes for v in totals.values()):raise MemoryError('Owned per-physical-GPU VRAM cap exceeded')
    return totals


def supervise(digest):
    p=validate(digest);environment_guard(device_mode=p['device_mode']);host_preflight(p['device_mode']);release_gate(digest)
    limits=p['resources']
    if not(LOCAL/'cache-ready.json').exists():raise ValueError('Shared cache must be separately built/admitted before workers')
    if(LOCAL/'execution.json').exists():raise ValueError('Existing execution requires explicit root-reviewed recovery; no implicit relaunch')
    start=time.monotonic();env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
    processes=[];files=[];terminal=[];error=None
    try:
        for slot in range(limits['workers']):
            f=(LOCAL/f'worker-{slot}.log').open('w');files.append(f)
            worker_env=dict(env,CUDA_VISIBLE_DEVICES=''if p['device_mode']=='cpu'else GPU_UUIDS[slot%3],CUBLAS_WORKSPACE_CONFIG=':4096:8')
            child=subprocess.Popen(worker_command(slot,digest,device_mode=p['device_mode']),cwd=ROOT,env=worker_env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
            processes.append((slot,child,time.monotonic()))
        while processes:
            live=[];total=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
            for slot,child,began in processes:
                code=child.poll();rows=_processes(child.pid);rss=sum(r for _,r in rows)
                if code is not None:
                    child.wait();terminal.append(dict(slot=slot,pid=child.pid,exit_code=code,reaped=True))
                    if code:raise RuntimeError(f'Worker{slot} exit{code}')
                else:
                    live.append((slot,child,began));total+=rss
                    if rss>RESOURCES['worker_RSS_cap_bytes']or time.monotonic()-began>limits['worker_wall_cap_s']:raise RuntimeError(f'Worker{slot} resource cap')
            if total>limits['aggregate_RSS_cap_bytes']or time.monotonic()-start>limits['global_wall_cap_s']:raise RuntimeError('Aggregate RSS/global wall cap')
            if p['device_mode']=='gpu' and available_memory()<limits['RAM_reserve_bytes']:raise MemoryError('30GiB host reserve breached')
            GPU_memory=GPU_usage([c.pid for _,c,_ in live],cap_bytes=limits['study_VRAM_cap_bytes_per_physical_GPU'])if p['device_mode']=='gpu'else{}
            write(LOCAL/'execution.json',dict(study_GPU_VRAM_bytes=GPU_memory,status='RUNNING',completed=False,pid=os.getpid(),protocol_SHA256=digest,
                live=[dict(slot=s,pid=c.pid)for s,c,_ in live],records=terminal,aggregate_RSS_bytes=total))
            processes=live;environment_guard(device_mode=p['device_mode']);time.sleep(1)
    except BaseException as exc:
        error=type(exc).__name__+': '+str(exc)
        for slot,child,_ in processes:
            if child.poll()is None:
                try:os.killpg(child.pid,signal.SIGTERM)
                except ProcessLookupError:pass
        deadline=time.monotonic()+5
        while any(c.poll()is None for _,c,_ in processes)and time.monotonic()<deadline:time.sleep(.1)
        for slot,child,_ in processes:
            if child.poll()is None:
                try:os.killpg(child.pid,signal.SIGKILL)
                except ProcessLookupError:pass
            code=child.wait()
            if not any(r['pid']==child.pid for r in terminal):terminal.append(dict(slot=slot,pid=child.pid,exit_code=code,reaped=True))
    finally:
        for f in files:f.close()
    absent=all(not _processes(row['pid'])for row in terminal)
    receipt=dict(status='COMPLETED'if error is None else'FAILED',completed=error is None,pid=os.getpid(),protocol_SHA256=digest,
        records=terminal,live=[],all_worker_groups_absent=absent,wall_s=time.monotonic()-start,error=error)
    write(LOCAL/'execution.json',receipt)
    if error is not None:raise RuntimeError(error)
    return receipt


def aggregate(digest):
    p=validate(digest);execution=json.loads((LOCAL/'execution.json').read_text())
    if not execution['completed']or execution['protocol_SHA256']!=digest or len(execution['records'])!=p['resources']['workers'] or not execution['all_worker_groups_absent']or any(_processes(r['pid'])for r in execution['records']):raise ValueError('All declared owned workers must be terminal/reaped/absent')
    objects=[json.loads((LOCAL/f'object-{i:02d}-LOCAL/terminal-LOCAL.json').read_text())for i in range(16)]
    if [r['index']for r in objects]!=list(range(16))or any(r['protocol_SHA256']!=digest for r in objects):raise ValueError('All16 retained protocol-bound objects required')
    admitted=all(r.get('completed')and r.get('physical_valid')and r.get('numerically_resolved')for r in objects)
    summary=dict(schema='surface-p2-study-v02',numerical_engine=p['numerical_engine'],selected_numerical_plan=p['selected_numerical_plan'],plugin_estimator='Highest likelihood among final positive quadrature nodes; approximate intrinsic MAP',protocol_SHA256=digest,execution_SHA256=sha(LOCAL/'execution.json'),all16_retained=True,
        objects=16,physically_valid_objects=sum(r.get('physical_valid',False)for r in objects),
        numerically_resolved_objects=sum(r.get('numerically_resolved',False)for r in objects),all_comparison_inputs_qualified=bool(admitted),
        laws={},raw_LOCAL=True,no_real_physical_calibration_claim=True,static_ideal_displacement_only=True)
    if admitted:
        for law in LAWS:
            rows=[r['laws'][law]for r in objects];report=dict(prefix_mean_RMS_m=float(np.mean([r['prefix']['posterior_RMS_m']for r in rows])),
                prefix_MAP_mean_RMS_m=float(np.mean([r['prefix']['MAP_RMS_m']for r in rows])),
                prefix_E_interval_coverage=float(np.mean([r['prefix']['E_90percent_covered']for r in rows])),
                prefix_K_interval_coverage=float(np.mean([r['prefix']['K_90percent_covered']for r in rows])),
                prefix_mean_predictive_SD_m=float(np.mean([r['prefix']['mean_predictive_SD_m']for r in rows])),
                prior_mean_RMS_m=float(np.mean([r['prior_RMS_m']for r in rows])),
                known_parameter_linear_oracle_mean_RMS_m=float(np.mean([r['known_true_parameter_linear_oracle_RMS_m']for r in rows])),families={})
            for family in study.ACTIONS:
                policies=('proposed','information','uniform_random','fixed_0','fixed_1','fixed_2')
                familyreport={policy:dict(mean_target_RMS_m=float(np.mean([r['policies'][family][policy]['posterior_RMS_m']for r in rows])),
                    mean_MAP_RMS_m=float(np.mean([r['policies'][family][policy]['MAP_RMS_m']for r in rows])),
                    E_interval_coverage=float(np.mean([r['policies'][family][policy]['E_90percent_covered']for r in rows])),
                    K_interval_coverage=float(np.mean([r['policies'][family][policy]['K_90percent_covered']for r in rows])),
                    mean_predictive_SD_m=float(np.mean([r['policies'][family][policy]['mean_predictive_SD_m']for r in rows])),
                    mean_E_unit_interval_width=float(np.mean([r['policies'][family][policy]['E_unit_interval_width']for r in rows])),
                    mean_K_unit_interval_width=float(np.mean([r['policies'][family][policy]['K_unit_interval_width']for r in rows])))for policy in policies}
                comparisons={}
                for baseline in policies[1:]:
                    differences=[r['policies'][family][baseline]['posterior_RMS_m']-r['policies'][family]['proposed']['posterior_RMS_m']for r in rows]
                    comparisons[baseline]=study.paired_cluster_summary(differences)
                familyreport['paired_baseline_minus_proposed_RMS_m']=comparisons
                familyreport['positive_CI_against_all_fixed']=all(comparisons[f'fixed_{i}']['CI95'][0]>0 for i in range(3))
                familyreport['scope']='Paired16-object conditional finite-model comparison; bootstrap exploratory, not multiplicity-adjusted population evidence'
                report['families'][family]=familyreport
            summary['laws'][law]=report
    else:summary['unresolved_reason']='No aggregate policy advantage computed: all16 physical/numerical gates required; every failed object retained'
    write(LOCAL/'summary.json',summary);return summary


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['prepare','cache','worker','run','aggregate','guard'])
    parser.add_argument('--protocol-sha256');parser.add_argument('--admission');parser.add_argument('--admission-sha256');parser.add_argument('--slot',type=int);parser.add_argument('--device-mode',choices=['cpu','gpu'],default='cpu');parser.add_argument('--numerical-engine',choices=['tensor','conditional_amplitude'],default='tensor')
    a=parser.parse_args()
    if a.mode=='prepare':out=prepare(a.admission,a.admission_sha256,device_mode=a.device_mode,numerical_engine=a.numerical_engine)
    elif a.mode=='cache':out=create_shared_cache(a.protocol_sha256)
    elif a.mode=='worker':out=worker(a.slot,a.protocol_sha256)
    elif a.mode=='run':out=supervise(a.protocol_sha256)
    elif a.mode=='aggregate':out=aggregate(a.protocol_sha256)
    else:environment_guard(a.slot,device_mode=a.device_mode);out=dict(guard_passed=True,CPU=sorted(os.sched_getaffinity(0)))
    print(json.dumps(out))
