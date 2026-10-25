"""Source-only prospective fresh64 paired fixed-probe calibration runner.

No truth/noise is generated on import/prepare. All896 inference contexts and
64independent object denominators are fixed before a root-approved run.
"""
import argparse,ast,hashlib,json,os,resource,signal,subprocess,sys,time,traceback,shutil
from dataclasses import asdict
from pathlib import Path
import numpy as np
from experiments import campaign_fresh64_nonlinear_calibration_v01 as study
from experiments import run_surface_p2_study_v04 as oldstudy
from experiments import run_surface_full_atlas_feasibility_v01 as full
from experiments import run_surface_nonlinear_posterior_cost_v03 as cost
from experiments import run_surface_nonlinear_GPU_control_v02 as GPU_control
from experiments.run_surface_nonlinear_posterior_cost_v01 import full_atlas_gate
from experiments.campaign_nonlinear_residual_atlas_v01 import NonlinearResidualAtlas
from experiments.campaign_nonlinear_residual_atlas_fast_v01 import FastResidualAtlas
from experiments.campaign_nonlinear_conditional_reference_v03 import NonlinearConditionalReferenceV03 as NonlinearConditionalReference
from experiments.campaign_surface_conditional_amplitude_v02 import ConditionalAmplitudeReferenceV02,AmplitudePWLPlan
ROOT=oldstudy.ROOT;LOCAL=ROOT/'runs/operator-dev/fresh64-nonlinear-calibration-v01'
SOURCE='experiments/run_fresh64_nonlinear_calibration_v01.py';TEST='tests/test_run_fresh64_nonlinear_calibration_v01.py'
DESIGN=ROOT/'runs/operator-dev/clamped-surface-bridge-review-v01/fresh64-nonlinear-calibration-protocol-draft-v02.json'
# Root/peer's prospective seed/phase/statistical design. No actual draws here.
DESIGN_SHA='c95488db3358dcfad3305e904f99a2a39cf3acbd6d60edaf2f82e3d832e60eb0'
PLAN=AmplitudePWLPlan(moment_chunk=1024)
PARAMETER_PHASE_LABELS=tuple(phase+'_'+parameter for phase in study.PHASES for parameter in('E','K'))
sha=oldstudy.sha;write=oldstudy.write;array_sha=oldstudy.array_sha
record=full.record


def canonical_plan():return dict(inner=list(PLAN.inner),outer=list(PLAN.outer),moment_chunk=PLAN.moment_chunk)


def sources():
    seen=set();pending=[SOURCE,TEST,'tests/test_fresh64_nonlinear_calibration_v01.py']
    while pending:
        name=pending.pop()
        if name in seen:continue
        seen.add(name)
        for node in ast.walk(ast.parse((ROOT/name).read_text())):
            modules=[]
            if isinstance(node,ast.Import):modules=[v.name for v in node.names]
            elif isinstance(node,ast.ImportFrom)and node.level==0:modules=[node.module or '']+[(node.module+'.'+v.name)for v in node.names if node.module]
            for module in modules:
                path=module.replace('.','/')+'.py'
                if (ROOT/path).is_file()and path not in seen:pending.append(path)
    return {name:sha(ROOT/name)for name in sorted(seen)}


def validate_resources(r):
    n=r.get('workers');GiB=1024**3
    if type(n)is not int or not 1<=n<=16 or r.get('worker_CPU_sets')!=[[i]for i in range(n)]or r.get('controller_CPU')!=16 or r.get('threads_per_worker')!=1 or r.get('device_mode')not in('cpu','gpu'):raise ValueError('1..16fixed CPUworker slots/controller16/one-thread/explicitmode')
    if r.get('worker_RSS_cap_bytes')!=4*GiB or r.get('aggregate_RSS_cap_bytes')!=n*4*GiB or r.get('RAM_reserve_bytes')!=30*GiB or r.get('disk_reserve_bytes')!=30*GiB or r.get('maximum_writes_bytes')!=12*GiB or r.get('host_preflight_available_bytes')!=r['aggregate_RSS_cap_bytes']+30*GiB:raise ValueError('Exact ownedRSS/reserve/write caps')
    if type(r.get('worker_wall_cap_s'))is not int or not 30<=r['worker_wall_cap_s']<=18000 or type(r.get('global_wall_cap_s'))is not int or not r['worker_wall_cap_s']<=r['global_wall_cap_s']<=19800:raise ValueError('Explicit cost-justified bounded worker/globalwalls')
    expected=list(oldstudy.GPU_UUIDS)if r['device_mode']=='gpu'else[]
    if r.get('allowed_GPU_UUIDs')!=expected or r.get('worker_GPU_UUIDs')!=([oldstudy.GPU_UUIDS[i%3]for i in range(n)]if expected else[]):raise ValueError('PinnedGPU0–2sharingonly; GPU3forbidden')
    if r.get('study_VRAM_cap_bytes_per_physical_GPU')!=(8*GiB if expected else 0):raise ValueError('Owned8GiB/deviceVRAMcap')
    return r


def guard(p,slot=None):
    r=validate_resources(p['resources']);cpu=16 if slot is None else slot
    if slot is not None and not 0<=slot<r['workers']:raise ValueError('Declared worker slot')
    device=''if slot is None or r['device_mode']=='cpu'else r['worker_GPU_UUIDs'][slot]
    if set(os.sched_getaffinity(0))!={cpu}or os.environ.get('CUDA_VISIBLE_DEVICES')!=device or any(os.environ.get(k)!='1'for k in('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'))or(device and os.environ.get('CUBLAS_WORKSPACE_CONFIG')!=':4096:8'):raise ValueError('ExactCPU/thread/deviceguard')
    if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024>r['worker_RSS_cap_bytes']:raise MemoryError('Own4GiB RSS')
    if shutil.disk_usage(ROOT).free<r['disk_reserve_bytes']or oldstudy.available_memory()<r['RAM_reserve_bytes']:raise MemoryError('30GiBphysicalRAM/diskreserves')
    if LOCAL.exists()and sum(f.stat().st_size for f in LOCAL.rglob('*')if f.is_file())>r['maximum_writes_bytes']:raise OSError('Own12GiBwrites')


def prepare(cost_protocol_SHA256,resources,GPU_control_protocol_SHA256=None):
    validate_resources(resources)
    if set(os.sched_getaffinity(0))!={16}or os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('CPU16controller/noGPUprepare')
    if sha(DESIGN)!=DESIGN_SHA:raise ValueError('Frozenfresh64designSHA')
    receipt=cost.completed_cost_gate(cost_protocol_SHA256)
    GPU_receipt=GPU_control.completed_GPU_control_gate(GPU_control_protocol_SHA256)
    full_records=receipt['origin_v01']['full_atlas']
    full_summary,full_protocol,admission=full_atlas_gate(ROOT/full_records['summary']['path'],full_records['summary']['sha256'])
    if oldstudy.available_memory()<resources['host_preflight_available_bytes']:raise MemoryError('Cost-reviewed hostbudget+30GiB reserve')
    p=dict(schema='fresh64-nonlinear-calibration-v01',status='FROZEN_ROOT_RELEASE_REQUIRED',source_SHA256=sources(),design=record(DESIGN),cost_protocol_SHA256=cost_protocol_SHA256,cost_receipt=receipt,GPU_control_protocol_SHA256=GPU_control_protocol_SHA256,GPU_control_receipt=GPU_receipt,full_atlas=admission,atlas_fingerprint=full_summary['atlas_fingerprint'],resources=resources,objects=64,expected_posteriors=896,parameter_seed_base=study.PARAMETER_SEED_BASE,noise_seed_base=study.NOISE_SEED_BASE,bootstrap_seed=study.BOOTSTRAP_SEED,models=list(study.MODELS),phases=list(study.PHASES),parameter_phase_labels=list(PARAMETER_PHASE_LABELS),truth_labels=list(study.TRUTH_LABELS),target_labels=list(study.TARGETS),plan=canonical_plan(),all_objects_retained=True,no_replacements=True,no_adaptive_policy=True,old16_data_read=False,raw_data_LOCAL=True,newtruth_solver='Same admitted n4 NH80/originalgates, sharedrestLU/modelperobject; onlyprefixexcitationgated')
    if (LOCAL/'protocol.json').exists():raise FileExistsError('No refreeze')
    write(LOCAL/'protocol.json',p);return record(LOCAL/'protocol.json')


def validate(digest):
    p=json.loads((LOCAL/'protocol.json').read_text())
    if sha(LOCAL/'protocol.json')!=digest or p['source_SHA256']!=sources()or p['design']!=record(DESIGN)or sha(DESIGN)!=DESIGN_SHA or p['cost_receipt']!=cost.completed_cost_gate(p['cost_protocol_SHA256']):raise ValueError('Frozen source/design/cost/protocol')
    if p['GPU_control_receipt']!=GPU_control.completed_GPU_control_gate(p['GPU_control_protocol_SHA256']):raise ValueError('Closed qualified GPUcontrol frozen receipt')
    _,_,admission=full_atlas_gate(ROOT/p['full_atlas']['summary']['path'],p['full_atlas']['summary']['sha256'])
    if admission!=p['full_atlas']or p['objects']!=64 or p['expected_posteriors']!=896 or p['parameter_phase_labels']!=list(PARAMETER_PHASE_LABELS)or p['parameter_seed_base']!=study.PARAMETER_SEED_BASE or p['noise_seed_base']!=study.NOISE_SEED_BASE or p['bootstrap_seed']!=study.BOOTSTRAP_SEED or p['plan']!=canonical_plan():raise ValueError('Exactly64/896/fixedseeds/labels/unchangednumerics')
    validate_resources(p['resources']);return p


def release_gate(digest):
    p=validate(digest);release=json.loads((LOCAL/'release.json').read_text())
    if release.get('approved')is not True or release.get('protocol_SHA256')!=digest or release.get('source_SHA256')!=p['source_SHA256']:raise ValueError('Concrete root release required before anydraw')
    return p


def load_atlas(p):
    path=ROOT/p['full_atlas']['archive']['path']
    if sha(path)!=p['full_atlas']['archive']['sha256']:raise ValueError('Admittedatlasbytes')
    with np.load(path,allow_pickle=False)as z:atlas=NonlinearResidualAtlas(z['delta'],tuple(str(v)for v in z['labels']),provenance_fingerprint=json.loads((ROOT/p['full_atlas']['summary']['path']).read_text())['protocol_SHA256'])
    if atlas.fingerprint!=p['atlas_fingerprint']:raise ValueError('Exactadmittedatlasfingerprint')
    return FastResidualAtlas(atlas,table_cache_bytes=8*1024**2)


def checkpoint(path,context,**arrays):return oldstudy.checkpoint_npz(path,context,**arrays)
def verify(path,context):return oldstudy.verify_npz(path,context)


def data_for_object(index,directory,digest):
    path=directory/'draws-LOCAL.npz';context=dict(protocol_SHA256=digest,index=index,kind='freshparameter_and_independentnoise')
    if oldstudy.checkpoint_present(path):verify(path,context)
    else:checkpoint(path,context,**study.draw_fresh_object(index))
    with np.load(path,allow_pickle=False)as z:return {key:z[key].copy()for key in z.files}


def truth_fields(index,directory,p,digest,fwd,draws):
    from experiments.campaign_exact_rest_pcg_v01 import ExactRestPCG
    from experiments.campaign_surface_preconditioned_p2_v01 import PreconditionedP2Reference
    from experiments.campaign_generalized_stability_v01 import generalized_stability
    from experiments.run_clamped_surface_p2_qualification_v03 import deformation
    from experiments.run_clamped_surface_p2_qualification_v01 import reaction_diagnostics
    from experiments.campaign_surface_p2_force_v03 import load_catalog
    from experiments.campaign_surface_p2_static_v02 import point_query_matrix
    from experiments.campaign_surface_p2_study_design_v01 import observation_points
    force_catalog=load_catalog(4);E,K=draws['physical_parameters'];rest=E*fwd.Kunit_free+K*fwd.W_free;directions=None;model=None;Q=point_query_matrix(fwd,observation_points());rows=[];responses=[]
    for label in study.TRUTH_LABELS:
        path=directory/(label+'-truth-LOCAL.npz');meta=path.with_suffix('.json');context=dict(protocol_SHA256=digest,index=index,label=label,draws_SHA256=sha(directory/'draws-LOCAL.npz'))
        present=oldstudy.checkpoint_present(path)
        if present!=meta.exists():raise ValueError('Orphantruthdiagnostics before anyPDE')
        if present:
            verify(path,context);row=json.loads(meta.read_text())
            if row['context']!=context or row['raw_SHA256']!=sha(path):raise ValueError('Truthdiagnosticscontext')
        else:
            if directions is None:
                directions=ExactRestPCG(rest);model=PreconditionedP2Reference.from_forward(fwd,E,K,quadrature_order=80,direction_solver=directions)
            force=force_catalog[label];uL=np.zeros_like(fwd.X);uL.ravel()[fwd.free]=directions.rest_solve(force.ravel()[fwd.free]);lr=(E*fwd.Kunit+K*fwd.W)@uL.ravel()-force.ravel();linear_R=float(np.linalg.norm(lr[fwd.free]))
            if linear_R>1e-10 or np.any(uL[fwd.fixed]!=0):raise RuntimeError('Originalexactlineargates')
            u,diag=model.solve(force,initial=uL,maximum_iterations=40,force_tolerance_N=1e-10);state=model.evaluate(u,force,tangent=True);R=float(np.linalg.norm(state['residual'].ravel()[fwd.free]));d=deformation(fwd,u);eq=reaction_diagnostics(fwd,u,force,E,K,state['residual']);stable=generalized_stability(state['tangent'][fwd.free][:,fwd.free],rest,directions.rest_LU.solve)
            material=d['minimum_J']>=.8 and d['maximum_absolute_symmetric_strain_eigenvalue']<=.05;passed=material and R<=1e-10 and eq['passed']and stable['passed']and np.all(u[fwd.fixed]==0)and(label!=study.PREFIX or d['physical_passed'])
            checkpoint(path,context,displacement_m=u,linear_displacement_m=uL,response_m=np.asarray(Q@u.ravel()[fwd.free]),unit=draws['unit'])
            row=dict(context=context,raw_SHA256=sha(path),passed=bool(passed),material_valid=bool(material),deformation=d,equilibrium=eq,stability=stable,original_free_residual_N=R,Newton=diag);write(meta,row)
        with np.load(path,allow_pickle=False)as z:responses.append(z['response_m'].copy())
        rows.append(row)
    return np.array(responses),dict(physical_valid=all(row['passed']for row in rows),loads=rows)


def one_posterior(path,model,atlas,ratio,observations,true_unit,*,kind,context):
    if kind not in study.MODELS:raise ValueError('Explicit pairedlikelihoodmodel')
    backend='conditional_amplitude_PWL_v02'if kind=='linear'else'nonlinear_conditional_reference_v03_MIS'
    recipe=hashlib.sha256(json.dumps(dict(kind=kind,numerical_backend=backend,plan=canonical_plan(),targets=study.TARGETS,atlas=atlas.fingerprint if kind=='nonlinear'else None),sort_keys=True).encode()).hexdigest()
    binding=dict(context,recipe_SHA256=recipe,model_fingerprint=oldstudy.model_fingerprint(model),observations_SHA256=oldstudy.observation_fingerprint(observations),true_unit_SHA256=array_sha(true_unit))
    if path.exists():
        rec=json.loads(path.read_text())
        if rec['context']!=binding:raise ValueError('Posterior data/model/recipe context changed')
        if rec.get('raw_SHA256')and sha(path.with_suffix('.npz'))!=rec['raw_SHA256']:raise ValueError('Posterior rawchecksum changed')
        return rec
    if path.with_suffix('.npz').exists():raise ValueError('Orphanposteriorarchive beforeanyinference')
    if kind=='linear':reference=ConditionalAmplitudeReferenceV02(None,observations,study.TARGETS,log_ratio_predict=ratio,retain_hypotheses=False)
    else:reference=NonlinearConditionalReference(ratio,None,observations,study.TARGETS,residual_predict=atlas.predict,retain_hypotheses=False,evaluation_chunk=1024)
    result=reference.qualify(plan=PLAN);final=result['local_final'];strict=cost.absolute_engine_witnesses(result);qualified=bool(result['qualified']and strict['passed']);record=dict(context=binding,qualified=qualified,physical_valid=True,diagnostics=result['diagnostics'],strict_absolute_checks=strict,kind=kind,numerical_backend=backend)
    if final is not None:
        arrays=dict(mean=final['predictive_mean'],SD=final['predictive_SD'],unit_mean=final['unit_mean'],unit_covariance=final['unit_covariance'],unit_CDF=final['unit_CDF'],unit_quantiles=final['unit_quantiles'],MAP_unit=final['plugin_unit'],MAP_prediction=final['plugin_prediction'],MAP_log_likelihood=final['plugin_log_likelihood'])
        if qualified:
            calibration=study.exact_density_calibration(reference,final,true_unit,outer=PLAN.outer[-1],inner=PLAN.inner[-1]);arrays.update(calibration)
            record.update({key:np.asarray(value).tolist()for key,value in calibration.items()})
        temporary=path.with_suffix('.tmp.npz');np.savez_compressed(temporary,**arrays);os.replace(temporary,path.with_suffix('.npz'));record['raw_SHA256']=sha(path.with_suffix('.npz'))
    write(path,record);return record


def process_object(index,fwd,model,atlas,p,digest,slot,ratio):
    directory=LOCAL/f'object-{index:02d}';directory.mkdir(exist_ok=True);summary=directory/'summary-LOCAL.json'
    if summary.exists():
        a=json.loads(summary.read_text())
        if a['protocol_SHA256']!=digest or a['index']!=index:raise ValueError('Objectcheckpointcontext')
        # Do not silently trust a summary without all bound archives.
        for rel,h in a['checkpoint_SHA256'].items():
            if sha(directory/rel)!=h:raise ValueError('Objectcheckpointtamper')
        return a
    result=dict(index=index,protocol_SHA256=digest,completed=False,physical_valid=False,posteriors={m:{}for m in study.MODELS},oracle_RMS_m={})
    try:
        draws=data_for_object(index,directory,digest);responses,physics=truth_fields(index,directory,p,digest,fwd,draws);result['physical_valid']=physics['physical_valid'];result['physics']=physics
        target=responses[[study.TRUTH_LABELS.index(label)for label in study.TARGETS]].ravel();unit=draws['unit']
        oracle_path=directory/'oracle-predictions-LOCAL.npz';oracle_context=dict(protocol_SHA256=digest,index=index,draws_SHA256=sha(directory/'draws-LOCAL.npz'),atlas_fingerprint=atlas.fingerprint,model_fingerprint=oldstudy.model_fingerprint(model),truth_SHA256={label:sha(directory/(label+'-truth-LOCAL.npz'))for label in study.TRUTH_LABELS})
        if oldstudy.checkpoint_present(oracle_path):verify(oracle_path,oracle_context)
        else:
            linear=np.asarray(model.select(study.TARGETS)(unit[None]))[0].ravel();nonlinear=linear+atlas.predict(unit[None],study.TARGETS)[0].ravel()
            checkpoint(oracle_path,oracle_context,linear_mean_m=linear,nonlinear_mean_m=nonlinear)
        with np.load(oracle_path,allow_pickle=False)as z:linear=z['linear_mean_m'];nonlinear=z['nonlinear_mean_m']
        if linear.shape!=(96,)or nonlinear.shape!=(96,)or not np.isfinite(linear).all()or not np.isfinite(nonlinear).all():raise ValueError('Knownparameter96targetoraclepredictions')
        result['oracle_RMS_m']=dict(linear=float(np.sqrt(np.mean((linear-target)**2))),nonlinear=float(np.sqrt(np.mean((nonlinear-target)**2))))
        contexts=[('prefix',None,None)]+[(study.FAMILIES[f]+f'-r{reset}',f,reset)for f in range(2)for reset in range(3)]
        for key,family,reset in contexts:
            observations=study.paired_observations(responses,draws['prefix_noise'],draws['probe_noise'],family=family,reset=reset)
            for kind in study.MODELS:
                guard(p,slot);path=directory/(kind+'-'+key+'-posterior-LOCAL.json');context=dict(protocol_SHA256=digest,index=index,phase=key,draws_SHA256=sha(directory/'draws-LOCAL.npz'))
                if not result['physical_valid']:
                    record=dict(context=context,qualified=False,physical_valid=False,blocked='Directtruthphysical/numericalqualificationfailed')
                    if path.exists()and json.loads(path.read_text())!=record:raise ValueError('Changedblockedrecord')
                    if not path.exists():write(path,record)
                else:
                    try:
                        record=one_posterior(path,model,atlas,ratio,observations,unit,kind=kind,context=context)
                    except Exception as error:
                        if isinstance(error,(MemoryError,TimeoutError,InterruptedError)):raise
                        record=dict(context=context,qualified=False,physical_valid=True,execution_failure=dict(type=type(error).__name__,message=str(error)))
                        if not path.exists()and not path.with_suffix('.npz').exists():write(path,record)
                    if record['qualified']:
                        with np.load(path.with_suffix('.npz'),allow_pickle=False)as z:record=dict(record,target_RMS_m=float(np.sqrt(np.mean((z['mean']-target)**2))),MAP_target_RMS_m=float(np.sqrt(np.mean((z['MAP_prediction']-target)**2))))
                result['posteriors'][kind][key]=record
                # Compact private checkpoint supports crash diagnosis without changing recipes.
                write(directory/'progress-LOCAL.json',dict(index=index,protocol_SHA256=digest,posterior_records=sum(len(v)for v in result['posteriors'].values()),qualified=sum(r.get('qualified',False)for v in result['posteriors'].values()for r in v.values())))
        result['completed']=True
    except Exception as error:
        result['execution_failure']=dict(type=type(error).__name__,message=str(error),traceback=traceback.format_exc())
        if isinstance(error,(MemoryError,TimeoutError,InterruptedError)):raise
    finally:
        for kind in study.MODELS:
            for key in ['prefix']+[family+f'-r{reset}'for family in study.FAMILIES for reset in range(3)]:
                if key not in result['posteriors'][kind]:
                    result['posteriors'][kind][key]=dict(qualified=False,physical_valid=result['physical_valid'],attempted=False,blocked='Objectphysical/numericalexecutionunresolved')
        result['checkpoint_SHA256']={f.name:sha(f)for f in directory.iterdir()if f.is_file()and f!=summary and not f.name.endswith('.tmp.npz')}
        write(summary,result)
    return result


def exact_ratio_callback(model):return GPU_control.exact_ratio_callback(model)


def worker(slot,digest):
    p=release_gate(digest);guard(p,slot);directory=LOCAL/f'worker-{slot}'
    if directory.exists():raise FileExistsError('One actual worker invocation; explicit recovery required')
    directory.mkdir();start=time.monotonic();state=dict(completed=False,slot=slot,pid=os.getpid(),protocol_SHA256=digest,objects=[])
    def save():state.update(wall_s=time.monotonic()-start,peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024);write(directory/'summary-LOCAL.json',state)
    def deadline(*_):raise TimeoutError('Actual frozenworker deadline')
    def terminated(*_):raise InterruptedError('Explicit supervisorSIGTERMcleanup')
    signal.signal(signal.SIGALRM,deadline);signal.signal(signal.SIGTERM,terminated);signal.alarm(p['resources']['worker_wall_cap_s'])
    try:
        fwd,model=oldstudy.make_model(device_mode=p['resources']['device_mode'],slot=slot);atlas=load_atlas(p);ratio=exact_ratio_callback(model);save()
        for index in range(slot,64,p['resources']['workers']):
            guard(p,slot);result=process_object(index,fwd,model,atlas,p,digest,slot,ratio);guard(p,slot)
            state['objects'].append(dict(index=index,completed=result['completed'],physical_valid=result['physical_valid'],posterior_contexts=sum(len(v)for v in result['posteriors'].values()),qualified=sum(r.get('qualified',False)for v in result['posteriors'].values()for r in v.values()),record=record(LOCAL/f'object-{index:02d}/summary-LOCAL.json')));save()
        state['completed']=True;state['memo']=ratio.diagnostics();save()
    except BaseException as error:state['failure']=dict(type=type(error).__name__,message=str(error),traceback=traceback.format_exc());save();raise
    finally:signal.alarm(0);save()
    return state


def worker_command(slot,digest,*,guard_only=False):
    if type(slot)is not int or not 0<=slot<16:raise ValueError('Worker CPUslots0..15')
    return ['taskset','-c',str(slot),sys.executable,'-u','-m','experiments.run_fresh64_nonlinear_calibration_v01','worker-guard'if guard_only else'worker','--slot',str(slot),'--sha',digest]


def placeholder(index,reason):
    keys=['prefix']+[family+f'-r{reset}'for family in study.FAMILIES for reset in range(3)]
    return dict(index=index,physical_valid=False,completed=False,missing=True,posteriors={kind:{key:dict(qualified=False,physical_valid=False,attempted=False,blocked=reason)for key in keys}for kind in study.MODELS})


def aggregate(digest):
    p=validate(digest);records=[]
    for index in range(64):
        path=LOCAL/f'object-{index:02d}/summary-LOCAL.json'
        if not path.exists():records.append(placeholder(index,'Objectnotcompleted; retained64denominator'));continue
        a=json.loads(path.read_text())
        if a['protocol_SHA256']!=digest or a['index']!=index:raise ValueError('Exactly64orderedoriginalobjectcontexts')
        for name,h in a['checkpoint_SHA256'].items():
            if sha(path.parent/name)!=h:raise ValueError('Objectcommitteddata/posteriorbinding')
        records.append(a)
    if [r['index']for r in records]!=list(range(64))or tuple(PARAMETER_PHASE_LABELS)!=study.CALIBRATION_LABELS:raise ValueError('64rows and fixedphase-majorE/K sixlabels')
    calibration=study.summarize_calibration(records);prediction=study.paired_prediction_metrics(records)
    arrays={}
    for kind in study.MODELS:
        covered,pit=study.primary_arrays(records,kind);arrays[kind+'_covered']=covered;arrays[kind+'_PIT']=pit
    target=LOCAL/'primary-calibration-LOCAL.npz'
    if target.exists():raise FileExistsError('No calibrationraw overwrite')
    np.savez_compressed(target,**arrays)
    qualified=sum(v.get('qualified',False)for row in records for model in row.get('posteriors',{}).values()for v in model.values())
    result=dict(completed=all(r.get('completed',False)for r in records),all64_denominators_retained=True,objects=64,expected_posteriors=896,posterior_contexts_retained=sum(len(v)for row in records for v in row.get('posteriors',{}).values()),qualified_posteriors=qualified,physical_valid_objects=sum(r.get('physical_valid',False)for r in records),comparison_blocked=calibration['comparison_blocked']or prediction['comparison_blocked']or qualified!=896,protocol_SHA256=digest,primary_calibration=calibration,paired_metrics=prediction,raw_calibration=record(target),no_active_policy_claim=True,old16_excluded=True)
    write(LOCAL/'summary-LOCAL.json',result);return result


def supervise(digest):
    p=release_gate(digest);guard(p);r=p['resources']
    if oldstudy.available_memory()<r['host_preflight_available_bytes']:raise MemoryError('Host preflight budget+30GiB reserve')
    if (LOCAL/'execution.json').exists()or any(LOCAL.glob('worker-*')):raise FileExistsError('No implicit relaunch')
    start=time.monotonic();children=[];logs=[];state=dict(completed=False,terminal=False,controller_pid=os.getpid(),protocol_SHA256=digest,release=record(LOCAL/'release.json'),workers=[])
    def save():state['wall_s']=time.monotonic()-start;write(LOCAL/'execution.json',state)
    def stopped(*_):raise InterruptedError('Explicit ownedcontrollerstop')
    handlers={sig:signal.signal(sig,stopped)for sig in(signal.SIGTERM,signal.SIGINT)}
    try:
        for slot in range(r['workers']):
            env={**os.environ,'CUDA_VISIBLE_DEVICES':r['worker_GPU_UUIDs'][slot]if r['device_mode']=='gpu'else'','CUBLAS_WORKSPACE_CONFIG':':4096:8','OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1'}
            log=(LOCAL/f'worker-{slot}.log').open('xb');logs.append(log);child=subprocess.Popen(worker_command(slot,digest),cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);children.append(child);state['workers'].append(dict(slot=slot,pid=child.pid,reaped=False,started_s=time.monotonic()-start));save()
        while any(child.poll()is None for child in children):
            guard(p);rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
            for child,row in zip(children,state['workers']):
                if child.poll()is not None:
                    row.update(reaped=True,exit_code=child.wait())
                    if child.returncode!=0:raise RuntimeError('Ownedworker abnormalexit; preservepartial64denominators')
                    continue
                own=full.owned_RSS(child);rss+=own
                if own>r['worker_RSS_cap_bytes']or time.monotonic()-start-row['started_s']>r['worker_wall_cap_s']:raise TimeoutError('IndependentworkerRSS/deadline')
            state['maximum_observed_aggregate_RSS_bytes']=max(state.get('maximum_observed_aggregate_RSS_bytes',0),rss)
            if rss>r['aggregate_RSS_cap_bytes']or time.monotonic()-start>r['global_wall_cap_s']:raise TimeoutError('AggregateRSS/globaldeadline')
            if r['device_mode']=='gpu':state['owned_GPU_usage']=oldstudy.GPU_usage([child.pid for child in children],cap_bytes=r['study_VRAM_cap_bytes_per_physical_GPU'])
            save();time.sleep(.5)
        for child,row in zip(children,state['workers']):row.update(reaped=True,exit_code=child.wait())
        if any(child.returncode!=0 for child in children):raise RuntimeError('Workerexitnotzero')
        for child in children:full.prefix.absent(child.pid)
        summary=aggregate(digest);state.update(completed=summary['completed'],comparison_blocked=summary['comparison_blocked'],summary=record(LOCAL/'summary-LOCAL.json'))
    except BaseException as error:state['failure']=dict(type=type(error).__name__,message=str(error),traceback=traceback.format_exc())
    finally:
        for child,row in zip(children,state['workers']):row.update(reaped=True,exit_code=full.prefix.cost.cleanup(child))
        for child in children:full.prefix.absent(child.pid)
        for row in state['workers']:
            worker_summary=LOCAL/f"worker-{row['slot']}/summary-LOCAL.json"
            if worker_summary.exists():row['summary']=record(worker_summary)
        if not(LOCAL/'summary-LOCAL.json').exists():
            try:
                summary=aggregate(digest);state.update(comparison_blocked=True,summary=record(LOCAL/'summary-LOCAL.json'))
            except Exception as error:state['terminal_aggregate_failure']=dict(type=type(error).__name__,message=str(error))
        for log in logs:log.close()
        for sig,handler in handlers.items():signal.signal(sig,handler)
        state['terminal']=True;save()
    return state


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['prepare','worker','run','worker-guard']);parser.add_argument('--sha');parser.add_argument('--slot',type=int);parser.add_argument('--cost-sha');parser.add_argument('--resources-json');parser.add_argument('--gpu-control-sha');args=parser.parse_args()
    if args.action=='prepare':print(json.dumps(prepare(args.cost_sha,json.loads(Path(args.resources_json).read_text()),args.gpu_control_sha)))
    elif args.action=='worker':a=worker(args.slot,args.sha);print(json.dumps(dict(completed=a['completed'],objects=len(a['objects']))))
    elif args.action=='run':a=supervise(args.sha);print(json.dumps(dict(completed=a['completed'],terminal=a['terminal'])));raise SystemExit(0 if a['completed'] else 1)
    else:
        guarded=dict(resources=json.loads(Path(args.resources_json).read_text()))if args.resources_json else validate(args.sha)
        guard(guarded,args.slot);print('guard passed')
