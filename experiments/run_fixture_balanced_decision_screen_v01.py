"""NEW balanced-target fixed36 LINEAR development task; old negative immutable.

All144 category references and1728 action/bank/outcome jobs remain in the
prospective queue. No fresh cohort/noise, nonlinear or policy-advantage claim.
"""
import argparse,ast,gc,hashlib,json,os,resource,shutil,signal,subprocess,sys,time,zipfile
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from experiments import campaign_fixture_gaussian_decision_screen_v01 as design
from experiments import run_fixture_gaussian_linear_screen_v01 as linear
from experiments import campaign_fixture_balanced_held_v01 as balanced
from experiments import run_fixture_gaussian_decision_screen_v01 as original
from experiments import audit_fixture_gaussian_decision_screen_v01 as original_audit
from scipy.sparse.linalg import splu
from experiments.run_clamped_surface_p2_qualification_v01 import reaction_diagnostics
from experiments import audit_fixture_gaussian_linear_screen_v01 as linear_audit
from experiments import run_layout_grouped_allactions_v01 as old
from experiments import audit_layout_grouped_allactions_v01 as oldaudit
from experiments.campaign_surface_p2_inference_v01 import LabelledModal
from experiments.campaign_surface_conditional_amplitude_v01 import RatioModal,ExactRatioMemo
from experiments.campaign_surface_conditional_amplitude_v02 import ConditionalAmplitudeReferenceV02,AmplitudePWLPlan
from experiments.campaign_surface_gpu_modal_v01 import Float64GPUModal
ROOT=old.ROOT;LOCAL=ROOT/'runs/operator-dev/fixture-balanced-decision-screen-v01'
SOURCE='experiments/run_fixture_balanced_decision_screen_v01.py';TEST='tests/test_run_fixture_balanced_decision_screen_v01.py'
sha,record,read,write=old.sha,old.record,old.read,old.write
RESOURCES=dict(builder_CPU=17,controller_CPU=18,worker_CPUs=[0,1,2],threads=1,builder_wall_s=900,worker_wall_s=6600,global_wall_s=7200,builder_RSS_bytes=8*1024**3,worker_RSS_bytes=8*1024**3,controller_RSS_bytes=4*1024**3,aggregate_RSS_bytes=28*1024**3,torch_allocated_bytes=512*1024**2,owned_VRAM_bytes=2*1024**3,RAM_reserve_bytes=30*1024**3,disk_reserve_bytes=30*1024**3,maximum_writes_bytes=512*1024**2,memo_bytes_per_category=64*1024**2)
LINEAR_PROTOCOL='b25a555a08568a97733ffd92378cca142f022ec766b1f515394aec5acf6b04c0'
LINEAR_RESULT_SHA='347a798cd6c8c08698a9e60ebb1c6051ba8966ff7f5e258611d1f141e4d79dd0'
LINEAR_REVIEW_SHA='fdf2ec441664804a746875b2b30b6b2e527250e521c76b0e4d30dfd0f6722b04'
REFERENCE_PLAN=dict(inner=[128,256,512],outer=[128,256,512],moment_chunk=1024)
DECISION_LEVELS=((512,8),(512,16),(512,32),(512,64),(256,64))
def sources():
    todo=[SOURCE,TEST,'experiments/audit_fixture_balanced_decision_screen_v01.py','tests/test_audit_fixture_balanced_decision_screen_v01.py','experiments/campaign_fixture_balanced_held_v01.py','tests/test_fixture_balanced_held_v01.py',original.SOURCE,original.TEST,original_audit.SOURCE,original_audit.TEST,'experiments/campaign_fixture_gaussian_decision_screen_v01.py','tests/test_fixture_gaussian_decision_screen_v01.py',oldaudit.SOURCE,oldaudit.TEST,'experiments/audit_fixture_gaussian_linear_screen_v01.py','tests/test_audit_fixture_gaussian_linear_screen_v01.py',*old.sources(),*linear.sources()];seen=set()
    while todo:
        name=todo.pop()
        if name in seen:continue
        seen.add(name)
        for node in ast.walk(ast.parse((ROOT/name).read_text())):
            modules=[]
            if isinstance(node,ast.Import):modules=[x.name for x in node.names]
            elif isinstance(node,ast.ImportFrom)and node.level==0:modules=[node.module or '']+[(node.module+'.'+x.name)for x in node.names if node.module]
            for module in modules:
                f=module.replace('.','/')+'.py'
                if(ROOT/f).is_file()and f not in seen:todo.append(f)
    return {p:sha(ROOT/p)for p in sorted(seen)}
def verify_records(value):old.parity.old.old.verify_mechanics_receipt(value)
def absent(pid):old.parity.old.old.mechanical_screen.shared.old.absent(pid)
def layout_id(index):
    if type(index)is not int or not 0<=index<4:raise ValueError('Canonical category index0..3')
    return linear.mechanics.bank.LAYOUTS[index]['id']
def origin_gate():
    # Called only during metadata preparation, AFTER both independent audits.
    digest=sha(linear.LOCAL/'protocol.json')
    if digest!=LINEAR_PROTOCOL or sha(linear.LOCAL/'result-LOCAL.json')!=LINEAR_RESULT_SHA or sha(linear.LOCAL/'independent-review.json')!=LINEAR_REVIEW_SHA:raise ValueError('Exact root-reviewed completed Gaussian72 inputs')
    p,e,r=linear_audit.metadata_barrier();review=read(linear.LOCAL/'independent-review.json')
    if not(e.get('completed')is True and e.get('terminal')is True and e.get('worker_exit_code')==0 and e.get('cleanup_exit_code')==0 and r.get('all72_retained')is True and r.get('all792_load_responses_retained')is True and review.get('audit_passed')is True and review['source_result']==record(linear.LOCAL/'result-LOCAL.json')):raise ValueError('Closed independently audited Gaussian72 LINEAR development prerequisite')
    if r['raw']!=record(ROOT/r['raw']['path']):raise ValueError('Original raw response bytes')
    for pid in(e['controller_pid'],e['worker_pid']):absent(pid)
    previous=read(old.LOCAL/'independent-terminal-review.json');pe=read(old.LOCAL/'execution.json');old.validate(oldaudit.EXPECTED_PROTOCOL)
    if previous['summary'].get('qualified')is not True or previous['summary'].get('actions_qualified')!=8 or previous['source_SHA256']!={s:sha(ROOT/s)for s in(oldaudit.SOURCE,oldaudit.TEST)}:raise ValueError('Independently qualified old all48 numerical development path')
    for pid in(pe['controller_pid'],*[x['pid']for x in pe['devices']]):absent(pid)
    metadata=[]
    for case in design.worklist():
        matches=[row for row in p['plan']['worklist']if row['layout']==layout_id(case['layout'])and all(row[k]==case[k]for k in('n','E_Pa','K_N_m'))]
        if len(matches)!=1:raise ValueError('Each fixed36 prefix has exact admitted n6 row')
        metadata.append(dict(case_index=case['index'],raw_row=matches[0]['index']))
    old_digest=original_audit.EXPECTED_PROTOCOL;original.validate(old_digest);oe=read(original.LOCAL/'execution.json');oa=read(original.LOCAL/'independent-terminal-review.json');oc=read(original.LOCAL/'root-closure.json');ob=read(original.LOCAL/'builder-result.json')
    if not(oe.get('completed')is True and oa['summary'].get('integrity_passed')is True and oa['summary']['adequacy']['adequacy_qualified']is False and oa['summary'].get('execution_clean')is True and oc.get('root_exit_code')==0 and oc.get('root_reaped')is True and oc['execution_SHA256']==sha(original.LOCAL/'execution.json')):raise ValueError('Exact closed old36 numerical-positive adequacy-negative remains immutable')
    if oc['source_SHA256']!=original.sources()or oa['source_SHA256']!={s:sha(ROOT/s)for s in(original_audit.SOURCE,original_audit.TEST)}or oe['builder']['result']!=record(original.LOCAL/'builder-result.json'):raise ValueError('Old36 exact source/audit/builder bindings')
    verify_records(oe['case_records'])
    for pid in oc['owned_pids']:absent(pid)
    witnesses=[]
    for item in oe['case_records']:
        row=read(ROOT/item['record']['path']);verify_records(row['partial_artifacts']);witnesses.append(dict(index=row['index'],categories=[v['record']for v in row['categories']],decisions=[v['record']for v in row['decision_levels']]))
    witnesses.sort(key=lambda v:v['index'])
    if[v['index']for v in witnesses]!=list(range(36)):raise ValueError('All old36 category/decision witnesses')
    return dict(old36_protocol=record(original.LOCAL/'protocol.json'),old36_release=record(original.LOCAL/'release.json'),old36_execution=record(original.LOCAL/'execution.json'),old36_review=record(original.LOCAL/'independent-terminal-review.json'),old36_closure=record(original.LOCAL/'root-closure.json'),old36_builder=record(original.LOCAL/'builder-result.json'),old36_prefixes=ob['prefixes'],old36_models=ob['models'],old36_witnesses=witnesses,linear_protocol=record(linear.LOCAL/'protocol.json'),linear_release=record(linear.LOCAL/'release.json'),linear_execution=record(linear.LOCAL/'execution.json'),linear_result=record(linear.LOCAL/'result-LOCAL.json'),linear_review=record(linear.LOCAL/'independent-review.json'),linear_raw=r['raw'],old_all48_protocol=record(old.LOCAL/'protocol.json'),old_all48_execution=record(old.LOCAL/'execution.json'),old_all48_review=record(old.LOCAL/'independent-terminal-review.json'),prefix_rows=metadata)
def prepare():
    guard('controller');origin=origin_gate()
    if(LOCAL/'protocol.json').exists():raise FileExistsError('No protocol refreeze')
    p=dict(schema='fixture-balanced-decision-screen-v01',source_SHA256=sources(),origin=origin,resources=RESOURCES,worklist=design.worklist(),devices=list(old.parity.GPU.ALLOWED_UUIDS),split=[[i for i in range(36)if i%3==k]for k in range(3)],reference_plan=REFERENCE_PLAN,decision_levels=[list(x)for x in DECISION_LEVELS],action_worklist=old.worklist(),gates=dict(outcome_relative=1e-4,latent_relative=1e-3),adequacy_gates=design.ADEQUACY,first_fixed_case_checkpoint=0,coefficient_response_bound_m=2e-14,unchanged_evidence_logZ_tolerance=1e-8,unchanged_layout_probability_tolerance=1e-10,new_held_direct_residual_N=1e-10,new_held_modal_parity_m=2e-12,new_target_design=balanced.design(),no_RNG=True,no_NH_PDE=True,scope='NEW balanced future-load objective; old negative immutable; all36 fixed saved-prefix matching-LINEAR development; no calibrated/NH/active-advantage admission; numerical indicators are not rigorous bounds.')
    write(LOCAL/'protocol.json',p);return record(LOCAL/'protocol.json')
def validate(digest):
    p=read(LOCAL/'protocol.json');r=read(LOCAL/'release.json')
    if sha(LOCAL/'protocol.json')!=digest or p['source_SHA256']!=sources()or p['resources']!=RESOURCES or p['worklist']!=design.worklist()or p['split']!=[[i for i in range(36)if i%3==k]for k in range(3)]or p['devices']!=list(old.parity.GPU.ALLOWED_UUIDS)or p['reference_plan']!=REFERENCE_PLAN or p['decision_levels']!=[list(x)for x in DECISION_LEVELS]or p['action_worklist']!=old.worklist()or p['gates']!=dict(outcome_relative=1e-4,latent_relative=1e-3)or p['adequacy_gates']!=design.ADEQUACY or p['new_target_design']!=balanced.design()or p['coefficient_response_bound_m']!=2e-14 or p['unchanged_evidence_logZ_tolerance']!=1e-8 or p['unchanged_layout_probability_tolerance']!=1e-10 or p['new_held_direct_residual_N']!=1e-10 or p['new_held_modal_parity_m']!=2e-12:raise ValueError('Exact frozen development sources/config/worklist')
    verify_records(p['origin'])
    if r.get('approved')is not True or r.get('protocol_SHA256')!=digest or r.get('source_SHA256')!=p['source_SHA256']or r.get('origin')!=p['origin']:raise ValueError('Root release BEFORE prefixes/models/GPU')
    for key,pids in(('linear_execution',('controller_pid','worker_pid')),('old_all48_execution',('controller_pid',))):
        e=read(ROOT/p['origin'][key]['path'])
        for k in pids:absent(e[k])
        if key=='old_all48_execution':
            for row in e['devices']:absent(row['pid'])
    oe=read(ROOT/p['origin']['old36_execution']['path'])
    for pid in[oe['controller_pid'],oe['builder']['pid'],*[v['pid']for v in oe['workers']]]:absent(pid)
    return p

def guard(role,slot=None):
    cpu=RESOURCES['worker_CPUs'][slot]if role=='worker'else RESOURCES[role+'_CPU']
    if set(os.sched_getaffinity(0))!={cpu}or any(os.environ.get(k)!='1'for k in('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS')):raise ValueError('Exact prospective own affinity and one-thread guard')
    if role=='worker':old.parity.GPU.pin_config(old.parity.GPU.ALLOWED_UUIDS[slot])
    elif os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('Builder/controller no CUDA')
    cap=RESOURCES['worker_RSS_bytes']if role=='worker'else RESOURCES[role+'_RSS_bytes']
    if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024>=cap:raise MemoryError('Own role RSS cap')
    available=int(next(x.split()[1]for x in Path('/proc/meminfo').read_text().splitlines()if x.startswith('MemAvailable:')))*1024
    if available<RESOURCES['RAM_reserve_bytes']or shutil.disk_usage(ROOT).free<RESOURCES['disk_reserve_bytes']+RESOURCES['maximum_writes_bytes']:raise MemoryError('Prospective30GiB RAM/disk reserve')
    if LOCAL.exists()and sum(x.stat().st_size for x in LOCAL.rglob('*')if x.is_file())>RESOURCES['maximum_writes_bytes']:raise MemoryError('Own512MiB writes')

def prefix_row(archive,row):
    """Read ONLY first four values of one fixed NPY row, never other columns."""
    if type(row)is not int or not 0<=row<72:raise ValueError('Fixed original row index')
    with zipfile.ZipFile(archive)as z,z.open('responses.npy')as f:
        version=np.lib.format.read_magic(f)
        if version==(1,0):shape,fortran,dtype=np.lib.format.read_array_header_1_0(f)
        elif version==(2,0):shape,fortran,dtype=np.lib.format.read_array_header_2_0(f)
        else:raise ValueError('Known NPY header version')
        if tuple(shape)!=(72,60)or fortran or dtype!=np.dtype('float64'):raise ValueError('Exact row-major finite scalarresponse archive')
        f.seek(f.tell()+row*60*dtype.itemsize);raw=f.read(4*dtype.itemsize)
        if len(raw)!=4*dtype.itemsize:raise ValueError('Complete prefix row bytes')
    return design.prefix_observation(np.frombuffer(raw,dtype=dtype))
def model_fingerprint(eigenvalues,coefficients,labels):
    h=hashlib.sha256()
    for a in(eigenvalues,coefficients):h.update(np.ascontiguousarray(a,dtype=np.float64).tobytes())
    h.update(json.dumps(list(labels),separators=(',',':')).encode());return h.hexdigest()
def coefficient_invariance(model,old_model):
    a=np.asarray(model.modal.eigenvalues);b=np.asarray(old_model.modal.eigenvalues)
    if not np.array_equal(a,b)or model.labels[:12]!=old_model.labels[:12]:raise ValueError('Identical allmode spectrum/first12 labels; no eigenvalue clipping')
    delta=np.abs(model.modal.coefficients[:12]-old_model.modal.coefficients[:12]);denominator=2000+np.where(a<0,30.,5.)*a
    if np.any(denominator<=0):raise ValueError('Positive full-prior modal denominators')
    bound=float(np.max(np.sum(delta/denominator[None,:],axis=1)))
    if bound>2e-14:raise ValueError('First12 coefficient full-prior response bound2e-14m')
    return dict(all_eigenvalues_byte_identical=True,first12_labels_identical=True,maximum_coefficient_difference=float(delta.max()),maximum_full_prior_response_difference_bound_m=bound,formula='sum(abs(delta_coefficient)/(Emin+Kextreme*lambda)); identical eigenvalues, Kextreme=Kmax for negative lambda else Kmin',floating_point_roundoff_not_certified=True)

def held_measurements(fields,n):
    points=balanced.measurement_points();q=balanced.geometry.field_queries(n,np.asarray(fields),points);loads=[v['id']for v in balanced.load_specs()[9:]];out=[]
    for spec in balanced.measurement_specs()[12:]:
        j=loads.index(spec['load']);i=next(i for i,p in enumerate(points)if np.array_equal(p,spec['point_m']));out.append(float(q[j,i]@np.asarray(spec['projection'])))
    return np.array(out)

def separation_energy(responses,cases):
    responses=np.asarray(responses,float)
    if responses.shape!=(36,48)or len(cases)!=36 or not np.isfinite(responses).all():raise ValueError('All36 newheld direct-response rows')
    per=[];energy=np.zeros(2);pairs=0
    for E in design.E_GRID:
        for K in design.K_GRID:
            ids=[i for i,spec in enumerate(cases)if(spec['E_Pa'],spec['K_N_m'])==(E,K)]
            if len(ids)!=4 or{cases[i]['layout']for i in ids}!=set(range(4)):raise ValueError('All4 supportcategories at each E/K')
            value=np.zeros(2)
            for j,i in enumerate(ids):
                for k in ids[j+1:]:value+=np.sum((responses[i]-responses[k]).reshape(2,24)**2,axis=1);pairs+=1
            energy+=value
            if value.sum()>0:per.append(float(value[0]/value.sum()))
    total=float(energy.sum());share=(energy/total).tolist()if total>0 else None
    return dict(same_E_K_layout_pairs=pairs,held_blocks=2,total_separation_squared_energy_m2=total,energy_share=share,first_block_share_range=[min(per),max(per)]if per else None,diagnostic_only_not_gate=True,not_assumed_equal=True)

def build(digest):
    p=validate(digest);guard('builder');directory=LOCAL/'models';directory.mkdir(parents=True,exist_ok=True)
    if(LOCAL/'builder-result.json').exists()or any(directory.iterdir()):raise FileExistsError('No model-build retry/partial reuse')
    start=time.monotonic();out=dict(completed=False,pid=os.getpid(),protocol_SHA256=digest,release=record(LOCAL/'release.json'),models=[],prefixes=p['origin']['old36_prefixes'],new_held_validations=[],old_negative_preserved=True,no_new_NH=True)
    responses=[];cases=[]
    try:
        if len(out['prefixes'])!=36 or[v['index']for v in out['prefixes']]!=list(range(36)):raise ValueError('Exactly36 original hash-reused prefixes')
        verify_records(out['prefixes']);labels=tuple(s['id']for s in balanced.measurement_specs());prefix_labels=labels[:4];target_labels=labels[12:]
        for layout in range(4):
            begin=time.monotonic();f=balanced.make_forward(6,layout_id(layout));loads,load_info=balanced.load_catalog(6);model=balanced.make_measurement_modal(f,loads);maximum=0.
            force_record_path=directory/f'held-forces-layout-{layout}-LOCAL.npz'
            original_entry=p['origin']['old36_models'][layout]
            with np.load(ROOT/original_entry['record']['path'],allow_pickle=False)as z:
                old_model=LabelledModal(SimpleNamespace(eigenvalues=z['eigenvalues'].copy(),coefficients=z['coefficients'].copy(),output_shape=(1,60)),tuple(str(v)for v in z['labels']))
            invariance=coefficient_invariance(model,old_model);held_ids=[v['id']for v in balanced.load_specs()[9:]];force=np.array([loads[name]for name in held_ids]);rhs=force.reshape(2,-1).T;np.savez_compressed(force_record_path,forces=force,load_ids=np.array(held_ids));force_record=record(force_record_path);force_SHA256=[hashlib.sha256(np.ascontiguousarray(v).tobytes()).hexdigest()for v in force]
            for case in p['worklist']:
                if case['layout']!=layout:continue
                unit=np.log(np.array([case['E_Pa']/2000,case['K_N_m']/5]))/np.log([4,6])
                with np.load(ROOT/out['prefixes'][case['index']]['record']['path'],allow_pickle=False)as z:observed=design.prefix_observation(z['observed_prefix'])
                maximum=max(maximum,float(np.max(abs(model.select(prefix_labels)(unit[None])[0,:,0]-observed))))
                E,K=case['E_Pa'],case['K_N_m'];lu=splu(E*f.Kunit_free+K*f.W_free);u=np.zeros_like(rhs);u[f.free]=lu.solve(rhs[f.free]);residual=(E*f.Kunit+K*f.W)@u-rhs;norms=np.linalg.norm(residual[f.free],axis=0);fields=u.T.reshape(2,len(f.X),3)
                if not np.isfinite(fields).all()or np.any(norms>1e-10)or np.any(fields[:,f.fixed]!=0):raise ValueError('All72 newheld original residual/exact-clamp gates')
                reaction=[reaction_diagnostics(f,fields[j],force[j],E,K)for j in range(2)]
                if not all(v['passed']for v in reaction):raise ValueError('All72 known-load equilibrium/reaction/moment gates')
                value=held_measurements(fields,6);modal=model.select(target_labels)(unit[None])[0,:,0];error=float(np.max(abs(value-modal)))
                if error>2e-12:raise ValueError('All72 newheld modal/direct2pm gates')
                out['new_held_validations'].append(dict(index=case['index'],load_ids=held_ids,force_record=force_record,force_SHA256=force_SHA256,free_residual_l2_N=norms.tolist(),exact_clamp_zeros=True,maximum_modal_parity_m=error,equilibrium=reaction));responses.append(value);cases.append(case);guard('builder')
            if maximum>2e-12:raise ValueError('Rebuilt allmode n6 prefix/direct oldarchive parity2pm')
            fp=model_fingerprint(model.modal.eigenvalues,model.modal.coefficients,labels);path=directory/f'layout-{layout}-LOCAL.npz';np.savez_compressed(path,eigenvalues=model.modal.eigenvalues,coefficients=model.modal.coefficients,labels=np.array(labels));out['models'].append(dict(layout=layout,record=record(path),old_model=original_entry['record'],model_fingerprint=fp,all_modes=len(model.modal.eigenvalues),output_shape=[1,60],maximum_prefix_parity_m=maximum,first12_invariance=invariance,construction_and9paired_held_solves_s=time.monotonic()-begin,all_mode_diagnostics=model.modal.diagnostics));del f,loads,model,old_model,lu,u,fields;gc.collect();guard('builder')
        if len(out['new_held_validations'])!=36 or{v['index']for v in out['new_held_validations']}!=set(range(36)):raise ValueError('Every fixed36 paired newheld validation retained')
        raw=directory/'new-held-responses-LOCAL.npz';np.savez_compressed(raw,responses=np.array(responses),case_index=np.array([c['index']for c in cases]));out.update(held_response_record=record(raw),new_held_equilibria=72,closed36_target_separation=separation_energy(responses,cases),completed=True)
    except BaseException as error:out['failure']=dict(type=type(error).__name__,message=str(error));raise
    finally:out.update(wall_s=time.monotonic()-start,peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024);write(LOCAL/'builder-result.json',out)
    return out

def builder_gate(p,digest):
    result=read(LOCAL/'builder-result.json')
    if not result.get('completed')or result['protocol_SHA256']!=digest or result['release']!=record(LOCAL/'release.json')or len(result['models'])!=4 or[m['layout']for m in result['models']]!=list(range(4))or result.get('new_held_equilibria')!=72 or len(result.get('new_held_validations',[]))!=36 or result['prefixes']!=p['origin']['old36_prefixes']or len(result['prefixes'])!=36 or[x['index']for x in result['prefixes']]!=list(range(36)):raise ValueError('Complete exact4models/36prefixes builder')
    absent(result['pid']);verify_records(result)
    return result

def load_models(metadata):
    models=[]
    for item in metadata['models']:
        with np.load(ROOT/item['record']['path'],allow_pickle=False)as z:e=z['eigenvalues'].copy();c=z['coefficients'].copy();labels=tuple(str(v)for v in z['labels'])
        if model_fingerprint(e,c,labels)!=item['model_fingerprint']or c.shape!=(60,len(e))or len(e)!=item['all_modes']:raise ValueError('Exact allmode coefficient archive fingerprint')
        models.append(LabelledModal(SimpleNamespace(eigenvalues=e,coefficients=c,output_shape=(1,60)),labels))
    return models

def make_ratio_callbacks(models,uuid):
    out=[]
    for model in models:
        cpu=RatioModal(model.modal,model.labels);gpu=Float64GPUModal(model.modal,model.labels,physical_UUID=uuid,allowed_UUIDs=old.parity.GPU.ALLOWED_UUIDS,allocated_cap_bytes=RESOURCES['torch_allocated_bytes']);selected={}
        def callback(z,labels,cpu=cpu,gpu=gpu,selected=selected):
            z=np.asarray(z,float)
            if len(z)<16:return cpu(np.exp(z),labels)
            labels=tuple(labels)
            if labels not in selected:selected[labels]=gpu.ratio_select(labels)
            return np.concatenate([selected[labels](z[i:i+4096])for i in range(0,len(z),4096)])
        fp=model_fingerprint(model.modal.eigenvalues,model.modal.coefficients,model.labels);memo=ExactRatioMemo(callback,model_fingerprint=fp,maximum_bytes=RESOURCES['memo_bytes_per_category']);out.append((cpu,memo))
    return out

def reference_evidence_invariance(old_diagnostics,new_diagnostics):
    oldlevels={(v['outer'],v['inner']):v for v in old_diagnostics['levels']};newlevels={(v['outer'],v['inner']):v for v in new_diagnostics['levels']};expected={(512,128),(512,256),(512,512),(128,512),(256,512)}
    if set(oldlevels)!=expected or set(newlevels)!=expected:raise ValueError('Exact old/new five latent-state identities')
    errors=[];same=True
    for key,oldvalue in oldlevels.items():
        newvalue=newlevels[key]
        if oldvalue['status']=='computed':
            if newvalue['status']!='computed':same=False;continue
            error=abs(oldvalue['log_evidence']-newvalue['log_evidence']);same=same and bool(np.isfinite(error)and error<=1e-8);errors.append(error)
    return dict(passed=bool(same),maximum_logZ_difference=max(errors,default=None),gate=1e-8,only_unchanged_prefix_density=True)

def decision_evidence_invariance(old_metadata,new_metadata):
    a=np.asarray(old_metadata['log_evidence']);b=np.asarray(new_metadata['log_evidence']);p=np.asarray(old_metadata['layout_probability']);q=np.asarray(new_metadata['layout_probability'])
    if any(v.shape!=(4,)or not np.isfinite(v).all()for v in(a,b,p,q)):raise ValueError('All four original/new categorical evidences/probabilities')
    logz=float(np.max(abs(a-b)));prob=float(np.max(abs(p-q)));return dict(passed=bool(logz<=1e-8 and prob<=1e-10),maximum_logZ_difference=logz,maximum_layout_probability_difference=prob,logZ_gate=1e-8,layout_probability_gate=1e-10,only_unchanged_prefix_density=True)

def policy(observation,callbacks,names,directory,checkpoint,old_witnesses):
    """No generating layout/E/K, candidate response or held truth input."""
    observation=design.prefix_observation(observation);refs=[];finals=[];category=[]
    for k,(cpu,memo)in enumerate(callbacks):
        ref=ConditionalAmplitudeReferenceV02(cpu,list(zip(names['prefix'],observation[:,None])),names['target'],log_ratio_predict=memo,retain_hypotheses=False);result=ref.qualify(plan=AmplitudePWLPlan(moment_chunk=1024));check=old.parity.old.reference_check(result);old_record=old_witnesses['categories'][k];before=read(ROOT/old_record['path']);unchanged=reference_evidence_invariance(before['diagnostics'],result['diagnostics']);path=directory/f'category-{k}-LOCAL.json';write(path,dict(qualified=result['qualified'],diagnostics=result['diagnostics'],strict=check,old_reference=old_record,unchanged_prefix=unchanged));category.append(dict(layout=k,qualified=bool(check['passed']and unchanged['passed']),record=record(path)));refs.append(ref);finals.append(result['local_final']);checkpoint()
    out=dict(categories=category,decision_levels=[],jobs=[],qualified=False)
    if not all(v['qualified']for v in category):return out
    screening={}
    for outer,amp in DECISION_LEVELS:
        bank,metadata=old.parity.CPU.decision_bank(refs,names['candidate'],names['target'],outer=outer,amplitude_order=amp);check=old.parity.CPU.reference_moment_checks(metadata,finals);old_record=old_witnesses['decisions'][len(out['decision_levels'])];before=read(ROOT/old_record['path']);unchanged=decision_evidence_invariance(before['metadata'],metadata);moments=bank.prior_moments();fraction=moments['between_layout_variance']/moments['total_variance'];screening[outer,amp]=dict(prior_variance=moments['total_variance'],between_layout_fraction=fraction,maximum_layout_probability=max(metadata['layout_probability']),layout_TV=check['layout_TV']);path=directory/f'decision-{outer}-{amp}-LOCAL.json';write(path,dict(checks=check,old_decision=old_record,unchanged_prefix=unchanged,metadata={k:v for k,v in metadata.items()if k!='category_moments'},posterior_screen=screening[outer,amp]));out['decision_levels'].append(dict(outer=outer,amplitude=amp,checks=check,unchanged_prefix=unchanged,record=record(path)));checkpoint()
        if(outer,amp)in old.VARIANTS:
            # Keep fixed candidates even when a bank witness fails; no winner admitted.
            engine=old.parity.GPU.Float64GroupedGPU(bank,physical_UUID=os.environ['CUDA_VISIBLE_DEVICES'],outcome_chunk=8,allocated_cap_bytes=RESOURCES['torch_allocated_bytes'])
            for action in range(8):
                for order in(8,16):
                    begin=time.monotonic();value=engine.expected_risk(action,linear.mechanics.bank.SIGMA_M,order=order);engine.torch.cuda.synchronize();out['jobs'].append(dict(job=dict(outer=outer,amplitude=amp,action=action,order=order),result=value,wall_s=time.monotonic()-begin));checkpoint()
            del engine
        del bank;gc.collect()
    ranking=old.classify(out['jobs'],dict(outcome_relative=1e-4,latent_relative=1e-3));bank_ok=all(v['checks']['passed']for v in out['decision_levels']if(v['outer'],v['amplitude'])in old.VARIANTS)and all(v['unchanged_prefix']['passed']for v in out['decision_levels'])
    if not bank_ok:ranking.update(qualified=False,unique_best_action=None)
    fine=screening[512,64];fine['between_layout_fraction_indicator']=abs(fine['between_layout_fraction']-screening[512,32]['between_layout_fraction'])+abs(fine['between_layout_fraction']-screening[256,64]['between_layout_fraction']);out.update(qualified=bool(ranking['qualified']),ranking=ranking,posterior_screen=fine);return out

def worker(slot,digest):
    p=validate(digest)
    if type(slot)is not int or not 0<=slot<3:raise ValueError('Fixed worker slot')
    guard('worker',slot);metadata=builder_gate(p,digest);models=load_models(metadata);callbacks=make_ratio_callbacks(models,p['devices'][slot]);labels=models[0].labels;names=dict(prefix=labels[:4],candidate=labels[4:12],target=labels[12:]);start=time.monotonic();state=dict(completed=False,pid=os.getpid(),slot=slot,physical_UUID=p['devices'][slot],protocol_SHA256=digest,release=record(LOCAL/'release.json'),builder=record(LOCAL/'builder-result.json'),queue=[dict(index=i,status='pending')for i in p['split'][slot]],cases=[])
    path=LOCAL/f'worker-{slot}/result-LOCAL.json';path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():raise FileExistsError('No worker retry')
    def checkpoint():guard('worker',slot);state.update(wall_s=time.monotonic()-start,peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024);write(path,state)
    try:
        for item in state['queue']:
            index=item['index'];directory=LOCAL/f'case-{index:02d}';directory.mkdir(parents=True,exist_ok=True)
            if any(directory.iterdir()):raise FileExistsError('No implicit case retry/partial reuse')
            item['status']='running';checkpoint();begin=time.monotonic();row=dict(index=index,case=p['worklist'][index],protocol_SHA256=digest,prefix=metadata['prefixes'][index]['record'],status='unresolved')
            with np.load(ROOT/row['prefix']['path'],allow_pickle=False)as z:observation=design.prefix_observation(z['observed_prefix'])
            try:row.update(policy(observation,callbacks,names,directory,checkpoint,p['origin']['old36_witnesses'][index]));row['status']='qualified'if row['qualified']else'unresolved'
            except (ValueError,FloatingPointError,RuntimeError)as error:row['failure']=dict(type=type(error).__name__,message=str(error))
            row['partial_artifacts']=[record(f)for f in sorted(directory.glob('*'))if f.is_file()];row['wall_s']=time.monotonic()-begin;target=directory/'result-LOCAL.json';write(target,row);state['cases'].append(dict(index=index,status=row['status'],record=record(target)));item['status']=row['status'];checkpoint();gc.collect()
            if index==0:write(LOCAL/'first-fixed-case-cost-checkpoint.json',dict(index=0,result=record(target),wall_s=row['wall_s'],same_unconditional36_queue=True,no_case_selection_or_retry=True,other_planned_cases=35))
        state['completed']=True
    except BaseException as error:state['failure']=dict(type=type(error).__name__,message=str(error));raise
    finally:checkpoint()
    return state

def finalize_child(child,row):
    row.update(exit_code=child.wait(),reaped=True,cleanup_exit_code=old.parity.old.old.mechanical_screen.shared.old.cost.cleanup(child))
    if row['exit_code']!=0 or row['cleanup_exit_code']!=0:raise RuntimeError('Owned stage failed at final wait/cleanup; no stage admission')
    return row
def run(digest):
    p=validate(digest);guard('controller')
    if(LOCAL/'execution.json').exists():raise FileExistsError('No controller retry')
    start=time.monotonic();children=[];state=dict(completed=False,controller_pid=os.getpid(),protocol_SHA256=digest,release=record(LOCAL/'release.json'),builder=None,workers=[],all36_retained=True)
    def save():state['wall_s']=time.monotonic()-start;write(LOCAL/'execution.json',state)
    def stop(*_):raise InterruptedError('Owned fixed36 controller stop')
    previous={s:signal.signal(s,stop)for s in(signal.SIGTERM,signal.SIGINT)}
    def launch(role,slot=None):
        cpu=17 if role=='build'else slot;uuid=''if role=='build'else p['devices'][slot];command=['taskset','-c',str(cpu),sys.executable,'-u','-m','experiments.run_fixture_balanced_decision_screen_v01',role,'--sha',digest]
        if slot is not None:command+=['--slot',str(slot)]
        log=(LOCAL/(f'worker-{slot}.log'if role=='worker'else'builder.log')).open('xb');child=subprocess.Popen(command,cwd=ROOT,env={**os.environ,'CUDA_VISIBLE_DEVICES':uuid,'CUBLAS_WORKSPACE_CONFIG':':4096:8','OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1'},stdout=log,stderr=subprocess.STDOUT,start_new_session=True);log.close();children.append(child);return child,dict(pid=child.pid,command=command,slot=slot,physical_UUID=uuid,start_wall_s=time.monotonic()-start)
    def watch(active,cap):
        while any(c.poll()is None for c,_ in active):
            total=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
            for child,row in active:
                if child.poll()is None:
                    rss=old.parity.old.old.mechanical_screen.shared.owned_RSS(child);total+=rss;row['peak_RSS_bytes']=max(row.get('peak_RSS_bytes',0),rss)
                    if rss>=cap:raise MemoryError('Owned child role RSS cap')
                    if row['physical_UUID']:
                        vram=old.parity.owned_VRAM(child.pid,row['physical_UUID']);row['peak_owned_VRAM_bytes']=max(row.get('peak_owned_VRAM_bytes',0),vram)
                        if vram>=RESOURCES['owned_VRAM_bytes']:raise MemoryError('Owned physical GPU VRAM cap')
                    limit=RESOURCES['builder_wall_s']if row['slot']is None else RESOURCES['worker_wall_s']
                    if time.monotonic()-start-row['start_wall_s']>limit:raise TimeoutError('Prospective owned child wall cap')
                elif child.returncode!=0:raise RuntimeError('Owned stage technical failure; no implicit retries')
            state['peak_aggregate_RSS_bytes']=max(state.get('peak_aggregate_RSS_bytes',0),total)
            if total>=RESOURCES['aggregate_RSS_bytes']:raise MemoryError('Owned aggregate28GiB cap')
            if time.monotonic()-start>RESOURCES['global_wall_s']:raise TimeoutError('Fixed36 global7200s cap')
            guard('controller');save();time.sleep(.5)
        for child,row in active:finalize_child(child,row)
    try:
        child,row=launch('build');state['builder']=row;save();watch([(child,row)],RESOURCES['builder_RSS_bytes']);row['result']=record(LOCAL/'builder-result.json');builder_gate(p,digest)
        active=[]
        for slot in range(3):
            child,row=launch('worker',slot);active.append((child,row));state['workers'].append(row);save()
        watch(active,RESOURCES['worker_RSS_bytes']);rows=[]
        for child,row in active:
            path=LOCAL/f"worker-{row['slot']}/result-LOCAL.json";v=read(path);row['result']=record(path)
            if v.get('completed')is not True or v['pid']!=row['pid']or v['slot']!=row['slot']or v['physical_UUID']!=row['physical_UUID']or v['protocol_SHA256']!=digest or v['release']!=record(LOCAL/'release.json')or[v['index']for v in v['cases']]!=p['split'][row['slot']]:raise ValueError('Exact complete launched worker context/worklist')
            for case in v['cases']:
                if record(ROOT/case['record']['path'])!=case['record']:raise ValueError('Bound case result bytes')
                rows.append(read(ROOT/case['record']['path']))
        state['adequacy']=design.adequacy(rows);state.update(completed=True,case_records=[v for row in active for v in read(ROOT/row[1]['result']['path'])['cases']])
    except BaseException as error:state['failure']=dict(type=type(error).__name__,message=str(error))
    finally:
        for child in children:
            code=old.parity.old.old.mechanical_screen.shared.old.cost.cleanup(child)
            for row in([state['builder']]if state['builder']else[])+state['workers']:
                if row['pid']==child.pid:row.update(cleanup_exit_code=code,reaped=True,exit_code=child.returncode)
        for s,h in previous.items():signal.signal(s,h)
        state['terminal']=True;save()
    return state
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['prepare','build','worker','run','guard']);ap.add_argument('--sha');ap.add_argument('--slot',type=int);ap.add_argument('--role',choices=['builder','controller','worker']);a=ap.parse_args()
    if a.action=='guard':guard(a.role,a.slot);print('Pure role/config/resource guard; no model/data/GPU')
    else:
        result=prepare()if a.action=='prepare'else(build(a.sha)if a.action=='build'else(worker(a.slot,a.sha)if a.action=='worker'else run(a.sha)));print(json.dumps(result))
        if a.action=='run'and not result['completed']:raise SystemExit(1)
