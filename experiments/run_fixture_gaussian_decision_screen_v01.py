"""Fixed36 noiseless Gaussian-load LINEAR development action adequacy.

All144 category references and1728 action/bank/outcome jobs remain in the
prospective queue. No fresh cohort/noise, nonlinear or policy-advantage claim.
"""
import argparse,ast,gc,hashlib,json,os,resource,shutil,signal,subprocess,sys,time,zipfile
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from experiments import campaign_fixture_gaussian_decision_screen_v01 as design
from experiments import run_fixture_gaussian_linear_screen_v01 as linear
from experiments import audit_fixture_gaussian_linear_screen_v01 as linear_audit
from experiments import run_layout_grouped_allactions_v01 as old
from experiments import audit_layout_grouped_allactions_v01 as oldaudit
from experiments.campaign_surface_p2_inference_v01 import LabelledModal
from experiments.campaign_surface_conditional_amplitude_v01 import RatioModal,ExactRatioMemo
from experiments.campaign_surface_conditional_amplitude_v02 import ConditionalAmplitudeReferenceV02,AmplitudePWLPlan
from experiments.campaign_surface_gpu_modal_v01 import Float64GPUModal
ROOT=old.ROOT;LOCAL=ROOT/'runs/operator-dev/fixture-Gaussian-decision-screen-v01'
SOURCE='experiments/run_fixture_gaussian_decision_screen_v01.py';TEST='tests/test_run_fixture_gaussian_decision_screen_v01.py'
sha,record,read,write=old.sha,old.record,old.read,old.write
RESOURCES=dict(builder_CPU=17,controller_CPU=18,worker_CPUs=[0,1,2],threads=1,builder_wall_s=900,worker_wall_s=6600,global_wall_s=7200,builder_RSS_bytes=8*1024**3,worker_RSS_bytes=8*1024**3,controller_RSS_bytes=4*1024**3,aggregate_RSS_bytes=28*1024**3,torch_allocated_bytes=512*1024**2,owned_VRAM_bytes=2*1024**3,RAM_reserve_bytes=30*1024**3,disk_reserve_bytes=30*1024**3,maximum_writes_bytes=512*1024**2,memo_bytes_per_category=64*1024**2)
LINEAR_PROTOCOL='b25a555a08568a97733ffd92378cca142f022ec766b1f515394aec5acf6b04c0'
LINEAR_RESULT_SHA='347a798cd6c8c08698a9e60ebb1c6051ba8966ff7f5e258611d1f141e4d79dd0'
LINEAR_REVIEW_SHA='fdf2ec441664804a746875b2b30b6b2e527250e521c76b0e4d30dfd0f6722b04'
REFERENCE_PLAN=dict(inner=[128,256,512],outer=[128,256,512],moment_chunk=1024)
DECISION_LEVELS=((512,8),(512,16),(512,32),(512,64),(256,64))
def sources():
    todo=[SOURCE,TEST,'experiments/campaign_fixture_gaussian_decision_screen_v01.py','tests/test_fixture_gaussian_decision_screen_v01.py',oldaudit.SOURCE,oldaudit.TEST,'experiments/audit_fixture_gaussian_linear_screen_v01.py','tests/test_audit_fixture_gaussian_linear_screen_v01.py',*old.sources(),*linear.sources()];seen=set()
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
    return dict(linear_protocol=record(linear.LOCAL/'protocol.json'),linear_release=record(linear.LOCAL/'release.json'),linear_execution=record(linear.LOCAL/'execution.json'),linear_result=record(linear.LOCAL/'result-LOCAL.json'),linear_review=record(linear.LOCAL/'independent-review.json'),linear_raw=r['raw'],old_all48_protocol=record(old.LOCAL/'protocol.json'),old_all48_execution=record(old.LOCAL/'execution.json'),old_all48_review=record(old.LOCAL/'independent-terminal-review.json'),prefix_rows=metadata)
def prepare():
    guard('controller');origin=origin_gate()
    if(LOCAL/'protocol.json').exists():raise FileExistsError('No protocol refreeze')
    p=dict(schema='fixture-Gaussian-decision-screen-v01',source_SHA256=sources(),origin=origin,resources=RESOURCES,worklist=design.worklist(),devices=list(old.parity.GPU.ALLOWED_UUIDS),split=[[i for i in range(36)if i%3==k]for k in range(3)],reference_plan=REFERENCE_PLAN,decision_levels=[list(x)for x in DECISION_LEVELS],action_worklist=old.worklist(),gates=dict(outcome_relative=1e-4,latent_relative=1e-3),adequacy_gates=design.ADEQUACY,first_fixed_case_checkpoint=0,no_RNG=True,no_NH_PDE=True,scope='All36 fixed noiseless matching-LINEAR development; no calibrated/NH/active-advantage admission; numerical indicators are not rigorous bounds.')
    write(LOCAL/'protocol.json',p);return record(LOCAL/'protocol.json')
def validate(digest):
    p=read(LOCAL/'protocol.json');r=read(LOCAL/'release.json')
    if sha(LOCAL/'protocol.json')!=digest or p['source_SHA256']!=sources()or p['resources']!=RESOURCES or p['worklist']!=design.worklist()or p['split']!=[[i for i in range(36)if i%3==k]for k in range(3)]or p['devices']!=list(old.parity.GPU.ALLOWED_UUIDS)or p['reference_plan']!=REFERENCE_PLAN or p['decision_levels']!=[list(x)for x in DECISION_LEVELS]or p['action_worklist']!=old.worklist()or p['gates']!=dict(outcome_relative=1e-4,latent_relative=1e-3)or p['adequacy_gates']!=design.ADEQUACY:raise ValueError('Exact frozen development sources/config/worklist')
    verify_records(p['origin'])
    if r.get('approved')is not True or r.get('protocol_SHA256')!=digest or r.get('source_SHA256')!=p['source_SHA256']or r.get('origin')!=p['origin']:raise ValueError('Root release BEFORE prefixes/models/GPU')
    for key,pids in(('linear_execution',('controller_pid','worker_pid')),('old_all48_execution',('controller_pid',))):
        e=read(ROOT/p['origin'][key]['path'])
        for k in pids:absent(e[k])
        if key=='old_all48_execution':
            for row in e['devices']:absent(row['pid'])
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
def build(digest):
    p=validate(digest);guard('builder');directory=LOCAL/'models';directory.mkdir(parents=True,exist_ok=True)
    if(LOCAL/'builder-result.json').exists()or any(directory.iterdir()):raise FileExistsError('No model-build retry/partial reuse')
    start=time.monotonic();out=dict(completed=False,pid=os.getpid(),protocol_SHA256=digest,release=record(LOCAL/'release.json'),models=[],prefixes=[])
    try:
        # Copy only allowed prefix values; no future scalar is materialized.
        for row in p['origin']['prefix_rows']:
            observation=prefix_row(ROOT/p['origin']['linear_raw']['path'],row['raw_row']);path=directory/f"prefix-{row['case_index']:02d}-LOCAL.npz";np.savez_compressed(path,observed_prefix=observation);out['prefixes'].append(dict(index=row['case_index'],raw_row=row['raw_row'],record=record(path)))
        labels=tuple(s['id']for s in linear.mechanics.bank.measurement_specs());prefix_labels=labels[:4]
        for layout in range(4):
            begin=time.monotonic();f=linear.mechanics.geometry.make_forward(6,layout_id(layout));loads,_=linear.mechanics.geometry.load_catalog(6);model=linear.mechanics.make_measurement_modal(f,loads);maximum=0.
            for case in p['worklist']:
                if case['layout']!=layout:continue
                unit=np.log(np.array([case['E_Pa']/2000,case['K_N_m']/5]))/np.log([4,6]);observed=prefix_row(ROOT/p['origin']['linear_raw']['path'],p['origin']['prefix_rows'][case['index']]['raw_row']);maximum=max(maximum,float(np.max(abs(model.select(prefix_labels)(unit[None])[0,:,0]-observed))))
            if maximum>2e-12:raise ValueError('Rebuilt allmode n6 prefix/direct archive parity2pm')
            fp=model_fingerprint(model.modal.eigenvalues,model.modal.coefficients,labels);path=directory/f'layout-{layout}-LOCAL.npz';np.savez_compressed(path,eigenvalues=model.modal.eigenvalues,coefficients=model.modal.coefficients,labels=np.array(labels));out['models'].append(dict(layout=layout,record=record(path),model_fingerprint=fp,all_modes=len(model.modal.eigenvalues),output_shape=[1,60],maximum_prefix_parity_m=maximum,construction_s=time.monotonic()-begin,all_mode_diagnostics=model.modal.diagnostics));del f,loads,model;gc.collect();guard('builder')
        out['completed']=True
    except BaseException as error:out['failure']=dict(type=type(error).__name__,message=str(error));raise
    finally:out.update(wall_s=time.monotonic()-start,peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024);write(LOCAL/'builder-result.json',out)
    return out

def builder_gate(p,digest):
    result=read(LOCAL/'builder-result.json')
    if not result.get('completed')or result['protocol_SHA256']!=digest or result['release']!=record(LOCAL/'release.json')or len(result['models'])!=4 or[m['layout']for m in result['models']]!=list(range(4))or len(result['prefixes'])!=36 or[x['index']for x in result['prefixes']]!=list(range(36)):raise ValueError('Complete exact4models/36prefixes builder')
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

def policy(observation,callbacks,names,directory,checkpoint):
    """No generating layout/E/K, candidate response or held truth input."""
    observation=design.prefix_observation(observation);refs=[];finals=[];category=[]
    for k,(cpu,memo)in enumerate(callbacks):
        ref=ConditionalAmplitudeReferenceV02(cpu,list(zip(names['prefix'],observation[:,None])),names['target'],log_ratio_predict=memo,retain_hypotheses=False);result=ref.qualify(plan=AmplitudePWLPlan(moment_chunk=1024));check=old.parity.old.reference_check(result);path=directory/f'category-{k}-LOCAL.json';write(path,dict(qualified=result['qualified'],diagnostics=result['diagnostics'],strict=check));category.append(dict(layout=k,qualified=check['passed'],record=record(path)));refs.append(ref);finals.append(result['local_final']);checkpoint()
    out=dict(categories=category,decision_levels=[],jobs=[],qualified=False)
    if not all(v['qualified']for v in category):return out
    screening={}
    for outer,amp in DECISION_LEVELS:
        bank,metadata=old.parity.CPU.decision_bank(refs,names['candidate'],names['target'],outer=outer,amplitude_order=amp);check=old.parity.CPU.reference_moment_checks(metadata,finals);moments=bank.prior_moments();fraction=moments['between_layout_variance']/moments['total_variance'];screening[outer,amp]=dict(prior_variance=moments['total_variance'],between_layout_fraction=fraction,maximum_layout_probability=max(metadata['layout_probability']),layout_TV=check['layout_TV']);path=directory/f'decision-{outer}-{amp}-LOCAL.json';write(path,dict(checks=check,metadata={k:v for k,v in metadata.items()if k!='category_moments'},posterior_screen=screening[outer,amp]));out['decision_levels'].append(dict(outer=outer,amplitude=amp,checks=check,record=record(path)));checkpoint()
        if(outer,amp)in old.VARIANTS:
            # Keep fixed candidates even when a bank witness fails; no winner admitted.
            engine=old.parity.GPU.Float64GroupedGPU(bank,physical_UUID=os.environ['CUDA_VISIBLE_DEVICES'],outcome_chunk=8,allocated_cap_bytes=RESOURCES['torch_allocated_bytes'])
            for action in range(8):
                for order in(8,16):
                    begin=time.monotonic();value=engine.expected_risk(action,linear.mechanics.bank.SIGMA_M,order=order);engine.torch.cuda.synchronize();out['jobs'].append(dict(job=dict(outer=outer,amplitude=amp,action=action,order=order),result=value,wall_s=time.monotonic()-begin));checkpoint()
            del engine
        del bank;gc.collect()
    ranking=old.classify(out['jobs'],dict(outcome_relative=1e-4,latent_relative=1e-3));bank_ok=all(v['checks']['passed']for v in out['decision_levels']if(v['outer'],v['amplitude'])in old.VARIANTS)
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
            try:row.update(policy(observation,callbacks,names,directory,checkpoint));row['status']='qualified'if row['qualified']else'unresolved'
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
        cpu=17 if role=='build'else slot;uuid=''if role=='build'else p['devices'][slot];command=['taskset','-c',str(cpu),sys.executable,'-u','-m','experiments.run_fixture_gaussian_decision_screen_v01',role,'--sha',digest]
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
