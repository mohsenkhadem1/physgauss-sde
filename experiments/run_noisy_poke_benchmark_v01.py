"""SOURCE-only eight-object, four-noise-realization LINEAR-model stress on NH truth.

All32prefix choices seal before SEPARATE evaluation opens any candidate/held
truth. Every joint posterior uses prefix+ONE resetpoke, not accumulated actions.
No solve or noise draw executes on import/prepare. This is fixed-grid/noise
stress, not population calibration or generalized neural prediction.
"""
import argparse,ast,gc,hashlib,json,os,resource,shutil,signal,subprocess,sys,time,zipfile
from pathlib import Path
import numpy as np
from scipy.special import logsumexp
from experiments import campaign_noisy_poke_benchmark_v01 as design
from experiments import run_fixture_balanced_decision_screen_v01 as base
from experiments import run_fixture_balanced_nh_catalog_v01 as physics_stage
ROOT=base.ROOT;LOCAL=ROOT/'runs/operator-dev/noisy-poke-benchmark-v01'
SOURCE='experiments/run_noisy_poke_benchmark_v01.py';TEST='tests/test_run_noisy_poke_benchmark_v01.py'
AUDIT_SOURCE='experiments/audit_noisy_poke_benchmark_v01.py';AUDIT_TEST='tests/test_audit_noisy_poke_benchmark_v01.py'
sha,read,write,record=base.sha,base.read,base.write,base.record
RESOURCES=dict(controller_CPU=18,input_CPU=17,worker_CPUs=[0,1,2],threads=1,input_wall_s=300,choice_wall_s=6600,evaluation_wall_s=10800,global_wall_s=14400,worker_RSS_bytes=8*1024**3,controller_RSS_bytes=4*1024**3,aggregate_RSS_bytes=28*1024**3,torch_allocated_bytes=512*1024**2,owned_VRAM_bytes=2*1024**3,RAM_reserve_bytes=30*1024**3,disk_reserve_bytes=30*1024**3,maximum_writes_bytes=1024**3)
DESIGN_PATH=ROOT/'runs/operator-dev/noisy-probing-design-v01/design.json'
DESIGN_SHA256='9579dd5c78045a2839262005ac6d61bd311b872681e9a584c5b6477acc2d91b8'
CONTEXT_PLAN=dict(inner=[128,256,512],outer=[128,256,512],moment_chunk=1024)

def sources():
    todo=[SOURCE,TEST,design.SOURCE,design.TEST,AUDIT_SOURCE,AUDIT_TEST,*base.sources(),*physics_stage.sources()];seen=set()
    while todo:
        name=todo.pop()
        if name in seen:continue
        seen.add(name)
        for node in ast.walk(ast.parse((ROOT/name).read_text())):
            modules=[]
            if isinstance(node,ast.Import):modules=[v.name for v in node.names]
            elif isinstance(node,ast.ImportFrom)and node.level==0:modules=[node.module or '']+[(node.module+'.'+v.name)for v in node.names if node.module]
            for module in modules:
                p=module.replace('.','/')+'.py'
                if(ROOT/p).is_file()and p not in seen:todo.append(p)
    return {p:sha(ROOT/p)for p in sorted(seen)}
def verify(value):base.verify_records(value)
def seed_registration_check():
    hits=[]
    for name in('protocol.json','recipe.json'):
        for path in(ROOT/'runs/operator-dev').rglob(name):
            if LOCAL in path.parents:continue
            text=path.read_text()
            if any(str(seed)in text for seed in(design.PREFIX_SEED,design.POKE_SEED)):hits.append(str(path.relative_to(ROOT)))
    if hits:raise ValueError('Noise seed already registered: '+','.join(hits))
    return dict(seeds=[design.PREFIX_SEED,design.POKE_SEED],existing_protocol_or_recipe_matches=0,no_noise_generated=True)

def design_gate():
    if sha(DESIGN_PATH)!=DESIGN_SHA256:raise ValueError('Exact prospective design frozen before physics/noise')
    d=read(DESIGN_PATH);noise=d['noise']
    if d['objects']!=design.objects()or d['trials']!=design.trials()or d['noise_replicates']!=4 or noise['prefix_seed']!=design.PREFIX_SEED or noise['poke_seed']!=design.POKE_SEED or noise['prefix_shape']!=[4,8,4]or noise['poke_shape']!=[4,8,8]or noise['sigma_m']!=design.physics.SIGMA_M or noise['draws_generated']is not False or noise['numpy_version']!=np.__version__ or d['fixed_baseline']['action']!=5:raise ValueError('Exact frozen interior8/four-noise/fixedaction5 design')
    return record(DESIGN_PATH)

def origin_gate():
    prospective=design_gate();truth=physics_stage.completed_catalog_gate();basep=read(base.LOCAL/'protocol.json');base.validate(sha(base.LOCAL/'protocol.json'));e=read(base.LOCAL/'execution.json');review=read(base.LOCAL/'independent-terminal-review.json');closure=read(base.LOCAL/'root-closure.json')
    if not(e.get('completed')is True and review['summary'].get('integrity_passed')is True and review['summary']['adequacy'].get('adequacy_qualified')is True and closure.get('root_reaped')is True and closure.get('root_exit_code')==0 and closure['execution_SHA256']==sha(base.LOCAL/'execution.json')):raise ValueError('Closed positive balanced36 development for fixedbaseline/model archives')
    for pid in closure['owned_pids']:base.absent(pid)
    verify(e['case_records']);development=[read(ROOT/v['record']['path'])for v in e['case_records']];fixed=design.fixed_action(development);builder=read(base.LOCAL/'builder-result.json');verify(builder)
    groups=truth['groups']
    if len(groups)!=8 or[v['index']for v in groups]!=list(range(8)):raise ValueError('All8 fixed physical truth groups')
    for expected,group in zip(design.objects(),groups):
        if any(group['group'][k]!=expected[k]for k in('index','layout','E_Pa','K_N_m')):raise ValueError('Exact interior8 reference membership')
    if fixed['action']!=5:raise ValueError('Fixed action5 selected prospectively from all36 old expected risks')
    return dict(prospective_design=prospective,truth=truth,development_protocol=record(base.LOCAL/'protocol.json'),development_execution=record(base.LOCAL/'execution.json'),development_review=record(base.LOCAL/'independent-terminal-review.json'),development_closure=record(base.LOCAL/'root-closure.json'),model_builder=record(base.LOCAL/'builder-result.json'),models=builder['models'],fixed_baseline=fixed)

def prepare():
    guard('controller');seeds=seed_registration_check();origin=origin_gate()
    if(LOCAL/'protocol.json').exists():raise FileExistsError('No benchmark refreeze')
    p=dict(schema='noisy-poke-benchmark-v01',source_SHA256=sources(),origin=origin,objects=design.objects(),trials=design.trials(),resources=RESOURCES,seeds=dict(prefix=design.PREFIX_SEED,poke=design.POKE_SEED),seed_check=seeds,noise_shapes=dict(prefix=[4,8,4],poke=[4,8,8]),sigma_m=design.physics.SIGMA_M,reference_plan=CONTEXT_PLAN,decision_levels=[list(x)for x in base.DECISION_LEVELS],utility_worklist=base.old.worklist(),devices=list(base.old.parity.GPU.ALLOWED_UUIDS),fixed_action=origin['fixed_baseline']['action'],choice_split=[[i for i in range(32)if i%3==k]for k in range(3)],evaluation_checkpoint=[0],evaluation_remaining=[[i for i in range(1,32)if i%3==k]for k in range(3)],field_interval='Explicit posterior mean±2SD MOMENTbands; no nominal coverage level',material_interval='Continuous full category-mixture central90%; descriptive fixedgrid inclusion, not SBC/calibration',primary='Paired clean48targetMSE; equal8objects after fourwithinobjectnoise means',random='Exactuniform average8onepoke counterfactual losses for this frozen noise realization; no noiseintegrated/population claim',all32_retained=True,no_PDE=True,known_linear_model_bias_stress=True)
    write(LOCAL/'protocol.json',p);return record(LOCAL/'protocol.json')
def validate(digest):
    p=read(LOCAL/'protocol.json');r=read(LOCAL/'release.json')
    if sha(LOCAL/'protocol.json')!=digest or p['source_SHA256']!=sources()or p['resources']!=RESOURCES or p['objects']!=design.objects()or p['trials']!=design.trials()or p['seeds']!=dict(prefix=design.PREFIX_SEED,poke=design.POKE_SEED)or p['reference_plan']!=CONTEXT_PLAN or p['decision_levels']!=[list(x)for x in base.DECISION_LEVELS]or p['utility_worklist']!=base.old.worklist()or p['devices']!=list(base.old.parity.GPU.ALLOWED_UUIDS)or p['choice_split']!=[[i for i in range(32)if i%3==k]for k in range(3)]or p['evaluation_checkpoint']!=[0]or p['evaluation_remaining']!=[[i for i in range(1,32)if i%3==k]for k in range(3)]:raise ValueError('Exact frozen benchmark source/config')
    if p['origin']['prospective_design']!=design_gate()or p['fixed_action']!=5 or p['noise_shapes']!=dict(prefix=[4,8,4],poke=[4,8,8])or p['sigma_m']!=design.physics.SIGMA_M:raise ValueError('Exact prospective design/noise/fixedbaseline binding')
    verify(p['origin'])
    if r.get('approved')is not True or r.get('protocol_SHA256')!=digest or r.get('source_SHA256')!=p['source_SHA256']or r.get('origin')!=p['origin']:raise ValueError('Root release BEFORE noise/truth/observations')
    return p

def available_memory():
    return int(next(line.split()[1]for line in Path('/proc/meminfo').read_text().splitlines()if line.startswith('MemAvailable:')))*1024

def guard(role,slot=None):
    cpu=RESOURCES['worker_CPUs'][slot]if role=='worker'else RESOURCES[role+'_CPU']
    if set(os.sched_getaffinity(0))!={cpu}or any(os.environ.get(k)!='1'for k in('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS')):raise ValueError('Exact benchmark own affinity/one thread')
    if role=='worker':base.old.parity.GPU.pin_config(base.old.parity.GPU.ALLOWED_UUIDS[slot])
    elif os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('Controller/input no CUDA')
    limit=RESOURCES['controller_RSS_bytes']if role=='controller'else RESOURCES['worker_RSS_bytes']
    if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024>=limit:raise MemoryError('Own benchmark RSS cap')
    available=available_memory()
    if available<RESOURCES['RAM_reserve_bytes']or shutil.disk_usage(ROOT).free<RESOURCES['disk_reserve_bytes']+RESOURCES['maximum_writes_bytes']:raise MemoryError('Own30GiB RAM/disk reserve')
    if LOCAL.exists()and sum(p.stat().st_size for p in LOCAL.rglob('*')if p.is_file())>RESOURCES['maximum_writes_bytes']:raise MemoryError('Own1GiB writes cap')

def capture(p,object_index,label):
    rows=p['origin']['truth']['groups'][object_index]['rows'];matches=[v['capture']for v in rows if v['n']==12 and v['label']==label]
    if len(matches)!=1:raise ValueError('One exact n12capture per group/load')
    return matches[0]
def prefix_capture_only(path):
    specs=design.physics.measurement_specs()[:4];points=design.physics.measurement_points()
    if any(v['role']!='prefix'or not np.array_equal(v['point_m'],points[i])or not np.array_equal(v['projection'],[0.,0.,1.])for i,v in enumerate(specs)):raise ValueError('Public first4query-Z prefix mapping')
    with zipfile.ZipFile(path)as z,z.open('query_displacement_m.npy')as f:
        version=np.lib.format.read_magic(f)
        if version==(1,0):shape,fortran,dtype=np.lib.format.read_array_header_1_0(f)
        elif version==(2,0):shape,fortran,dtype=np.lib.format.read_array_header_2_0(f)
        else:raise ValueError('Declared NPY version')
        if tuple(shape)!=(16,3)or fortran or dtype!=np.dtype('float64'):raise ValueError('Exact16×3query snapshot')
        start=f.tell();values=[]
        for i in range(4):f.seek(start+(i*3+2)*8);values.append(np.frombuffer(f.read(8),dtype=np.float64)[0])
    return design.prefix_only(values)

def inputs(digest):
    p=validate(digest);guard('input');start=time.monotonic();directory=LOCAL/'inputs';directory.mkdir(parents=True,exist_ok=True)
    if any(directory.iterdir())or(LOCAL/'inputs.json').exists():raise FileExistsError('No noise/observation redraw')
    noise=design.noise_streams();prefix_label=design.physics.load_specs()[0]['id'];records=[]
    for trial in p['trials']:
        i,r=trial['object_index'],trial['replicate'];c=capture(p,i,prefix_label);truth=prefix_capture_only(ROOT/c['path']);observed=truth+noise['prefix'][r,i];path=directory/f'prefix-{trial["index"]:02d}-LOCAL.npz';np.savez_compressed(path,observed_prefix=observed,prefix_noise=noise['prefix'][r,i]);records.append(dict(index=trial['index'],record=record(path),capture=c,observation_fingerprint=design.fingerprint(observed)))
        guard('input')
        if time.monotonic()-start>RESOURCES['input_wall_s']:raise TimeoutError('Frozen300s observation/noise input cap')
    path=directory/'poke-noise-LOCAL.npz';np.savez_compressed(path,poke_noise=noise['poke']);out=dict(completed=True,protocol_SHA256=digest,release=record(LOCAL/'release.json'),prefixes=records,poke_noise=record(path),seeds=p['seeds'],shapes=p['noise_shapes'],candidate_or_held_truth_read=False,wall_s=time.monotonic()-start);write(LOCAL/'inputs.json',out);return record(LOCAL/'inputs.json')

def inputs_gate(digest):
    m=read(LOCAL/'inputs.json');p=read(LOCAL/'protocol.json')
    if m.get('completed')is not True or m['protocol_SHA256']!=digest or m['release']!=record(LOCAL/'release.json')or len(m['prefixes'])!=32 or[v['index']for v in m['prefixes']]!=list(range(32))or m['seeds']!=p['seeds']or m['shapes']!=p['noise_shapes']or m.get('candidate_or_held_truth_read')is not False:raise ValueError('All32 original-noise inputs committed')
    verify(m)
    label=design.physics.load_specs()[0]['id']
    for row in m['prefixes']:
        if row['capture']!=capture(p,p['trials'][row['index']]['object_index'],label):raise ValueError('Exact initialprefix source capture pertrial')
    return m

def probabilities(finals):
    logs=np.array([r['log_evidence']for r in finals]);v=np.exp(logs-logsumexp(logs));return v/v.sum()
def mixture_moments(finals,p):
    means=np.array([r['predictive_mean']for r in finals]);sd=np.array([r['predictive_SD']for r in finals]);anchor=means[0];delta=means-anchor;mean=anchor+p@delta;variance=p@(sd**2+(delta-p@delta)**2);return mean,np.sqrt(variance)
def infer(observations,callbacks,names,directory,checkpoint):
    refs=[];finals=[];out=dict(categories=[],decision_levels=[],qualified=False)
    for k,(cpu,memo)in enumerate(callbacks):
        context=dict(observations_fingerprint=observations_fingerprint(observations),model_fingerprint=memo.model_fingerprint,reference_plan=CONTEXT_PLAN,target_labels=list(names['target']),sigma_m=design.physics.SIGMA_M)
        ref=base.ConditionalAmplitudeReferenceV02(cpu,observations,names['target'],log_ratio_predict=memo,retain_hypotheses=False);result=ref.qualify(plan=base.AmplitudePWLPlan(moment_chunk=1024));check=base.old.parity.old.reference_check(result);path=directory/f'category-{k}-LOCAL.json';write(path,dict(qualified=result['qualified'],diagnostics=result['diagnostics'],strict=check,reference_context=context));out['categories'].append(dict(layout=k,qualified=check['passed'],record=record(path)));refs.append(ref);finals.append(result['local_final']);checkpoint()
    if not all(v['qualified']for v in out['categories']):return out,None,None,None
    for outer,amp in base.DECISION_LEVELS:
        bank,metadata=base.old.parity.CPU.decision_bank(refs,names['candidate'],names['target'],outer=outer,amplitude_order=amp);check=base.old.parity.CPU.reference_moment_checks(metadata,finals);path=directory/f'decision-{outer}-{amp}-LOCAL.json';write(path,dict(checks=check,metadata={k:v for k,v in metadata.items()if k!='category_moments'}));out['decision_levels'].append(dict(outer=outer,amplitude=amp,checks=check,record=record(path)));checkpoint();del bank
    if not all(v['checks']['passed']for v in out['decision_levels']if(v['outer'],v['amplitude'])in base.old.VARIANTS):return out,None,None,None
    p=probabilities(finals);mean,SD=mixture_moments(finals,p);unitmean=p@np.array([f['unit_mean']for f in finals]);intervals=[]
    for outer,inner in((512,512),(512,256),(256,512)):
        weights=probabilities([dict(log_evidence=ref._state(outer,inner)['log_evidence'])for ref in refs]);intervals.append(design.material_intervals(refs,weights,outer=outer,inner=inner))
    interval=intervals[0];cdf_error=max(float(np.max(abs(v['unit_CDF']-interval['unit_CDF'])))for v in intervals[1:]);q_error=max(float(np.max(abs(v['unit_quantiles']-interval['unit_quantiles'])))for v in intervals[1:]);out['mixture_interval_checks']=dict(passed=bool(cdf_error<=.005 and q_error<=.002),maximum_CDF_difference=cdf_error,maximum_unit_quantile_difference=q_error,CDF_gate=.005,unit_quantile_gate=.002,levels=[[512,512],[512,256],[256,512]],weights_from_continuous_category_logZ=True)
    if not out['mixture_interval_checks']['passed']:return out,None,None,None
    snapshot=directory/'posterior-LOCAL.npz';np.savez_compressed(snapshot,mean_m=mean,SD_m=SD,unit_mean=unitmean,layout_probability=p,material_quantiles=interval['physical_quantiles'],material_unit_CDF=interval['unit_CDF'],material_unit_quantiles=interval['unit_quantiles']);out.update(qualified=True,snapshot=record(snapshot),material_interval_method=interval['method']);return out,refs,finals,names

def utility(refs,names,checkpoint):
    rows=[]
    for outer,amp in base.old.VARIANTS:
        bank,_=base.old.parity.CPU.decision_bank(refs,names['candidate'],names['target'],outer=outer,amplitude_order=amp);engine=base.old.parity.GPU.Float64GroupedGPU(bank,physical_UUID=os.environ['CUDA_VISIBLE_DEVICES'],outcome_chunk=8,allocated_cap_bytes=RESOURCES['torch_allocated_bytes'])
        for action in range(8):
            for order in(8,16):rows.append(dict(job=dict(outer=outer,amplitude=amp,action=action,order=order),result=engine.expected_risk(action,design.physics.SIGMA_M,order=order)));checkpoint()
        del engine,bank;gc.collect()
    return rows,base.old.classify(rows,dict(outcome_relative=1e-4,latent_relative=1e-3))

def observations_fingerprint(observations):
    h=hashlib.sha256()
    for label,value in observations:h.update(label.encode());h.update(np.ascontiguousarray(value,dtype=np.float64).tobytes())
    return h.hexdigest()
def load_common(p,slot):
    models=base.load_models(dict(models=p['origin']['models']));callbacks=base.make_ratio_callbacks(models,p['devices'][slot])
    if len(callbacks)!=4 or any(memo.model_fingerprint!=p['origin']['models'][k]['model_fingerprint']for k,(cpu,memo)in enumerate(callbacks)):raise ValueError('Each exact category callback fingerprint anchored to frozen modalarchive')
    labels=models[0].labels;names=dict(prefix=labels[:4],candidate=labels[4:12],target=labels[12:]);return models,callbacks,names

def choose_worker(slot,digest):
    p=validate(digest);guard('worker',slot);m=inputs_gate(digest);models,callbacks,names=load_common(p,slot);state=dict(completed=False,pid=os.getpid(),slot=slot,phase='choices',physical_UUID=p['devices'][slot],protocol_SHA256=digest,release=record(LOCAL/'release.json'),input=record(LOCAL/'inputs.json'),trials=[]);start=time.monotonic();path=LOCAL/f'choices-worker-{slot}.json'
    if path.exists():raise FileExistsError('No selection worker retry')
    def checkpoint():guard('worker',slot);state.update(wall_s=time.monotonic()-start,peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024);write(path,state)
    try:
        for index in p['choice_split'][slot]:
            directory=LOCAL/f'trial-{index:02d}'/'prefix';directory.mkdir(parents=True,exist_ok=True)
            if any(directory.iterdir()):raise FileExistsError('No selection partial reuse')
            with np.load(ROOT/m['prefixes'][index]['record']['path'],allow_pickle=False)as z:observed=design.prefix_only(z['observed_prefix'])
            if design.fingerprint(observed)!=m['prefixes'][index]['observation_fingerprint']:raise ValueError('Exact noisy prefix fingerprint')
            observations=list(zip(names['prefix'],observed[:,None]));out,refs,finals,_=infer(observations,callbacks,names,directory,checkpoint);out.update(index=index,protocol_SHA256=digest,observations_fingerprint=observations_fingerprint(observations),prefix_input=m['prefixes'][index]['record'],candidate_or_held_truth_read=False)
            if out['qualified']:
                rows,ranking=utility(refs,names,checkpoint);out.update(utility_jobs=rows,ranking=ranking,qualified=bool(ranking['qualified']))
            target=directory/'result-LOCAL.json';write(target,out);choice=dict(index=index,protocol_SHA256=digest,release=record(LOCAL/'release.json'),prefix_result=record(target),prefix_input=m['prefixes'][index]['record'],fixed_action=p['fixed_action'],adaptive_action=out.get('ranking',{}).get('unique_best_action')if out['qualified']else None,numerical_ties=out.get('ranking',{}).get('numerical_ties'),qualified=bool(out['qualified']),tie_rule='Lowest catalog index among numerical ties, if all gates qualified',candidate_or_held_truth_read=False)
            if choice['qualified']and choice['adaptive_action']is None:choice['adaptive_action']=min(choice['numerical_ties'])
            seal=LOCAL/f'trial-{index:02d}'/'choice-seal.json';write(seal,choice);state['trials'].append(dict(index=index,seal=record(seal)));checkpoint();del refs,finals;gc.collect()
        state['completed']=True
    except BaseException as error:state['failure']=dict(type=type(error).__name__,message=str(error));raise
    finally:checkpoint()
    return state

def all_seals_gate(digest):
    completed_phase('choices',digest);p=read(LOCAL/'protocol.json');m=inputs_gate(digest);result=[]
    for slot in range(3):
        w=read(LOCAL/f'choices-worker-{slot}.json')
        if w.get('completed')is not True or w['protocol_SHA256']!=digest or w['release']!=record(LOCAL/'release.json')or w['slot']!=slot or w['physical_UUID']!=p['devices'][slot]or w['input']!=record(LOCAL/'inputs.json')or[v['index']for v in w['trials']]!=p['choice_split'][slot]:raise ValueError('All32 selection worklist sealed before evaluation')
        base.absent(w['pid']);verify(w['trials']);result.extend(w['trials'])
    result.sort(key=lambda v:v['index'])
    for row in result:
        index=row['index'];expected_seal=record(LOCAL/f'trial-{index:02d}'/'choice-seal.json')
        if row['seal']!=expected_seal:raise ValueError('Canonical pertrial seal path/hash')
        value=read(ROOT/row['seal']['path']);expected_result=record(LOCAL/f'trial-{index:02d}'/'prefix'/'result-LOCAL.json');expected_input=m['prefixes'][index]['record']
        if value['prefix_result']!=expected_result or value['prefix_input']!=expected_input:raise ValueError('Exact canonical prefix/input commitment pertrial')
        prefix=read(ROOT/expected_result['path']);verify(prefix)
        if value['index']!=index or value['protocol_SHA256']!=digest or value['release']!=record(LOCAL/'release.json')or value.get('candidate_or_held_truth_read')is not False or prefix['index']!=index or prefix['protocol_SHA256']!=digest or prefix['prefix_input']!=expected_input or prefix.get('candidate_or_held_truth_read')is not False:raise ValueError('Every exact causal prefix/choice commitment')
        ranking=prefix.get('ranking',{});qualified=prefix.get('qualified')is True
        if qualified and ranking.get('qualified')is not True:raise ValueError('Qualified selection requires qualified risk ranking')
        adaptive=ranking.get('unique_best_action')if qualified else None;ties=ranking.get('numerical_ties')
        if qualified and adaptive is None:
            if not isinstance(ties,list)or not ties or any(type(a)is not int or a not in range(8)for a in ties):raise ValueError('Declared nonempty candidate tie set')
            adaptive=min(ties)
        if value['qualified']!=qualified or value['fixed_action']!=p['fixed_action']or value['fixed_action']!=5 or value['adaptive_action']!=adaptive or value.get('numerical_ties')!=ties:raise ValueError('Choices exactly match sealed prefix-only numerical ranking')
    return result

def query_truth(p,object_index):
    """Evaluator-only; caller must establish all32 byte-bound choice seals first."""
    specs=design.physics.measurement_specs();points=design.physics.measurement_points();labels=[s['id']for s in design.physics.load_specs()];query={};linear_target={};capture_records=[]
    for label in labels:
        c=capture(p,object_index,label);capture_records.append(c)
        with np.load(ROOT/c['path'],allow_pickle=False)as z:
            query[label]=z['query_displacement_m'].copy()
            if label in labels[9:]:linear_target[label]=physics_stage.geometry.field_queries(12,z['linear_displacement_m'],points)
    value=[];warm=[]
    for spec in specs:
        i=next(i for i,v in enumerate(points)if np.array_equal(v,spec['point_m']));value.append(float(query[spec['load']][i]@np.asarray(spec['projection'])))
        if spec['role']=='target':warm.append(float(linear_target[spec['load']][i]@np.asarray(spec['projection'])))
    value=np.asarray(value);warm=np.asarray(warm)
    if value.shape!=(60,)or warm.shape!=(48,)or not np.isfinite(value).all()or not np.isfinite(warm).all():raise ValueError('Finite admitted n12 clean observations/heldtruth')
    return value,warm,capture_records

def snapshot_metrics(path,truth,parameters):
    with np.load(path,allow_pickle=False)as z:
        field=design.prediction_metrics(z['mean_m'],z['SD_m'],truth);material=design.material_metrics(dict(physical_quantiles=z['material_quantiles']),z['unit_mean'],parameters);rho=z['layout_probability'].copy()
    return dict(field=field,material=material,layout_probability=rho.tolist())

def evaluate_worker(slot,phase,digest):
    p=validate(digest);guard('worker',slot);seals=all_seals_gate(digest);m=inputs_gate(digest)
    # No candidate/held array, model or noise access above the complete seal barrier.
    models,callbacks,names=load_common(p,slot);indices=p['evaluation_checkpoint']if phase=='checkpoint'else p['evaluation_remaining'][slot];start=time.monotonic();state=dict(completed=False,pid=os.getpid(),slot=slot,phase=phase,physical_UUID=p['devices'][slot],protocol_SHA256=digest,release=record(LOCAL/'release.json'),all32_choice_seals=seals,trials=[]);path=LOCAL/f'evaluation-{phase}-worker-{slot}.json'
    if path.exists():raise FileExistsError('No evaluation worker retry')
    with np.load(ROOT/m['poke_noise']['path'],allow_pickle=False)as z:noise=z['poke_noise'].copy()
    if noise.shape!=(4,8,8)or not np.isfinite(noise).all():raise ValueError('Exact fourreplicate noise archive')
    def checkpoint():guard('worker',slot);state.update(wall_s=time.monotonic()-start,peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024);write(path,state)
    try:
        for index in indices:
            trial=p['trials'][index];i,r=trial['object_index'],trial['replicate'];directory=LOCAL/f'trial-{index:02d}'/'evaluation';directory.mkdir(parents=True,exist_ok=True)
            if any(directory.iterdir()):raise FileExistsError('No evaluation partial reuse')
            seal=read(ROOT/seals[index]['seal']['path']);prefix=read(ROOT/seal['prefix_result']['path']);values,warm,captures=query_truth(p,i);parameters=np.array([p['objects'][i]['E_Pa'],p['objects'][i]['K_N_m']]);unit=(np.log(parameters)-np.log([2000.,5.]))/np.log([4.,6.]);layout=next(k for k,v in enumerate(design.physics.LAYOUTS)if v['id']==p['objects'][i]['layout']);oracle=models[layout].select(names['target'])(unit[None])[0,:,0];truth=values[12:]
            truth_path=directory/'truth-LOCAL.npz';np.savez_compressed(truth_path,NH_target_m=truth,n12_linear_target_m=warm,n6_true_parameter_oracle_m=oracle,clean_candidate_m=values[4:12]);out=dict(index=index,object_index=i,replicate=r,protocol_SHA256=digest,choice_seal=seals[index]['seal'],prefix_result=seal['prefix_result'],truth=record(truth_path),source_captures=captures,posteriors=[],qualified=False,oracle_metrics=dict(n12_linear_vs_n12_NH_MSE_m2=float(np.mean((warm-truth)**2)),n6_modal_vs_n12_linear_MSE_m2=float(np.mean((oracle-warm)**2)),n6_modal_vs_n12_NH_MSE_m2=float(np.mean((oracle-truth)**2))))
            with np.load(ROOT/m['prefixes'][index]['record']['path'],allow_pickle=False)as z:observed=design.prefix_only(z['observed_prefix'])
            before=None
            if prefix.get('qualified')is True:before=snapshot_metrics(ROOT/prefix['snapshot']['path'],truth,parameters);out['before']=before
            after=[]
            for action in range(8):
                action_dir=directory/f'action-{action}';action_dir.mkdir();obs=list(zip(names['prefix'],observed[:,None]))+[(names['candidate'][action],np.array([values[4+action]+noise[r,i,action]]))];obs_path=action_dir/'observations-LOCAL.npz';np.savez_compressed(obs_path,labels=np.array([v[0]for v in obs]),values=np.array([v[1]for v in obs]));begin=time.monotonic();result,refs,finals,_=infer(obs,callbacks,names,action_dir,checkpoint);result.update(index=index,action=action,protocol_SHA256=digest,choice_seal=seals[index]['seal'],observations=record(obs_path),observations_fingerprint=observations_fingerprint(obs),prefix_plus_ONE_reset_poke=True,wall_s=time.monotonic()-begin);target=action_dir/'result-LOCAL.json';write(target,result);entry=dict(action=action,qualified=result['qualified'],record=record(target))
                if result['qualified']:entry['metrics']=snapshot_metrics(ROOT/result['snapshot']['path'],truth,parameters);after.append(entry['metrics']['field'])
                else:after.append(None)
                out['posteriors'].append(entry);del refs,finals;gc.collect();checkpoint()
            out['qualified']=bool(before is not None and seal['qualified']and all(v['qualified']for v in out['posteriors']))
            if out['qualified']:out['policies']=design.compare_policies(before['field'],after,int(seal['adaptive_action']),int(seal['fixed_action']));out['material_policies']=design.compare_material(before['material'],[v['metrics']['material']for v in out['posteriors']],int(seal['adaptive_action']),int(seal['fixed_action']))
            target=directory/'result-LOCAL.json';write(target,out);state['trials'].append(dict(index=index,qualified=out['qualified'],record=record(target)));checkpoint()
        state['completed']=True
    except BaseException as error:state['failure']=dict(type=type(error).__name__,message=str(error));raise
    finally:checkpoint()
    return state

def completed_phase(phase,digest):
    if phase not in('choices','checkpoint','remaining'):raise ValueError('Exact phase')
    p=read(LOCAL/'protocol.json');e=read(LOCAL/f'execution-{phase}.json');release=record(LOCAL/'release.json')
    if sha(LOCAL/'protocol.json')!=digest or e.get('completed')is not True or e.get('terminal')is not True or e['phase']!=phase or e['protocol_SHA256']!=digest or e['release']!=release or 'failure'in e:raise ValueError('Exact clean completed benchmark phase')
    expected_slots=[0]if phase=='checkpoint'else[0,1,2]
    if [row['slot']for row in e['workers']]!=expected_slots:raise ValueError('Every exact assigned phase worker')
    owned=[e['controller_pid']]
    for row in e['workers']:
        slot=row['slot'];owned.append(row['pid']);expected_path=LOCAL/(f'choices-worker-{slot}.json'if phase=='choices'else f'evaluation-{phase}-worker-{slot}.json')
        if row.get('exit_code')!=0 or row.get('cleanup_exit_code')!=0 or row.get('reaped')is not True or row['physical_UUID']!=p['devices'][slot]or row['result']!=record(expected_path):raise ValueError('Exact reaped phase worker records')
        w=read(expected_path);indices=p['choice_split'][slot]if phase=='choices'else(p['evaluation_checkpoint']if phase=='checkpoint'else p['evaluation_remaining'][slot])
        if w.get('completed')is not True or 'failure'in w or w['pid']!=row['pid']or w['slot']!=slot or w['phase']!=phase or w['protocol_SHA256']!=digest or w['release']!=release or w['physical_UUID']!=p['devices'][slot]or[v['index']for v in w['trials']]!=indices:raise ValueError('Exact closed worker context/worklist')
        verify(w['trials'])
    if len(set(owned))!=len(owned):raise ValueError('Distinct owned controller/worker identities')
    closure=read(LOCAL/f'root-closure-{phase}.json')
    if closure.get('root_reaped')is not True or closure.get('root_exit_code')!=0 or closure.get('protocol_SHA256')!=digest or closure.get('source_SHA256')!=p['source_SHA256']or closure.get('execution_SHA256')!=sha(LOCAL/f'execution-{phase}.json')or closure.get('owned_pids')!=owned:raise ValueError('External root-owned exact phase closure/source proof')
    for pid in owned:base.absent(pid)
    return e

def continuation_gate(digest):
    p=validate(digest);seals=all_seals_gate(digest);e=completed_phase('checkpoint',digest);w=read(ROOT/e['workers'][0]['result']['path']);trial=read(ROOT/w['trials'][0]['record']['path']);r=read(LOCAL/'continuation-release.json')
    if not(w['completed']and len(w['trials'])==1 and w['trials'][0]['index']==0 and trial['qualified']and len(trial['posteriors'])==8):raise ValueError('First fixed trial0 qualified cost checkpoint, all8 actions; no replacement case')
    worst=max(read(ROOT/v['record']['path'])['wall_s']for v in trial['posteriors']);projection=3*worst*8*11
    if projection>RESOURCES['evaluation_wall_s']:raise ValueError('Measured conservative remaining cost exceeds frozenworker cap')
    if not(r.get('approved')is True and r.get('protocol_SHA256')==digest and r.get('source_SHA256')==p['source_SHA256']and r.get('checkpoint')==record(LOCAL/'execution-checkpoint.json')and r.get('all32_choice_seals')==seals):raise ValueError('Root same32 continuation release after retainedfirsttrial cost')
    return p

def run(phase,digest):
    p=validate(digest);guard('controller')
    if phase not in('choices','checkpoint','remaining'):raise ValueError('Fixed threephase worklist')
    if phase=='choices':inputs_gate(digest)
    else:all_seals_gate(digest)
    if phase=='remaining':continuation_gate(digest)
    worker_count=1 if phase=='checkpoint'else 3
    if available_memory()<worker_count*RESOURCES['worker_RSS_bytes']+RESOURCES['controller_RSS_bytes']+RESOURCES['RAM_reserve_bytes']:raise MemoryError('Initial owned worker/controller budgets plus30GiBreserve')
    execution=LOCAL/f'execution-{phase}.json'
    if execution.exists():raise FileExistsError('No phase retry/overwrite')
    prior=sum(read(LOCAL/f'execution-{v}.json')['wall_s']for v in('choices','checkpoint')if v!=phase and(LOCAL/f'execution-{v}.json').exists());start=time.monotonic();children=[];state=dict(completed=False,controller_pid=os.getpid(),phase=phase,protocol_SHA256=digest,release=record(LOCAL/'release.json'),workers=[])
    def save():state['wall_s']=time.monotonic()-start;write(execution,state)
    def stop(*_):raise InterruptedError('Owned benchmark controller stop')
    previous={s:signal.signal(s,stop)for s in(signal.SIGTERM,signal.SIGINT)}
    try:
        for slot in([0]if phase=='checkpoint'else range(3)):
            action='choose-worker'if phase=='choices'else'evaluate-worker';command=['taskset','-c',str(slot),sys.executable,'-u','-m','experiments.run_noisy_poke_benchmark_v01',action,'--slot',str(slot),'--sha',digest]
            if phase!='choices':command+=['--phase',phase]
            with(LOCAL/f'{phase}-worker-{slot}.log').open('xb')as log:child=subprocess.Popen(command,cwd=ROOT,env={**os.environ,'CUDA_VISIBLE_DEVICES':p['devices'][slot],'CUBLAS_WORKSPACE_CONFIG':':4096:8','OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1'},stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            children.append(child);state['workers'].append(dict(slot=slot,pid=child.pid,physical_UUID=p['devices'][slot],command=command));save()
        while any(c.poll()is None for c in children):
            total=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
            for child,row in zip(children,state['workers']):
                if child.poll()is None:
                    rss=base.old.parity.old.old.mechanical_screen.shared.owned_RSS(child);vram=base.old.parity.owned_VRAM(child.pid,row['physical_UUID']);total+=rss;row['peak_RSS_bytes']=max(row.get('peak_RSS_bytes',0),rss);row['peak_owned_VRAM_bytes']=max(row.get('peak_owned_VRAM_bytes',0),vram)
                    if rss>=RESOURCES['worker_RSS_bytes']or vram>=RESOURCES['owned_VRAM_bytes']:raise MemoryError('Owned worker RSS/VRAM cap')
                elif child.returncode!=0:raise RuntimeError('Owned stage failed; preserve all planned trials')
            limit=RESOURCES['choice_wall_s']if phase=='choices'else RESOURCES['evaluation_wall_s']
            if time.monotonic()-start>limit or prior+time.monotonic()-start>RESOURCES['global_wall_s']:raise TimeoutError('Frozen stage/global benchmark wall cap')
            if total>=RESOURCES['aggregate_RSS_bytes']:raise MemoryError('Own aggregate28GiB cap')
            state['peak_aggregate_RSS_bytes']=max(state.get('peak_aggregate_RSS_bytes',0),total);guard('controller');save();time.sleep(.5)
        for child,row in zip(children,state['workers']):
            base.finalize_child(child,row);path=LOCAL/(f"choices-worker-{row['slot']}.json"if phase=='choices'else f"evaluation-{phase}-worker-{row['slot']}.json");row['result']=record(path);value=read(path)
            if value.get('completed')is not True or value['pid']!=row['pid']or value['slot']!=row['slot']or value['protocol_SHA256']!=digest or value['release']!=record(LOCAL/'release.json')or value['phase']!=phase or value['physical_UUID']!=row['physical_UUID']:raise ValueError('Exact launched worker completion/context')
            expected=p['choice_split'][row['slot']]if phase=='choices'else(p['evaluation_checkpoint']if phase=='checkpoint'else p['evaluation_remaining'][row['slot']])
            if[v['index']for v in value['trials']]!=expected:raise ValueError('Every exact assigned trial retained at worker completion')
        state['completed']=True
    except BaseException as error:state['failure']=dict(type=type(error).__name__,message=str(error))
    finally:
        for child,row in zip(children,state['workers']):row.update(cleanup_exit_code=base.old.parity.old.old.mechanical_screen.shared.old.cost.cleanup(child),exit_code=child.returncode,reaped=True)
        for s,h in previous.items():signal.signal(s,h)
        state['terminal']=True;save()
    return state

def summarize(digest):
    p=validate(digest);all_seals_gate(digest);phases=[completed_phase(v,digest)for v in('choices','checkpoint','remaining')];rows=[]
    for phase in phases[1:]:
        for worker in phase['workers']:
            value=read(ROOT/worker['result']['path'])
            for trial in value['trials']:verify(trial['record']);rows.append(read(ROOT/trial['record']['path']))
    result=dict(protocol_SHA256=digest,summary=design.aggregate(rows),phases=[record(LOCAL/f'execution-{v}.json')for v in('choices','checkpoint','remaining')],all32_trial_records=[v for phase in phases[1:]for worker in phase['workers']for v in read(ROOT/worker['result']['path'])['trials']],scope='Known LINEAR forward-model stress on admitted NHtruth; no calibrated NHposterior/NN/population/SBC claim')
    target=LOCAL/'summary-LOCAL.json'
    if target.exists():raise FileExistsError('No summary overwrite')
    write(target,result);return record(target)
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['prepare','inputs','choose-worker','evaluate-worker','run','summarize','guard']);ap.add_argument('--sha');ap.add_argument('--slot',type=int);ap.add_argument('--phase',choices=['choices','checkpoint','remaining']);ap.add_argument('--role',choices=['controller','input','worker']);a=ap.parse_args()
    if a.action=='guard':guard(a.role,a.slot);print('No model/data/RNG/GPU guard')
    else:
        result=prepare()if a.action=='prepare'else(inputs(a.sha)if a.action=='inputs'else(choose_worker(a.slot,a.sha)if a.action=='choose-worker'else(evaluate_worker(a.slot,a.phase,a.sha)if a.action=='evaluate-worker'else(run(a.phase,a.sha)if a.action=='run'else summarize(a.sha)))));print(json.dumps(result))
        if a.action=='run'and not result['completed']:raise SystemExit(1)
