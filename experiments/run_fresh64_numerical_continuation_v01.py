"""New numerical-only continuation; parent64/896 evidence remains immutable.

Prepare runs parent/fine-control semantic admissions once, then freezes their
transitive source/artifact byte closure. Workers verify that flat closure and
own inputs; they never recursively replay the expensive semantic gate chain.
No source preparation/import/test calls read real parent outputs or draw RNG.
"""
import argparse,ast,hashlib,json,os,resource,shutil,signal,subprocess,sys,time,traceback
from pathlib import Path
import numpy as np
from experiments import campaign_fresh64_continuation_v01 as lineage
from experiments import run_fresh64_nonlinear_calibration_v01 as parent
from experiments import audit_fresh64_nonlinear_calibration_v01 as parent_audit
from experiments import run_fresh64_inner_resolution_control_v01 as fine
from experiments import campaign_fresh64_nonlinear_calibration_v01 as design
from experiments.campaign_nonlinear_conditional_reference_v04 import NonlinearConditionalReferenceV04,ExtendedAmplitudePWLPlan
from experiments.campaign_linear_amplitude_extended_v01 import LinearAmplitudeExtendedReferenceV01
ROOT=parent.ROOT;LOCAL=ROOT/'runs/operator-dev/fresh64-numerical-continuation-v01'
SOURCE='experiments/run_fresh64_numerical_continuation_v01.py';TEST='tests/test_run_fresh64_numerical_continuation_v01.py'
PARENT_SHA='ab1ecddf49342ec337401f2629d681ec88dc4f1424986872f9ed173dcbd20323'
BACKENDS=dict(linear='linear_amplitude_extended_v01_axis_extension',nonlinear='nonlinear_conditional_reference_v04_MIS_axis_extension')
sha,record,write=parent.sha,parent.record,parent.write


def sources():
    todo=[SOURCE,TEST,'tests/test_fresh64_continuation_v01.py','tests/test_linear_amplitude_extended_v01.py'];seen=set()
    while todo:
        path=todo.pop()
        if path in seen:continue
        seen.add(path)
        for node in ast.walk(ast.parse((ROOT/path).read_text())):
            names=[]
            if isinstance(node,ast.Import):names=[x.name for x in node.names]
            elif isinstance(node,ast.ImportFrom)and node.level==0:names=[node.module or '']+[(node.module+'.'+x.name)for x in node.names if node.module]
            for name in names:
                candidate=name.replace('.','/')+'.py'
                if(ROOT/candidate).is_file()and candidate not in seen:todo.append(candidate)
    return {p:sha(ROOT/p)for p in sorted(seen)}


def _hex(v):return isinstance(v,str)and len(v)==64 and all(c in'0123456789abcdef'for c in v)


def flatten_pins(roots,extra=()):
    """Freeze transitive hash-bound JSON records and source/checkpoint maps.

    This is byte provenance, not a replacement for semantic admissions. Roots
    must FIRST be accepted by parent/fine-control gates in prepare().
    """
    pins={};pending=[]
    def add(path,digest):
        path=Path(path).resolve()
        if not path.is_relative_to(ROOT.resolve())or not _hex(digest)or not path.is_file()or sha(path)!=digest:raise ValueError('Exact in-workspace transitive artifact/source bytes')
        key=str(path.relative_to(ROOT.resolve()))
        if key in pins:
            if pins[key]!=digest:raise ValueError('Conflicting transitive commitment')
            return
        pins[key]=digest
        if path.suffix=='.json':pending.append(path)
    def walk(value,base):
        if isinstance(value,dict):
            if 'path'in value and 'sha256'in value:add(ROOT/value['path'],value['sha256'])
            for key,item in value.items():
                if isinstance(key,str)and _hex(item):
                    options=[ROOT/key,base/key];found=next((p for p in options if p.is_file()),None)
                    if found is not None:add(found,item)
                    elif '/'in key or '\\'in key or Path(key).suffix.lower()in('.json','.npz','.npy','.py','.txt','.csv','.sha256'):raise ValueError('Missing transitive source/artifact commitment: '+key)
                walk(item,base)
        elif isinstance(value,list):
            for item in value:walk(item,base)
    for value in roots:walk(value,ROOT)
    for item in extra:add(ROOT/item['path'],item['sha256'])
    while pending:
        path=pending.pop();walk(json.loads(path.read_text()),path.parent)
    return dict(sorted(pins.items()))


def verify_flat(pins):
    if not isinstance(pins,dict)or not pins:raise ValueError('Nonempty admitted transitive input closure')
    for path,digest in pins.items():
        if not _hex(digest)or sha(ROOT/path)!=digest:raise ValueError('Changed flat input/source bytes: '+path)


def validate_resources(r,both_axes=False):
    n=r.get('workers');GiB=1024**3
    if type(n)is not int or not 1<=n<=8 or r.get('worker_CPUs')!=list(range(n))or r.get('controller_CPU')!=16 or r.get('threads')!=1 or r.get('GPU')is not False:raise ValueError('1..8 CPU0..7 workers/controller16/one-thread/noGPU')
    cap=r.get('worker_RSS_bytes')
    if cap not in(4*GiB,8*GiB)or(both_axes and cap!=8*GiB)or r.get('controller_RSS_bytes')!=4*GiB or r.get('aggregate_RSS_bytes')!=n*cap+4*GiB or r.get('RAM_reserve_bytes')!=30*GiB or r.get('disk_reserve_bytes')!=30*GiB or r.get('host_preflight_bytes')!=r['aggregate_RSS_bytes']+30*GiB or r.get('maximum_writes_bytes')!=12*GiB:raise ValueError('Frozen4/8GiB worker+controller/aggregate/reserve/write budget')
    if type(r.get('worker_wall_s'))is not int or not 30<=r['worker_wall_s']<=18000 or type(r.get('global_wall_s'))is not int or not r['worker_wall_s']<=r['global_wall_s']<=19800:raise ValueError('Cost-reviewed finite worker/global deadlines')
    return r


def guard(r,slot=None):
    cpu=r['controller_CPU']if slot is None else r['worker_CPUs'][slot]
    if set(os.sched_getaffinity(0))!={cpu}or os.environ.get('CUDA_VISIBLE_DEVICES')!=''or any(os.environ.get(k)!='1'for k in('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS')):raise ValueError('Exact owned CPU/thread/noGPU guard')
    cap=r['controller_RSS_bytes']if slot is None else r['worker_RSS_bytes']
    if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024>cap:raise MemoryError('Own frozen RSS highwater')
    if parent.oldstudy.available_memory()<r['RAM_reserve_bytes']or shutil.disk_usage(ROOT).free<r['disk_reserve_bytes']:raise MemoryError('30GiB RAM/disk reserve')
    if LOCAL.exists()and sum(f.stat().st_size for f in LOCAL.rglob('*')if f.is_file())>r['maximum_writes_bytes']:raise OSError('Own12GiB write cap')


def parent_admission():
    """Called once in prepare AFTER terminal; no arrays or scientific scoring."""
    e=json.loads((parent.LOCAL/'execution.json').read_text())
    if not all(e.get(k)is True for k in('completed','terminal'))or'failure'in e or any(not row.get('reaped')or row.get('exit_code')!=0 for row in e['workers']):raise ValueError('Original complete clean execution required')
    a=json.loads((parent_audit.LOCAL/'result.json').read_text())
    if a.get('completed')is not True or a.get('integrity_passed')is not True or a.get('execution_clean')is not True or a.get('comparison_blocked')is not True or a.get('qualified_posteriors',896)>=896:raise ValueError('Original independent integrity PASS/numerical-blocked receipt required')
    if a['protocol']!=record(parent.LOCAL/'protocol.json')or a['release']!=record(parent.LOCAL/'release.json')or a['execution']!=record(parent.LOCAL/'execution.json')or a['summary']!=record(parent.LOCAL/'summary-LOCAL.json')or a['source_SHA256']!=parent_audit.sources():raise ValueError('Exact frozen original independent audit bindings')
    release=json.loads((parent.LOCAL/'release.json').read_text())
    if release.get('audit_source_SHA256')!={name:sha(ROOT/name)for name in(parent_audit.SOURCE,parent_audit.TEST)}:raise ValueError('Predeclared original auditor source pins')
    p,ee,ss,rows,manifest=parent_audit.terminal_metadata(PARENT_SHA,sha(parent.LOCAL/'execution.json'),sha(parent.LOCAL/'summary-LOCAL.json'))
    closed=[e['controller_pid']]+[row['pid']for row in e['workers']]
    for pid in closed:parent.full.prefix.absent(pid)
    roots=dict(protocol=record(parent.LOCAL/'protocol.json'),release=record(parent.LOCAL/'release.json'),execution=record(parent.LOCAL/'execution.json'),summary=record(parent.LOCAL/'summary-LOCAL.json'),independent_audit=record(parent_audit.LOCAL/'result.json'),source_SHA256=p['source_SHA256'],audit_source_SHA256=a['source_SHA256'])
    return p,rows,manifest,roots,closed


def project_numerical(rows):
    """No truth/PIT/coverage/RMS field participates in the census decision."""
    keys=('kind','qualified','physical_valid','numerical_backend','raw_SHA256','context','diagnostics','strict_absolute_checks')
    return [dict(index=row['index'],completed=row.get('completed',False),physical_valid=row.get('physical_valid',False),posteriors={kind:{phase:{key:r[key]for key in keys if key in r}for phase,r in row.get('posteriors',{}).get(kind,{}).items()}for kind in lineage.MODELS})for row in rows]


def posterior_bindings(rows):
    bindings={}
    for row in rows:
        for kind,values in row.get('posteriors',{}).items():
            for phase,r in values.items():
                path=parent.LOCAL/f"object-{row['index']:02d}"/(kind+'-'+phase+'-posterior-LOCAL.json')
                if path.exists():bindings[lineage.context_id(row['index'],kind,phase)]=dict(json=record(path),raw=record(path.with_suffix('.npz'))if path.with_suffix('.npz').exists()else None)
    return bindings


def prepare(fine_control_SHA256,resources):
    validate_resources(resources);guard(resources)
    pp,rows,manifest,old,closed=parent_admission();control=fine.completed_control_gate(fine_control_SHA256)
    if control['origin']['primary_protocol']!=old['protocol']or control['origin']['primary_release']!=old['release']:raise ValueError('Same-parent qualified fine control')
    classification=check_census(lineage.census(project_numerical(rows),posterior_bindings(rows)));both=any(e.get('plan',{}).get('inner',[])[-1:]==[1024]and e.get('plan',{}).get('outer',[])[-1:]==[1024]for e in classification['entries']);validate_resources(resources,both);guard(resources)
    if parent.oldstudy.available_memory()<resources['host_preflight_bytes']:raise MemoryError('Whole owned budget+30GiB reserve preflight')
    if(LOCAL/'protocol.json').exists():raise FileExistsError('No implicit refreeze')
    ce=json.loads((ROOT/control['execution']['path']).read_text());closed+=[ce['controller_pid'],ce['worker_pid']]
    pins=flatten_pins([old,control],extra=[dict(path=str(Path(name).relative_to(ROOT)),sha256=digest)for name,digest in manifest.items()]);verify_flat(pins)
    write(LOCAL/'census-LOCAL.json',classification);write(LOCAL/'input-pins.json',pins)
    p=dict(schema='fresh64-numerical-continuation-v01',source_SHA256=sources(),parent=old,parent_protocol_SHA256=PARENT_SHA,parent_configuration=dict(atlas_fingerprint=pp['atlas_fingerprint'],full_atlas=pp['full_atlas'],cost_receipt=pp['cost_receipt'],target_labels=pp['target_labels']),fine_control=control,census=record(LOCAL/'census-LOCAL.json'),input_pins=record(LOCAL/'input-pins.json'),closed_prerequisite_PIDs=sorted(set(closed)),resources=resources,both_axis_control_required=both,objects=64,contexts=896,no_new_draws_or_PDE=True,numerical_census_only=True,original_qualified_reused_unchanged=True,backends=BACKENDS)
    write(LOCAL/'protocol.json',p);return record(LOCAL/'protocol.json')


def check_census(c):
    expected={lineage.context_id(i,m,k)for i in range(64)for m in lineage.MODELS for k in lineage.PHASES}
    entries=c.get('entries',[])
    if c.get('objects')!=64 or c.get('contexts')!=896 or len(entries)!=896 or {e.get('id')for e in entries}!=expected:raise ValueError('Frozen ALL64/896 census membership')
    counts={name:sum(e.get('decision')==name for e in entries)for name in('reuse_qualified','refine','blocked')}
    if sum(counts.values())!=896 or c.get('counts')!=counts:raise ValueError('Frozen exhaustive census decisions')
    for e in entries:
        if e['id']!=lineage.context_id(e['index'],e['kind'],e['phase']):raise ValueError('Exact census identity')
        if e['decision']=='refine'and(e['levels']!=lineage.integration_levels(e['plan'])or e['backend_family']!=lineage.backend_family(e['kind'],e['plan'])):raise ValueError('Exact fixed per-axis extended plan')
    return c


def validate(digest):
    p=json.loads((LOCAL/'protocol.json').read_text())
    if sha(LOCAL/'protocol.json')!=digest or p['source_SHA256']!=sources()or p['backends']!=BACKENDS or p['objects']!=64 or p['contexts']!=896 or p['no_new_draws_or_PDE']is not True:raise ValueError('Frozen new source/protocol/membership')
    for key in('census','input_pins'):
        if record(ROOT/p[key]['path'])!=p[key]:raise ValueError('Frozen census/input closure')
    classification=check_census(json.loads((ROOT/p['census']['path']).read_text()));pins=json.loads((ROOT/p['input_pins']['path']).read_text());verify_flat(pins)
    for pid in p['closed_prerequisite_PIDs']:parent.full.prefix.absent(pid)
    both=any(e.get('plan',{}).get('inner',[])[-1:]==[1024]and e.get('plan',{}).get('outer',[])[-1:]==[1024]for e in classification['entries'])
    if p.get('both_axis_control_required')is not both:raise ValueError('Frozen both-axis control requirement matches complete census')
    validate_resources(p['resources'],both)
    return p,classification


def release_gate(digest):
    p,c=validate(digest)
    if p['both_axis_control_required']:raise ValueError('Both-axis1024 jobs require a separately reviewed same-observation 8GiB control and a new runner version before release')
    r=json.loads((LOCAL/'release.json').read_text())
    if r.get('approved')is not True or r.get('protocol_SHA256')!=digest or r.get('source_SHA256')!=p['source_SHA256']or r.get('parent')!=p['parent']or r.get('fine_control')!=p['fine_control']:raise ValueError('Explicit root same-data numerical continuation release')
    return p,c


def recipe_SHA(kind,entry,targets,atlas_fingerprint):
    payload=dict(kind=kind,numerical_backend=BACKENDS[kind],plan=entry['plan'],targets=targets,atlas=atlas_fingerprint if kind=='nonlinear'else None,parent_record_SHA256=entry['original']['json']['sha256'])
    return hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).hexdigest()


def acquired_observations(entry):
    i=entry['index'];phase=entry['phase'];directory=parent.LOCAL/f'object-{i:02d}'
    with np.load(directory/(design.PREFIX+'-truth-LOCAL.npz'),allow_pickle=False)as z:prefix_response=z['response_m'].copy()
    with np.load(directory/'draws-LOCAL.npz',allow_pickle=False)as z:prefix_noise=z['prefix_noise'].copy();probe_noise=z['probe_noise'].copy()
    obs=[(design.PREFIX,prefix_response+prefix_noise)]
    if phase!='prefix':
        family,reset=phase.split('-r');f=('normal','axial').index(family);label=design.PROBES[f]
        if label==design.PREFIX:response=prefix_response
        else:
            with np.load(directory/(label+'-truth-LOCAL.npz'),allow_pickle=False)as z:response=z['response_m'].copy()
        obs.append((label,response+probe_noise[f,int(reset)]))
    if parent.oldstudy.observation_fingerprint(obs)!=entry['original']['context']['observations_SHA256']:raise ValueError('Exact original acquired data before inference')
    return obs


def posterior_job(entry,p,digest,model,atlas,ratio):
    if entry.get('decision')!='refine':raise ValueError('Only original unqualified census jobs may run')
    directory=LOCAL/'jobs'/f"{entry['index']:02d}-{entry['kind']}-{entry['phase']}";directory.mkdir(parents=True,exist_ok=False);path=directory/'posterior-LOCAL.json'
    old=entry['original']['context'];kind=entry['kind'];plan=ExtendedAmplitudePWLPlan(**{k:tuple(v)if k in('inner','outer')else v for k,v in entry['plan'].items()})
    context=dict(protocol_SHA256=digest,index=entry['index'],phase=entry['phase'],parent_protocol_SHA256=old['protocol_SHA256'],parent_record_SHA256=entry['original']['json']['sha256'],draws_SHA256=old['draws_SHA256'],observations_SHA256=old['observations_SHA256'],model_fingerprint=old['model_fingerprint'],true_unit_SHA256=old['true_unit_SHA256'],refinement_plan=entry['plan'],recipe_SHA256=recipe_SHA(kind,entry,p['parent_configuration']['target_labels'],p['parent_configuration']['atlas_fingerprint']))
    row=dict(context=context,kind=kind,physical_valid=True,qualified=False,numerical_backend=BACKENDS[kind],original_negative=entry['original'],expected_levels=entry['levels'])
    try:
        observations=acquired_observations(entry)
        if parent.oldstudy.model_fingerprint(model)!=old['model_fingerprint']:raise ValueError('Same canonical modal model')
        if kind=='linear':reference=LinearAmplitudeExtendedReferenceV01(None,observations,p['parent_configuration']['target_labels'],log_ratio_predict=ratio,retain_hypotheses=False)
        else:reference=NonlinearConditionalReferenceV04(ratio,None,observations,p['parent_configuration']['target_labels'],residual_predict=atlas.predict,retain_hypotheses=False,evaluation_chunk=1024)
        answer=reference.qualify(plan=plan);last=answer['local_final'];checks=parent.cost.engine_gates(answer);strict=parent.cost.absolute_engine_witnesses(answer);qualified=bool(checks['passed']and strict['passed']);row.update(diagnostics=answer['diagnostics'],strict_absolute_checks=strict,numerical_admission_checks=checks,qualified=qualified)
        if [(x.get('outer'),x.get('inner'))for x in answer['diagnostics']['levels']]!=[(x['outer'],x['inner'])for x in entry['levels']]:raise ValueError('Every exact six/seven integration level retained')
        if last is not None:
            arrays=dict(mean=last['predictive_mean'],SD=last['predictive_SD'],unit_mean=last['unit_mean'],unit_covariance=last['unit_covariance'],unit_CDF=last['unit_CDF'],unit_quantiles=last['unit_quantiles'],MAP_unit=last['plugin_unit'],MAP_prediction=last['plugin_prediction'],MAP_log_likelihood=last['plugin_log_likelihood'])
            if qualified:
                # Truth is accessed ONLY for unchanged post-qualification PIT/scoring.
                original=parent.LOCAL/f"object-{entry['index']:02d}"
                with np.load(original/'draws-LOCAL.npz',allow_pickle=False)as z:unit=z['unit'].copy()
                if hashlib.sha256(np.ascontiguousarray(unit,dtype='<f8').tobytes()).hexdigest()!=old['true_unit_SHA256']:raise ValueError('Unchanged calibration truth commitment')
                metrics=design.exact_density_calibration(reference,last,unit,outer=plan.outer[-1],inner=plan.inner[-1]);arrays.update(metrics);row.update({k:np.asarray(v).tolist()for k,v in metrics.items()})
                targets=[]
                for label in design.TARGETS:
                    with np.load(original/(label+'-truth-LOCAL.npz'),allow_pickle=False)as z:targets.append(z['response_m'].copy())
                truth=np.concatenate(targets);row.update(target_RMS_m=float(np.sqrt(np.mean((last['predictive_mean']-truth)**2))),MAP_target_RMS_m=float(np.sqrt(np.mean((last['plugin_prediction']-truth)**2))))
            target=path.with_suffix('.npz');np.savez_compressed(target,**arrays);row['raw_SHA256']=sha(target)
    except Exception as error:
        if isinstance(error,(MemoryError,TimeoutError,InterruptedError)):raise
        row.update(qualified=False,failure=dict(type=type(error).__name__,message=str(error),traceback=traceback.format_exc()))
    write(path,row);return dict(record=row,binding=dict(json=record(path),raw=record(path.with_suffix('.npz'))if path.with_suffix('.npz').exists()else None),plan=entry['plan'])


def worker(slot,digest):
    p,c=release_gate(digest);r=p['resources'];guard(r,slot)
    directory=LOCAL/f'worker-{slot}';directory.mkdir(exist_ok=False);start=time.monotonic();state=dict(slot=slot,pid=os.getpid(),completed=False,protocol_SHA256=digest,release=record(LOCAL/'release.json'),jobs=[])
    def save():state.update(wall_s=time.monotonic()-start,peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024);write(directory/'summary-LOCAL.json',state)
    def stopped(*_):raise InterruptedError('Explicit owned worker interruption')
    def timed(*_):raise TimeoutError('Frozen continuation worker deadline')
    previous={sig:signal.signal(sig,handler)for sig,handler in((signal.SIGTERM,stopped),(signal.SIGALRM,timed))};signal.alarm(r['worker_wall_s'])
    try:
        fwd,model=parent.oldstudy.make_model(device_mode='cpu');atlas=parent.load_atlas(p['parent_configuration']);ratio=parent.exact_ratio_callback(model);save();guard(r,slot)
        jobs=[entry for entry in c['entries']if entry['decision']=='refine']
        for entry in jobs[slot::r['workers']]:
            update=posterior_job(entry,p,digest,model,atlas,ratio);guard(r,slot);state['jobs'].append(dict(id=entry['id'],binding=update['binding'],plan=update['plan']));save()
        state['completed']=True;save()
    except BaseException as error:state['failure']=dict(type=type(error).__name__,message=str(error),traceback=traceback.format_exc());save();raise
    finally:
        signal.alarm(0)
        for sig,handler in previous.items():signal.signal(sig,handler)
        save()
    return state


def load_parent_rows():
    rows=[]
    for i in range(64):
        path=parent.LOCAL/f'object-{i:02d}/summary-LOCAL.json'
        rows.append(json.loads(path.read_text())if path.exists()else parent.placeholder(i,'Original missing object retained'))
    return rows


def collect_updates(p,c,digest):
    jobs={e['id']:e for e in c['entries']if e['decision']=='refine'};updates={}
    for slot in range(p['resources']['workers']):
        path=LOCAL/f'worker-{slot}/summary-LOCAL.json'
        if not path.exists():continue
        w=json.loads(path.read_text())
        if w['slot']!=slot or w.get('protocol_SHA256')!=digest:raise ValueError('Original fixed worker slot/protocol')
        assigned={entry['id']for entry in list(jobs.values())[slot::p['resources']['workers']]}
        for job in w['jobs']:
            identity=job['id'];binding=job['binding']
            if identity not in assigned or identity in updates or job['plan']!=jobs[identity]['plan']:raise ValueError('No replacement/duplicate/unassigned refinement')
            if record(ROOT/binding['json']['path'])!=binding['json']:raise ValueError('Committed refined JSON')
            row=json.loads((ROOT/binding['json']['path']).read_text())
            if row.get('raw_SHA256'):
                if binding.get('raw')is None or record(ROOT/binding['raw']['path'])!=binding['raw']or row['raw_SHA256']!=binding['raw']['sha256']:raise ValueError('Committed refined raw archive')
            updates[identity]=dict(record=row,binding=binding,plan=job['plan'])
    return updates


def aggregate(p,c,digest):
    updates=collect_updates(p,c,digest);merged=lineage.merge_accepted(load_parent_rows(),c,updates,protocol_SHA256=digest);rows=merged['rows'];objects=[]
    for row in rows:
        i=row['index'];path=LOCAL/f'object-{i:02d}/summary-LOCAL.json'
        if path.exists():raise FileExistsError('No merged object overwrite')
        row['parent_protocol_SHA256']=row.get('protocol_SHA256',PARENT_SHA);row['protocol_SHA256']=digest
        if 'checkpoint_SHA256'in row:row['parent_checkpoint_SHA256']=row.pop('checkpoint_SHA256')
        old=parent.LOCAL/f'object-{i:02d}/summary-LOCAL.json';row['parent_object_summary']=record(old)if old.exists()else None;write(path,row);objects.append(record(path))
    write(LOCAL/'merged-rows-LOCAL.json',rows);write(LOCAL/'update-manifest-LOCAL.json',{key:dict(binding=value['binding'],plan=value['plan'])for key,value in updates.items()})
    calibration=design.summarize_calibration(rows);prediction=design.paired_prediction_metrics(rows);arrays={}
    for kind in lineage.MODELS:
        covered,pit=design.primary_arrays(rows,kind);arrays[kind+'_covered']=covered;arrays[kind+'_PIT']=pit
    target=LOCAL/'primary-calibration-LOCAL.npz'
    if target.exists():raise FileExistsError('No continued calibration archive overwrite')
    np.savez_compressed(target,**arrays)
    result=dict(completed=all(row.get('completed',False)for row in rows),objects=64,expected_posteriors=896,posterior_contexts_retained=896,all64_denominators_retained=True,qualified_posteriors=merged['reused_qualified']+merged['newly_qualified'],reused_qualified=merged['reused_qualified'],newly_qualified=merged['newly_qualified'],unresolved=merged['unresolved'],physical_valid_objects=sum(row.get('physical_valid',False)for row in rows),comparison_blocked=merged['comparison_blocked']or calibration['comparison_blocked']or prediction['comparison_blocked'],protocol_SHA256=digest,parent=p['parent'],fine_control=p['fine_control'],primary_calibration=calibration,paired_metrics=prediction,raw_calibration=record(target),merged_rows=record(LOCAL/'merged-rows-LOCAL.json'),update_manifest=record(LOCAL/'update-manifest-LOCAL.json'),object_summaries=objects,old_primary_evidence_unchanged=True,no_new_draws_or_PDE=True)
    write(LOCAL/'summary-LOCAL.json',result);return result


def worker_command(slot,digest,*,guard_only=False):return ['taskset','-c',str(slot),sys.executable,'-u','-m','experiments.run_fresh64_numerical_continuation_v01','worker-guard'if guard_only else'worker','--sha',digest,'--slot',str(slot)]


def run(digest):
    p,c=release_gate(digest);r=p['resources'];guard(r)
    if parent.oldstudy.available_memory()<r['host_preflight_bytes']:raise MemoryError('Whole owned budget+30GiB reserve')
    if(LOCAL/'execution.json').exists()or any(LOCAL.glob('worker-*')):raise FileExistsError('No implicit continuation retry')
    start=time.monotonic();children=[];logs=[];state=dict(completed=False,terminal=False,controller_pid=os.getpid(),protocol_SHA256=digest,release=record(LOCAL/'release.json'),workers=[])
    def save():state['wall_s']=time.monotonic()-start;write(LOCAL/'execution.json',state)
    def stopped(*_):raise InterruptedError('Explicit owned continuation controller stop')
    previous={sig:signal.signal(sig,stopped)for sig in(signal.SIGTERM,signal.SIGINT)}
    try:
        for slot in range(r['workers']):
            env={**os.environ,'CUDA_VISIBLE_DEVICES':'','OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1'};log=(LOCAL/f'worker-{slot}.log').open('xb');logs.append(log);child=subprocess.Popen(worker_command(slot,digest),cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);children.append(child);state['workers'].append(dict(slot=slot,pid=child.pid,reaped=False,started_s=time.monotonic()-start));save()
        while any(child.poll()is None for child in children):
            guard(r);rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
            for child,row in zip(children,state['workers']):
                if child.poll()is not None:
                    row.update(reaped=True,exit_code=child.wait())
                    if child.returncode!=0:raise RuntimeError('Refinement worker execution failed; preserve partial896 denominator')
                    continue
                own=parent.full.owned_RSS(child);rss+=own
                if own>r['worker_RSS_bytes']or time.monotonic()-start-row['started_s']>r['worker_wall_s']:raise TimeoutError('Independent worker RSS/deadline')
            state['maximum_aggregate_RSS_bytes']=max(state.get('maximum_aggregate_RSS_bytes',0),rss)
            if rss>r['aggregate_RSS_bytes']or time.monotonic()-start>r['global_wall_s']:raise TimeoutError('Aggregate RSS/global deadline')
            save();time.sleep(.5)
        for child,row in zip(children,state['workers']):row.update(reaped=True,exit_code=child.wait())
        if any(child.returncode!=0 for child in children):raise RuntimeError('Worker exit not0')
        for child in children:parent.full.prefix.absent(child.pid)
        result=aggregate(p,c,digest);state.update(completed=result['completed'],comparison_blocked=result['comparison_blocked'],summary=record(LOCAL/'summary-LOCAL.json'))
    except BaseException as error:state['failure']=dict(type=type(error).__name__,message=str(error),traceback=traceback.format_exc())
    finally:
        for child,row in zip(children,state['workers']):
            row.update(reaped=True,exit_code=parent.full.prefix.cost.cleanup(child));parent.full.prefix.absent(child.pid)
            path=LOCAL/f"worker-{row['slot']}/summary-LOCAL.json"
            if path.exists():row['summary']=record(path)
        if not(LOCAL/'summary-LOCAL.json').exists():
            try:
                result=aggregate(p,c,digest);state.update(comparison_blocked=True,summary=record(LOCAL/'summary-LOCAL.json'))
            except Exception as error:state['terminal_aggregate_failure']=dict(type=type(error).__name__,message=str(error))
        for log in logs:log.close()
        for sig,handler in previous.items():signal.signal(sig,handler)
        state['terminal']=True;save()
    return state


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['prepare','worker','run','worker-guard']);parser.add_argument('--sha');parser.add_argument('--slot',type=int);parser.add_argument('--fine-sha');parser.add_argument('--resources-json');args=parser.parse_args()
    if args.action=='prepare':print(json.dumps(prepare(args.fine_sha,json.loads(Path(args.resources_json).read_text()))))
    elif args.action=='worker':r=worker(args.slot,args.sha);print(json.dumps(dict(completed=r['completed'],jobs=len(r['jobs']))))
    elif args.action=='run':r=run(args.sha);print(json.dumps(dict(completed=r['completed'],terminal=r['terminal'])));raise SystemExit(0 if r['completed']else 1)
    else:
        r=json.loads(Path(args.resources_json).read_text())if args.resources_json else validate(args.sha)[0]['resources'];validate_resources(r);guard(r,args.slot);print('guard passed')
