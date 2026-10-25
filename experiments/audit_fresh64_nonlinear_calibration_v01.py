"""Independent terminal audit of saved fresh64 evidence; no PDE/posterior replay.

Import/tests/preparation never access actual fresh arrays or draw seeds. Audit
requires exact terminal seals and checks every committed file before np.load.
Missing planned members stay in the64-object denominator and block comparisons.
"""
from __future__ import annotations
import argparse,ast,hashlib,json,os,resource,signal,time,traceback
from pathlib import Path
import numpy as np
from experiments import run_fresh64_nonlinear_calibration_v01 as fresh
from experiments import campaign_fresh64_nonlinear_calibration_v01 as design
from experiments.campaign_cluster_calibration_v01 import primary_calibration,paired_cluster_intervals
ROOT=fresh.ROOT;LOCAL=ROOT/'runs/operator-dev/fresh64-independent-audit-v01'
SOURCE='experiments/audit_fresh64_nonlinear_calibration_v01.py';TEST='tests/test_audit_fresh64_nonlinear_calibration_v01.py'
MODELS=('linear','nonlinear');KEYS=('prefix','normal-r0','normal-r1','normal-r2','axial-r0','axial-r1','axial-r2')
sha,record,write=fresh.sha,fresh.record,fresh.write


def demand(test,message):
    if not test:raise ValueError(message)


def array_digest(a):return hashlib.sha256(np.ascontiguousarray(a,dtype='<f8').tobytes()).hexdigest()


def compare(a,b,where='report'):
    """Structural equality with tight scalar arithmetic tolerance, no NaN repair."""
    if isinstance(a,dict):
        demand(isinstance(b,dict)and set(a)==set(b),'Keys differ: '+where)
        for k in a:compare(a[k],b[k],where+'.'+k)
    elif isinstance(a,(list,tuple)):
        demand(isinstance(b,(list,tuple))and len(a)==len(b),'Length differs: '+where)
        for i,(x,y)in enumerate(zip(a,b)):compare(x,y,where+f'[{i}]')
    elif isinstance(a,(float,np.floating)):
        demand(isinstance(b,(float,int))and np.isfinite(a)and np.isfinite(b)and abs(a-b)<=1e-12,'Numeric difference: '+where)
    else:demand(a==b,'Value differs: '+where)


def planned_records(rows):
    by_index={}
    for r in rows:
        i=r.get('index');demand(type(i)is int and 0<=i<64 and i not in by_index,'Duplicate/replaced/out-of-plan object')
        by_index[i]=r
    result=[]
    for i in range(64):
        r=by_index.get(i,dict(index=i,completed=False,physical_valid=False,missing=True));r=dict(r);post=r.get('posteriors',{})
        demand(not(set(post)-set(MODELS)),'Unexpected model membership')
        r['posteriors']={}
        for model in MODELS:
            values=post.get(model,{});demand(not(set(values)-set(KEYS)),'Unexpected phase/reset membership')
            r['posteriors'][model]={k:values.get(k,dict(qualified=False,physical_valid=False,attempted=False))for k in KEYS}
        result.append(r)
    return result


def lean_metrics(z,true_unit,target):
    u=np.asarray(true_unit,float);target=np.asarray(target,float)
    expected={'mean':(96,),'SD':(96,),'unit_mean':(2,),'unit_quantiles':(2,3),'unit_CDF':(2,33),'PIT':(2,),'covered':(2,),'intrinsic_interval_width':(2,),'intrinsic_signed_bias':(2,),'MAP_prediction':(96,)}
    for k,shape in expected.items():demand(k in z and z[k].shape==shape and np.isfinite(z[k]).all(),'Malformed saved lean '+k)
    demand(u.shape==(2,)and np.isfinite(u).all()and np.all((u>=0)&(u<1))and target.shape==(96,)and np.isfinite(target).all(),'Saved truth/target layout')
    q=z['unit_quantiles'];cdf=z['unit_CDF'];pit=z['PIT']
    demand(np.all(z['SD']>=0)and np.all(np.diff(q,axis=1)>=0)and np.all((q>=0)&(q<=1)),'Invalid SD/quantiles')
    demand(np.all(np.diff(cdf,axis=1)>=-1e-12)and np.all(cdf[:,0]==0)and np.all(cdf[:,-1]==1)and np.all((cdf>=0)&(cdf<=1+32*np.finfo(float).eps)),'Invalid qualified saved CDF grid')
    demand(np.all((pit>=0)&(pit<=1)),'Strict PIT domain; no clipping')
    lower=np.floor(u*32).astype(int);upper=np.ceil(u*32).astype(int);ar=np.arange(2)
    demand(np.all(pit>=cdf[ar,lower]-1e-12)and np.all(pit<=cdf[ar,upper]+1e-12),'PIT incompatible with same saved qualified CDF bracket')
    answer=dict(covered=(u>=q[:,0])&(u<=q[:,2]),PIT=pit.copy(),intrinsic_interval_width=q[:,2]-q[:,0],intrinsic_signed_bias=z['unit_mean']-u,target_RMS_m=float(np.sqrt(np.mean((z['mean']-target)**2))),MAP_target_RMS_m=float(np.sqrt(np.mean((z['MAP_prediction']-target)**2))))
    for k in('covered','intrinsic_interval_width','intrinsic_signed_bias'):demand(np.allclose(z[k],answer[k],rtol=0,atol=1e-15),'Saved calibration field differs: '+k)
    return answer


ORIGINAL_GATES=dict(logZ=.01,CDF=.005,quantile=.002,unit_mean=.001,target_absolute_m=2e-6,target_relative_SD=.01)
ORIGINAL_PAIRS=((128,256),(256,512))
ORIGINAL_LEVELS=((512,128),(512,256),(512,512),(128,512),(256,512))


def numerical_record_checks(r,model):
    """Verify original declared rules/plan before accepting saved diagnostics."""
    d=r['diagnostics'];schema='conditional-amplitude-v02'if model=='linear'else'nonlinear-conditional-reference-v03'
    demand(d.get('schema')==schema and d.get('qualified')is True and d.get('independent_adaptive_per_posterior_required')is True,'Qualified same-source CDF/adaptive declaration')
    demand(d.get('gates')==ORIGINAL_GATES,'Exact original numerical gates; no metadata relaxation')
    levels=d.get('levels',[])
    demand(len(levels)==5 and [(v.get('outer'),v.get('inner'))for v in levels]==list(ORIGINAL_LEVELS),'Exact five original integration levels')
    for v in levels:
        if v.get('status')=='computed':demand(np.isfinite(v.get('log_evidence',np.nan))and type(v.get('weighted_hypotheses'))is int and v['weighted_hypotheses']>0,'Finite resolved integration level')
        else:demand(v.get('status')=='unresolved'and isinstance(v.get('reason'),str)and bool(v['reason']),'Coarse unresolved level requires preserved reason')
    needed={(512,256),(512,512),(256,512)}
    demand(all(v.get('status')=='computed'for v in levels if(v['outer'],v['inner'])in needed),'Final original refinement states must resolve')
    demand(set(d.get('successive',{}))=={'inner','outer'},'Both original refinement axes')
    errors=('log_evidence_difference','maximum_CDF_difference','maximum_unit_quantile_difference','maximum_predictive_mean_difference_m','maximum_predictive_SD_difference_m','maximum_unit_mean_difference')
    for axis in('inner','outer'):
        rows=d['successive'][axis]
        demand(len(rows)==2 and [(v.get('coarse'),v.get('fine'))for v in rows]==list(ORIGINAL_PAIRS),'Exact original order pairs/final512')
        for v in rows:
            demand(type(v.get('passed'))is bool,'Preserved refinement status')
            if 'reason'in v:demand(v['passed']is False and isinstance(v['reason'],str)and bool(v['reason']),'Unresolved comparison reason')
            else:demand(all(isinstance(v.get(k),(float,int))and np.isfinite(v[k])and v[k]>=0 for k in errors),'Finite saved comparison errors')
    demand(fresh.cost.engine_gates(dict(qualified=True,diagnostics=d))['passed'],'Every original/adaptive/actual2um posterior gate')
    compare(r['strict_absolute_checks'],fresh.cost.absolute_engine_witnesses(dict(qualified=True,diagnostics=d)),'Strict2um record')
    return True


def aggregate(records):
    """Independently assemble metrics, then use frozen statistical primitives."""
    records=planned_records(records);cal={};primary={};labels=tuple(p+'_'+d for p in design.PHASES for d in('E','K'))
    def post(row,model,key):return row['posteriors'][model][key]
    for model in MODELS:
        covered=np.full((64,6),np.nan);pit=covered.copy()
        for row in records:
            for j,key in enumerate(('prefix','normal-r0','axial-r0')):
                r=post(row,model,key)
                if r.get('qualified')and r.get('physical_valid'):covered[row['index'],2*j:2*j+2]=r['covered'];pit[row['index'],2*j:2*j+2]=r['PIT']
        primary[model+'_covered']=covered;primary[model+'_PIT']=pit;cal[model]=primary_calibration(covered,pit,labels=labels)
    cal.update(nonlinear_primary_family=list(labels),linear_role='Comparator diagnostics only; nonlinear six-entry family is predeclared primary',comparison_blocked=not all(cal[m]['complete']for m in MODELS))
    names=[];columns=[]
    def add(name,v):names.append(name);columns.append(np.asarray(v,float))
    def field(model,key,attr,parameter=None):
        answer=[]
        for row in records:
            r=post(row,model,key);v=r.get(attr,np.nan)if r.get('qualified')and r.get('physical_valid')else np.nan
            answer.append(v if parameter is None or np.ndim(v)==0 else v[parameter])
        return np.asarray(answer,float)
    for model in MODELS:
        prefix=field(model,'prefix','target_RMS_m');add(model+'_prefix_RMS_m',prefix)
        for family in('normal','axial'):
            values=np.column_stack([field(model,family+f'-r{r}','target_RMS_m')for r in range(3)])
            for suffix,v in(('reset0',values[:,0]),('mean3',values.mean(axis=1))):add(model+'_'+family+'_'+suffix+'_RMS_m',v)
            for suffix,v in(('reset0',values[:,0]),('mean3',values.mean(axis=1))):add(model+'_prefix_minus_'+family+'_'+suffix+'_RMS_m',prefix-v)
    for key in('prefix','normal-r0','axial-r0'):add('nonlinear_minus_linear_'+key+'_RMS_m',field('nonlinear',key,'target_RMS_m')-field('linear',key,'target_RMS_m'))
    for model in MODELS:
        for p,name in enumerate(('E','K')):
            for key in('prefix','normal-r0','axial-r0'):
                for attr in('intrinsic_interval_width','intrinsic_signed_bias'):add(model+'_'+key+'_'+name+'_'+attr,field(model,key,attr,p))
            for family in('normal','axial'):
                for attr in('covered','intrinsic_interval_width','intrinsic_signed_bias'):add(model+'_'+family+'_'+name+'_mean3_'+attr,np.column_stack([field(model,family+f'-r{r}',attr,p)for r in range(3)]).mean(axis=1))
        add(model+'_knownparameter_oracle_RMS_m',[r.get('oracle_RMS_m',{}).get(model,np.nan)for r in records])
    matrix=np.column_stack(columns)
    if np.isfinite(matrix).all():prediction=dict(complete=True,comparison_blocked=False,objects=64,shared_bootstrap=paired_cluster_intervals(matrix,labels=names,seed=design.BOOTSTRAP_SEED,resamples=10000))
    else:prediction=dict(complete=False,comparison_blocked=True,objects=64,unresolved_entries=int((~np.isfinite(matrix)).sum()),metric_labels=names)
    return cal,prediction,primary


def terminal_metadata(digest,execution_sha,summary_sha):
    """Full metadata/byte barrier before opening ANY actual NPZ."""
    p=fresh.release_gate(digest)
    demand(p['models']==list(MODELS)and p['phases']==list(design.PHASES)and p['truth_labels']==list(design.TRUTH_LABELS)and p['target_labels']==list(design.TARGETS),'Exact fixed models/phases/six truth/four target catalog')
    e=json.loads((fresh.LOCAL/'execution.json').read_text());summary=json.loads((fresh.LOCAL/'summary-LOCAL.json').read_text())
    demand(sha(fresh.LOCAL/'execution.json')==execution_sha and sha(fresh.LOCAL/'summary-LOCAL.json')==summary_sha,'Exact terminal seals')
    demand(e.get('terminal')is True and e.get('protocol_SHA256')==digest and e['summary']==record(fresh.LOCAL/'summary-LOCAL.json')and e['release']==record(fresh.LOCAL/'release.json'),'Final aggregate/release/terminal binding')
    demand(len(e['workers'])==p['resources']['workers']and sorted(r['slot']for r in e['workers'])==list(range(p['resources']['workers'])),'All original worker slots')
    fresh.full.prefix.absent(e['controller_pid'])
    for worker in e['workers']:
        demand(worker['reaped']is True and isinstance(worker['exit_code'],int),'Every owned worker reaped; failures retained');fresh.full.prefix.absent(worker['pid'])
    demand(summary['protocol_SHA256']==digest and summary['objects']==64 and summary['expected_posteriors']==896 and summary['all64_denominators_retained']is True,'Original64/896 denominator')
    rows=[];committed={}
    for index in range(64):
        directory=fresh.LOCAL/f'object-{index:02d}';path=directory/'summary-LOCAL.json'
        if not path.exists():continue
        row=json.loads(path.read_text());demand(row['index']==index and row['protocol_SHA256']==digest,'Original object ID/protocol')
        for name,h in row['checkpoint_SHA256'].items():
            demand(Path(name).name==name and sha(directory/name)==h,'All object files bound before arrays');committed[str(directory/name)]=h
        rows.append(row)
    actual_dirs={p.name for p in fresh.LOCAL.glob('object-*')if p.is_dir()};demand(actual_dirs<={f'object-{i:02d}'for i in range(64)},'No substituted extra objects')
    rows=planned_records(rows)
    for worker in e['workers']:
        path=fresh.LOCAL/f"worker-{worker['slot']}"/'summary-LOCAL.json'
        if path.exists():
            demand(worker.get('summary')==record(path),'Terminal execution must hash each worker summary')
            w=json.loads(path.read_text());demand(w['slot']==worker['slot']and w['pid']==worker['pid']and w['protocol_SHA256']==digest,'Worker identity')
            expected=set(range(worker['slot'],64,p['resources']['workers']));seen=set()
            for obj in w['objects']:
                demand(obj['index']in expected and obj['index']not in seen,'Worker replacement/duplicate');seen.add(obj['index']);demand(obj['record']==record(fresh.LOCAL/f"object-{obj['index']:02d}"/'summary-LOCAL.json'),'Worker object record hash')
            if w.get('completed'):demand(seen==expected,'Completed worker cannot omit assigned objects')
    demand(summary['raw_calibration']==record(ROOT/summary['raw_calibration']['path']),'Raw calibration final hash')
    return p,e,summary,rows,committed


def commit(path,context,manifest):
    demand(str(path)in manifest and str(path.with_suffix('.commit.json'))in manifest,'Uncommitted archive/commit')
    demand(sha(path)==manifest[str(path)]and sha(path.with_suffix('.commit.json'))==manifest[str(path.with_suffix('.commit.json'))],'Changed committed archive/commit bytes')
    c=json.loads(path.with_suffix('.commit.json').read_text());demand(c['context']==context and c['raw_SHA256']==sha(path),'Archive context/hash')


def truth_array_checks(z,unit,*,n=4):
    from experiments.campaign_surface_p2_static_v02 import geometry,field_queries
    from experiments.campaign_surface_p2_study_design_v01 import observation_points
    rest,_,_,fixed,_=geometry(n)
    demand(np.array_equal(z['unit'],unit),'Truth unit/object mismatch')
    for name in('displacement_m','linear_displacement_m'):
        u=z[name];demand(u.shape==rest.shape and np.isfinite(u).all()and np.all(u[fixed]==0),'Full saved truth/linear field finite with exact fixed nodes')
    response=field_queries(n,z['displacement_m'],observation_points()).ravel()
    demand(z['response_m'].shape==(24,)and np.isfinite(z['response_m']).all()and np.max(abs(z['response_m']-response))<=2e-12,'Saved direct target query/field mismatch')
    return z['response_m'].copy()


def canonical_modal_identity(value,p):
    expected=p['cost_receipt']['memo']['model_fingerprint']
    demand(isinstance(expected,str)and len(expected)==64 and all(c in '0123456789abcdef'for c in expected)and value==expected,'Canonical source-bound full-modal fingerprint')
    return value


def object_arrays(row,p,digest,manifest):
    """Only saved arrays, no physical model, reference, RNG or oracle execution."""
    i=row['index'];directory=fresh.LOCAL/f'object-{i:02d}';answer=dict(row,posteriors={m:dict(v)for m,v in row['posteriors'].items()})
    if row.get('missing'):return answer,dict(index=i,missing=True,qualified=0)
    draws=directory/'draws-LOCAL.npz'
    if not draws.exists():demand(not row.get('physical_valid'),'Claimed valid object without draw');return answer,dict(index=i,missing_draw=True,qualified=0)
    commit(draws,dict(protocol_SHA256=digest,index=i,kind='freshparameter_and_independentnoise'),manifest)
    with np.load(draws,allow_pickle=False)as z:unit=z['unit'].copy();physical=z['physical_parameters'].copy();prefix=z['prefix_noise'].copy();noise=z['probe_noise'].copy()
    demand(unit.shape==(2,)and np.all((unit>=0)&(unit<1))and np.isfinite(unit).all()and prefix.shape==(24,)and noise.shape==(2,3,24)and np.isfinite(prefix).all()and np.isfinite(noise).all(),'Proper saved unit/noise shapes')
    demand(np.array_equal(physical,np.exp(np.log([2000.,5.])+unit*np.log([4.,6.]))),'Retained physical/unit parameter mapping')
    responses=[];truth_sha={};physical_rows=[]
    for label in design.TRUTH_LABELS:
        path=directory/(label+'-truth-LOCAL.npz');meta=path.with_suffix('.json')
        if not path.exists()or not meta.exists():demand(not row.get('physical_valid'),'Claimed valid object missing six truth records');continue
        context=dict(protocol_SHA256=digest,index=i,label=label,draws_SHA256=sha(draws));commit(path,context,manifest);d=json.loads(meta.read_text())
        demand(str(meta)in manifest and d['context']==context and d['raw_SHA256']==sha(path),'Direct truth diagnostics context')
        with np.load(path,allow_pickle=False)as z:
            responses.append(truth_array_checks(z,unit))
        truth_sha[label]=sha(path);physical_rows.append(d)
        if d['passed']:demand(d['original_free_residual_N']<=1e-10 and d['material_valid']and d['deformation']['minimum_J']>=.8 and d['deformation']['maximum_absolute_symmetric_strain_eigenvalue']<=.05 and d['equilibrium']['passed']and d['stability']['passed'],'Original physical gates in claimed passing truth')
    if len(responses)!=6:
        demand(not any(v.get('qualified')for values in row['posteriors'].values()for v in values.values()),'Posterior admitted without all direct targets');return answer,dict(index=i,incomplete_truth=True,truth_records=len(physical_rows),qualified=0)
    compare(row['physics']['loads'],physical_rows,'six physical records');demand(row['physical_valid']==all(r['passed']for r in physical_rows),'Object physical summary')
    responses=np.asarray(responses);target=responses[2:].ravel();observations={}
    for key in KEYS:
        values=[(design.PREFIX,responses[0]+prefix)]
        if key!='prefix':family,reset=key.split('-r');f=('normal','axial').index(family);values.append((design.PROBES[f],responses[f]+noise[f,int(reset)]))
        h=hashlib.sha256()
        for label,value in values:h.update(label.encode());h.update(np.ascontiguousarray(value,dtype='<f8').tobytes())
        observations[key]=h.hexdigest()
    qualified=0;fingerprints=set()
    for model in MODELS:
        backend='conditional_amplitude_PWL_v02'if model=='linear'else'nonlinear_conditional_reference_v03_MIS'
        recipe=hashlib.sha256(json.dumps(dict(kind=model,numerical_backend=backend,plan=p['plan'],targets=design.TARGETS,atlas=p['atlas_fingerprint']if model=='nonlinear'else None),sort_keys=True).encode()).hexdigest()
        for key in KEYS:
            r=row['posteriors'][model][key];path=directory/(model+'-'+key+'-posterior-LOCAL.json')
            if not path.exists():demand(not r.get('qualified'),'Qualified missing posterior');continue
            demand(str(path)in manifest,'Posterior file absent from object commit');saved=json.loads(path.read_text())
            for field,value in saved.items():compare(value,r.get(field),model+'/'+key+'/'+field)
            base=dict(protocol_SHA256=digest,index=i,phase=key,draws_SHA256=sha(draws))
            if not r.get('qualified'):demand(all(r['context'].get(k)==v for k,v in base.items()),'Unqualified context');continue
            demand(row['physical_valid']and r['physical_valid']and r['numerical_backend']==backend,'Same admitted physical/backend context')
            context=r['context'];expected=dict(base,recipe_SHA256=recipe,model_fingerprint=p['cost_receipt']['memo']['model_fingerprint'],observations_SHA256=observations[key],true_unit_SHA256=array_digest(unit));demand(context==expected,'Paired noise/model/recipe/trueunit binding/canonical modal fingerprint');fingerprints.add(canonical_modal_identity(context['model_fingerprint'],p))
            numerical_record_checks(r,model)
            raw=path.with_suffix('.npz');demand(str(raw)in manifest and sha(raw)==r['raw_SHA256'],'Lean posterior hash')
            with np.load(raw,allow_pickle=False)as z:metrics=lean_metrics(z,unit,target)
            for k,v in metrics.items():compare(v.tolist()if isinstance(v,np.ndarray)else v,r[k],model+'/'+key+'/'+k)
            answer['posteriors'][model][key]=dict(r,**metrics);qualified+=1
    demand(len(fingerprints)<=1,'Paired models must use same full linear modal fingerprint')
    oracle=directory/'oracle-predictions-LOCAL.npz'
    if oracle.exists():
        expected=dict(protocol_SHA256=digest,index=i,draws_SHA256=sha(draws),atlas_fingerprint=p['atlas_fingerprint'],model_fingerprint=p['cost_receipt']['memo']['model_fingerprint'],truth_SHA256=truth_sha)
        commit(oracle,expected,manifest);values={}
        with np.load(oracle,allow_pickle=False)as z:
            for model in MODELS:
                v=z[model+'_mean_m'];demand(v.shape==(96,)and np.isfinite(v).all(),'Knownparameter oracle shape');values[model]=float(np.sqrt(np.mean((v-target)**2)))
        compare(values,row['oracle_RMS_m'],'Knownparameter oracle');answer['oracle_RMS_m']=values
    elif row.get('completed'):raise ValueError('Completed six-truth object lacks independent oracle archive')
    return answer,dict(index=i,truth_records=len(physical_rows),qualified=qualified,all14_slots_retained=True,noise_contexts=7,verified_modal_fingerprints=sorted(fingerprints),oracle_modal_fingerprint=p['cost_receipt']['memo']['model_fingerprint']if oracle.exists()else None,PIT_scope='Bracket/source/context sanity only; no repeated continuous CDF evaluation')


def sources():
    pending=[SOURCE,TEST];seen=set()
    while pending:
        f=pending.pop()
        if f in seen:continue
        seen.add(f)
        for node in ast.walk(ast.parse((ROOT/f).read_text())):
            modules=[]
            if isinstance(node,ast.Import):modules=[x.name for x in node.names]
            elif isinstance(node,ast.ImportFrom)and node.level==0:modules=[node.module or '']+[(node.module+'.'+x.name)for x in node.names if node.module]
            for name in modules:
                rel=name.replace('.','/')+'.py'
                if(ROOT/rel).is_file()and rel not in seen:pending.append(rel)
    return {f:sha(ROOT/f)for f in sorted(seen)}


def audit(digest,execution_sha,summary_sha):
    demand(set(os.sched_getaffinity(0))=={18}and os.environ.get('CUDA_VISIBLE_DEVICES')==''and all(os.environ.get(k)=='1'for k in('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS')),'Audit CPU18/one thread/noGPU')
    demand(not(LOCAL/'result.json').exists(),'No audit overwrite')
    start=time.monotonic();result=dict(completed=False,integrity_passed=False,qualified=False,source_SHA256=sources(),no_PDE_or_posterior_replay=True,no_parameter_or_noise_redraw=True)
    def stopped(*_):raise TimeoutError('300s independent audit cap')
    previous=signal.signal(signal.SIGALRM,stopped);signal.alarm(300)
    try:
        p,e,s,rows,manifest=terminal_metadata(digest,execution_sha,summary_sha);audited=[];checks=[]
        for row in rows:
            a,c=object_arrays(row,p,digest,manifest);audited.append(a);checks.append(c)
            demand(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024<4*1024**3,'Own4GiB audit RSS')
        calibration,prediction,primary=aggregate(audited);compare(calibration,s['primary_calibration'],'Primary statistics');compare(prediction,s['paired_metrics'],'Shared-object bootstrap')
        with np.load(ROOT/s['raw_calibration']['path'],allow_pickle=False)as z:
            demand(set(z.files)==set(primary),'Raw primary memberships')
            for k,v in primary.items():demand(np.array_equal(z[k],v,equal_nan=True),'Raw coverage/PIT differs')
        count=sum(r.get('qualified',False)for row in audited for group in row['posteriors'].values()for r in group.values());blocked=calibration['comparison_blocked']or prediction['comparison_blocked']or count!=896
        demand(s['posterior_contexts_retained']==896 and s['qualified_posteriors']==count and s['physical_valid_objects']==sum(r.get('physical_valid',False)for r in audited)and s['comparison_blocked']==blocked,'Aggregate denominators/qualification flags')
        clean=bool(e.get('completed')is True and 'failure'not in e and all(w['exit_code']==0 for w in e['workers']))
        result.update(completed=True,integrity_passed=True,comparison_blocked=bool(blocked or not clean),execution_clean=clean,runner_comparison_blocked=blocked,objects=64,expected_posteriors=896,qualified_posteriors=count,physical_records_verified=sum(c.get('truth_records',0)for c in checks),object_checks=checks,protocol=record(fresh.LOCAL/'protocol.json'),release=record(fresh.LOCAL/'release.json'),execution=record(fresh.LOCAL/'execution.json'),summary=record(fresh.LOCAL/'summary-LOCAL.json'),sources_and_cost_fullatlas_origins_verified=True,closures_verified=True,aggregate_statistics_reproduced=True,expected_physical_records=384,canonical_modal_fingerprint=p['cost_receipt']['memo']['model_fingerprint'],verified_modal_fingerprints=sorted({f for c in checks for f in c.get('verified_modal_fingerprints',[])}),scope='Integrity/numerical-context audit, not a coverage pass criterion or independent PDE/CDF recalculation.')
    except BaseException as error:result['failure']=dict(type=type(error).__name__,message=str(error),traceback=traceback.format_exc());raise
    finally:
        signal.alarm(0);signal.signal(signal.SIGALRM,previous);result.update(wall_s=time.monotonic()-start,peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024);write(LOCAL/'result.json',result)
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--protocol-sha',required=True);p.add_argument('--execution-sha',required=True);p.add_argument('--summary-sha',required=True);a=p.parse_args();r=audit(a.protocol_sha,a.execution_sha,a.summary_sha);print(json.dumps(dict(completed=r['completed'],integrity_passed=r['integrity_passed'],comparison_blocked=r['comparison_blocked'])))
