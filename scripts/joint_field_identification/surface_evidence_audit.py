"""Independent saved-evidence integrity audit; never treats advantage as correctness.

--watch refreshes every <=60 seconds until all declared jobs finish or deadline.
Only this audit's reports are written. Frozen experimental sources are untouched.
"""
from pathlib import Path
import argparse, collections, csv, datetime, hashlib, itertools, json, time
import numpy as np
from . import surface as bank
ROOT=bank.ROOT
REPORT=ROOT/'reports/joint-field-surface-v04'
RUN=ROOT/'runs/joint-field-surface-v04'
POLICIES=('active','fixed','fixed-view','fixed-probe','random')
POPS=('matched','continuous-coarse','offgrid-fine','force-prior')
METRICS=('rmse_mm','hidden_rmse_mm','changed_load_rmse_mm','material_relative_rmse','force_rmse_N','coverage90','material_coverage90','force_coverage90','interval_width_mm')
_HASH={}
def sha(path):
    p=Path(path);s=p.stat();key=(str(p),s.st_size,s.st_mtime_ns)
    if key not in _HASH:
        h=hashlib.sha256()
        with p.open('rb') as f:
            for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
        _HASH[key]=h.hexdigest()
    return _HASH[key]
def read(path):return json.loads(Path(path).read_text())
def save(path,doc):
    p=Path(path);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(doc,indent=2,allow_nan=False));tmp.replace(p)
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def jobname(j):return f"{j['shape']}-{j['population']}-case{j['case']:03d}-{j['stage']}-n{j['particles']}-s{j['seed']}"
def finite_values(x):
    if isinstance(x,dict):return all(finite_values(v) for v in x.values())
    if isinstance(x,list):return all(finite_values(v) for v in x)
    return not isinstance(x,(float,int)) or np.isfinite(x)
def main_audit():
    protocol=read(REPORT/'protocol.json');ph=sha(REPORT/'protocol.json');issues=[];pending=[];checks=collections.Counter();rows=[];counts={};truth_cache={}
    def check(condition,label,context=''):
        checks[label]+=1
        if not condition:issues.append(dict(check=label,context=str(context)))
    def hashes(mapping,label):
        for file,expected in mapping.items():check((ROOT/file).exists() and sha(ROOT/file)==expected,label,file)
    hashes(protocol['source_sha256'],'frozen_main_source')
    for shape in protocol['shapes']:
        bp=RUN/f'{shape}.npz';check(sha(bp)==protocol['bank_sha256'][shape],'frozen_bank',shape)
        data=dict(np.load(bp));fixed=read(REPORT/shape/'fixed-schedules.json')
        check(fixed==protocol['fixed_schedules'][shape],'frozen_development_schedule',shape)
        dev=np.load(RUN/shape/'development.npz');check(int(dev['seed'])==121001+(shape=='step'),'development_seed',shape)
        for pop in POPS:
            count=protocol['case_counts'][pop];tp=RUN/shape/f'{pop}-truth.npz'
            if not tp.exists():pending.append(str(tp.relative_to(ROOT)));continue
            truth=dict(np.load(tp));truth_cache[(shape,pop)]=truth
            seed=131001+10000*(shape=='step')+{'matched':0,'offgrid-fine':1000,'force-prior':2000,'continuous-coarse':3000}[pop]
            context=f'{shape}/{pop}'
            check(int(truth['seed'])==seed and str(truth['kind'])==pop and str(truth['protocol_sha256'])==ph,'truth_identity',context)
            check(len(truth['theta'])==count,'truth_count',context)
            check(all(np.all(np.isfinite(v)) for k,v in truth.items() if np.issubdtype(v.dtype,np.number)),'truth_finite',context)
            rng=np.random.default_rng(seed);idx=rng.integers(len(data['theta']),size=count)
            th=data['theta'][idx] if pop in ('matched','force-prior') else rng.uniform(bank.BOUNDS[:,0],bank.BOUNDS[:,1],size=(count,5))
            check(np.array_equal(idx,truth['indices']) and np.array_equal(th,truth['theta']),'truth_rng_theta',context)
            K=bank.load_cov()
            if pop=='force-prior':
                r=np.sqrt(3)*np.abs(bank.TIMES[:,None]-bank.TIMES[None,:])/.10;K=bank.FORCE_STD**2*np.kron((1+r)*np.exp(-r),np.eye(2))
            force=rng.normal(size=(count,16))@np.linalg.cholesky(K).T
            check(np.array_equal(force,truth['force']),'truth_rng_force',context)
            yn=rng.normal(scale=bank.NOISE,size=(count,24));an=rng.normal(scale=bank.NOISE,size=(count,16))
            if pop in ('matched','force-prior'):
                subset={k:v[idx] for k,v in data.items() if k in ('theta','unit','probe','material')};subset['points']=data['points'];m=bank.maps(subset)
                check(np.allclose(truth['y'],np.einsum('hof,hf->ho',m['H'],force)+yn,atol=1e-15,rtol=0),'observed_noise_exact_rng',context)
                predicted=np.stack([np.einsum('hf,hf->h',a['h'],force)+a['offset'] for a in m['actions']],1)
                check(np.allclose(truth['measurements'],predicted+an,atol=1e-15,rtol=0),'action_noise_exact_rng',context)
                check(np.allclose(truth['target'],np.einsum('hdf,hf->hd',m['T'],force)+m['offset'],atol=1e-15,rtol=0),'target_no_noise_identity',context)
            paths=sorted((REPORT/shape/pop).glob('case-*.json'));ids=[]
            for path in paths:
                try:d=read(path)
                except json.JSONDecodeError:pending.append(str(path.relative_to(ROOT)));continue
                i=d['case'];ids.append(i);ctx=f'{context}/{i}'
                check(d['scene']==shape and d['population']==pop and d['protocol_sha256']==ph and d['seed']==seed+1000000+100*i,'case_identity',ctx)
                check(finite_values(d),'case_metrics_finite',ctx)
                rr=d['rows'];check(len(rr)==30,'case_row_count',ctx)
                for pol in POLICIES:
                    pr=sorted([r for r in rr if r['policy']==pol],key=lambda r:r['step'])
                    check([r['step'] for r in pr]==list(range(5)),'policy_steps',ctx+'/'+pol)
                    actions=[r['action'] for r in pr if r['step']>0]
                    check(len(actions)==4 and len(set(actions))==4 and all(0<=a<16 for a in actions),'unique_valid_actions',ctx+'/'+pol)
                    if pol.startswith('fixed'):check(actions==fixed[pol]['actions'],'fixed_actions',ctx+'/'+pol)
                    if pol=='random':check(actions==np.random.default_rng(d['seed']+901).permutation(16)[:4].tolist(),'random_actions',ctx)
                    for r in pr:
                        check(all(k in r for k in METRICS[:5]),'required_metrics',ctx+'/'+pol)
                        if r['step']>0:check(r['action_kind']==('probe' if r['action']>=10 else 'view'),'action_kind',ctx)
                chosen=[]
                for a in d['acquisition']:
                    check(a['chosen']==min(a['scores'],key=lambda x:x['expected_risk'])['action'],'actual_argmin_action',ctx)
                    check({s['action'] for s in a['scores']}==set(range(16))-set(chosen),'candidate_exclusion',ctx)
                    check(all(s['seed']==d['seed']+a['step'] and s['n_samples']==128 for s in a['scores']),'planner_rng',ctx)
                    chosen.append(a['chosen'])
                actual=[r['action'] for r in rr if r['policy']=='active' and r['step']>0]
                check(chosen==actual,'active_record_identity',ctx)
                rows.extend(dict(shape=shape,population=pop,case=i,**r) for r in rr)
            check(len(ids)==len(set(ids)) and set(ids)<=set(range(count)),'case_unique_range',context)
            counts[context]=dict(completed=len(ids),expected=count,missing=sorted(set(range(count))-set(ids)))
    # Contract identity is separately unit tested; reference its artifact instead of claiming a new FEM solve.
    pts=bank.points();hidden=set(np.flatnonzero(pts[:,1]>0))-{3,11}
    actionpoints={a['point'] for a in bank.action_specs() if a['kind']=='view'}
    initialpoints={1,5,9,13}
    check(not hidden & (actionpoints|initialpoints),'surface_target_points_separate')
    check(set(s['x'] for s in bank.probe_specs()[:6]).isdisjoint(set(s['x'] for s in bank.probe_specs()[6:])),'surface_reset_loads_separate')
    check(all(abs(np.dot(a['projection'],a['camera_direction']))<1e-12 for a in bank.action_specs()),'projection_camera_orthogonality')
    check(np.all(np.isclose(np.abs(pts[:,1]),.03)),'actual_surface_points')

    figure=ROOT/'output/joint-field-surface-v04/figure-manifest.json'
    figure_status={}
    if figure.exists():
        fd=read(figure);hashes(fd['sources'],'figure_source_provenance');check(fd.get('version')=='joint-field-surface-v04' and fd['actual_geometry_scale'] is True and fd['known_surface_marker_correspondences'] is True,'figure_final_true_scale')
        check('case000' in fd['selection'],'figure_predeclared_case0')
        for shape,parity in fd['full_node_cache_checks'].items():
            check(parity['parameter_grid_identical'] and parity['unit_point_interpolation_max_error_m_per_N']<1e-12 and parity['probe_point_interpolation_max_error_m_per_N']<1e-12,'surface_nodal_identity',shape)
        figure_status=dict(selection=fd['selection'],scope=fd['scope'],errors=fd['changed_load_surface_error_rms_mm'],nodal_identity='Prior independent figure audit; this audit validates immutable nodal archive provenance, not a new physics solve.')
    # Recompute saved summary independently from CSV. Race/stale snapshots are pending, not fabricated mismatches.
    csvpath=REPORT/'all-case-metrics.csv';sp=REPORT/'summary.json';aggregate={}
    if csvpath.exists() and sp.exists():
        csvrows=list(csv.DictReader(csvpath.open()));summary=read(sp);groups=collections.defaultdict(list)
        for r in csvrows:groups[(r['population'],r['policy'],int(r['step']))].append(r)
        errors=[]
        rowkey=lambda r:(r['shape'],r['population'],int(r['case']),r['policy'],int(r['step']))
        originals={rowkey(r):r for r in rows}
        for r in csvrows:
            original=originals.get(rowkey(r))
            if original is None:pending.append('CSV contains a newly completed case after raw snapshot; retry on next refresh')
            if original is not None:
                check(all(str(v)==r.get(k) or (isinstance(v,(int,float)) and r.get(k,'')!='' and np.isclose(float(r[k]),v,atol=1e-14,rtol=1e-12)) for k,v in original.items()),'csv_row_identity',rowkey(r))
        for s in summary['summary']:
            rr=groups[(s['population'],s['policy'],s['step'])]
            if len(rr)!=s['n']:errors.append(dict(group=[s['population'],s['policy'],s['step']],metric='n'))
            for k,v in s.items():
                if k in ('population','policy','step','n'):continue
                vv=[float(r[k]) for r in rr if r.get(k,'')!='']
                if not vv or not np.isclose(np.mean(vv),v,atol=1e-12,rtol=1e-11):errors.append(dict(group=[s['population'],s['policy'],s['step']],metric=k))
        for comparison in summary.get('comparisons',[]):
            def policy_values(policy):
                return {(r['shape'],r['case']):float(r[comparison['metric']]) for r in csvrows if r['population']==comparison['population'] and r['policy']==policy and int(r['step'])==4}
            av,bv=policy_values(comparison['first']),policy_values(comparison['second']);ids=sorted(set(av)&set(bv))
            differences=np.array([av[i]-bv[i] for i in ids])
            if len(ids)!=comparison['n'] or not np.isclose(differences.mean(),comparison['mean_difference'],atol=1e-12,rtol=1e-11):errors.append(dict(comparison=comparison['metric'],population=comparison['population'],second=comparison['second']))
        aggregate=dict(csv_rows=len(csvrows),current_case_rows=len(rows),summary_groups=len(summary['summary']),arithmetic_discrepancies=errors,snapshot_current=len(csvrows)==len(rows))
        if errors:pending.append('CSV/summary arithmetic snapshot differs; rerun aggregation or inspect discrepancies')
        if len(csvrows)!=len(rows):pending.append('CSV summary predates latest case records')
    continuous=continuous_audit(check,hashes,truth_cache,rows)
    hierarchy={};hp=ROOT/'reports/joint-field-hierarchy-surface-v05/protocol.json'
    if hp.exists():
        hd=read(hp);hashes(hd['source_sha256'],'hierarchy_frozen_source');hc={}
        for shape,n in hd['case_counts'].items():
            paths=list((hp.parent/shape).glob('case-*.json'));hc[shape]=dict(completed=len(paths),expected=n)
        for shape,n in hd['case_counts'].items():
            for path in (hp.parent/shape).glob('case-*.json'):
                d=read(path)
                check(d['protocol_sha256']==sha(hp) and d['shape']==shape and 0<=d['case']<n,'hierarchy_case_identity',str(path))
                check(finite_values(d),'hierarchy_finite',str(path))
                check({(r['model'],r['condition']) for r in d['rows']}==set(itertools.product(hd['models'],('initial','probes','views'))),'hierarchy_model_conditions',str(path))
        hierarchy=dict(protocol_sha256=sha(hp),counts=hc,scope=hd['scope'],separate_from_v01=True)
    finite_complete=all(c['completed']==c['expected'] for c in counts.values()) and len(counts)==8
    done=finite_complete and continuous['complete']
    doc=dict(updated_utc=now(),complete=done,integrity_pass=not issues,checks=dict(checks),issues=issues,pending=pending,case_counts=counts,aggregate=aggregate,continuous=continuous,hierarchy=hierarchy,figure=figure_status,
        limits=['Continuous/fine truth RNG and source fingerprints checked; noise/forward identities replayed for matched and force-prior populations only.','Numerical flags never exclude cases. No advantage, coverage, or positive result required.','Coverage is empirical marginal coverage; SMC sensitivity is not a convergence proof.'])
    save(REPORT/'evidence-audit.json',doc)
    lines=[f"Evidence audit at {doc['updated_utc']}",f"Complete: {done}; integrity checks passed: {not issues}",f"Finite cases: {sum(c['completed'] for c in counts.values())}/320; saved rows: {len(rows)}",f"Continuous jobs: {continuous['completed']}/288; sensitivity pairs: {continuous['comparison_count']}; descriptive flags: {continuous['flagged_comparisons']}",f"Integrity issues: {len(issues)}; pending snapshot items: {len(pending)}",'No result is excluded for poor accuracy, coverage, or numerical sensitivity.','Continuous summary uses matched case identities for each model/stage and reports n; do not compare partial stages as if cohorts are identical.','Details: evidence-audit.json, continuous-summary.json/.csv, continuous-numerical-comparisons.json.']
    (REPORT/'evidence-audit.txt').write_text('\n'.join(lines)+'\n')
    return doc


def continuous_audit(check,hashes,truths,finite_rows):
    manifestpath=REPORT/'continuous/manifest.json';manifest=read(manifestpath);hashes(manifest['source_sha256'],'continuous_frozen_source');mh=sha(manifestpath)
    complete=[];pending=[];groups=collections.defaultdict(list);summaries=[];comparisons=[]
    for j in manifest['allowed_jobs']:
        name=jobname(j);p=REPORT/'continuous'/f'{name}.json'
        if not p.exists():pending.append(name);continue
        try:d=read(p)
        except json.JSONDecodeError:pending.append(name);continue
        if not d.get('complete'):pending.append(name);continue
        check(d['job']==j and d['seed']==j['seed'] and d['temperature']==1,'continuous_job_identity',name)
        check(finite_values(d),'continuous_metrics_finite',name)
        pr=d['protocol'];check(pr['manifest_sha256']==mh,'continuous_manifest',name);hashes(pr['source_sha256'],'continuous_job_sources')
        shape,pop,i,stage=j['shape'],j['population'],j['case'],j['stage'];tp=RUN/shape/f'{pop}-truth.npz';rp=REPORT/shape/pop/f'case-{i:03d}.json'
        check(pr['data_sha256']==sha(tp) and pr['record_sha256']==(None if stage=='initial' else sha(rp)),'continuous_evidence_fingerprints',name)
        rec=read(rp);actions=[] if stage=='initial' else [r['action'] for r in rec['rows'] if r['policy']==stage and r['step']>0]
        check(d['actions']==actions,'continuous_policy_evidence',name)
        truth=truths[(shape,pop)];obs=np.r_[truth['y'][i],truth['measurements'][i,actions]]
        check(np.array_equal(obs,pr['observations']) and pr['noise_m']==bank.NOISE,'continuous_observation_identity',name)
        complete.append(d)
        if pop!='continuous-coarse':groups[(shape,pop,i,stage)].append(d)
    for key,dd in groups.items():
        for a,b in itertools.combinations(dd,2):
            aa=np.load(ROOT/a['prediction_archive']);bb=np.load(ROOT/b['prediction_archive'])
            cdf=float(np.max(np.abs(aa['cdf']-bb['cdf'])));rms=float(np.sqrt(np.mean((aa['target_mean']-bb['target_mean'])**2))*1000)
            comparisons.append(dict(shape=key[0],population=key[1],case=key[2],stage=key[3],first=jobname(a['job']),second=jobname(b['job']),max_cdf_difference=cdf,target_mean_rms_difference_mm=rms,flag_cdf=cdf>.1,flag_prediction=rms>.01,log_evidence_difference=a['log_evidence']-b['log_evidence']))
    for pop in POPS:
      for stage in ('initial','active','fixed'):
        dd=[d for d in complete if d['job']['population']==pop and (pop=='continuous-coarse' or (d['job']['particles']==1024 and d['job']['seed']==171001)) and d['job']['stage']==stage]
        if not dd:continue
        for model in ('continuous-smc','finite-dictionary'):
            if model=='continuous-smc':values=[d['metrics'] for d in dd]
            else:
                ids={(d['job']['shape'],d['job']['case']) for d in dd};values=[r for r in finite_rows if r['population']==pop and (r['shape'],r['case']) in ids and r['policy']==('active' if stage=='initial' else stage) and r['step']==(0 if stage=='initial' else 4)]
            row=dict(population=pop,stage=stage,model=model,n=len(values))
            for metric in METRICS:
                vv=[r[metric] for r in values if metric in r]
                if vv:row[metric]=float(np.mean(vv))
            summaries.append(row)
    states=[]
    for p in (REPORT/'continuous').glob('supervisor-*.json'):
        d=read(p);states.append(dict(shard=d['shard'],status=d['status'],current=d.get('current'),failures=d.get('failures',[]),heartbeat_utc=d.get('heartbeat_utc')))
    dispatcherpath=REPORT/'continuous/dynamic-dispatch.json';dispatcher={}
    if dispatcherpath.exists():
        dispatch=read(dispatcherpath)
        check(dispatch['manifest_sha256']==mh,'dynamic_dispatch_manifest')
        dispatcher={k:dispatch.get(k) for k in ['pid','status','updated_utc','source_sha256','running','failures','exception']}
        dispatcher['completed_count']=len(dispatch.get('completed',[]));dispatcher['pending_count']=len(dispatch.get('pending',[]))
    c=dict(updated_utc=now(),expected=len(manifest['allowed_jobs']),completed=len(complete),pending_count=len(pending),complete=len(complete)==len(manifest['allowed_jobs']),main_completed=sum(d['job']['population']=='continuous-coarse' for d in complete),diagnostic_completed=sum(d['job']['population']!='continuous-coarse' for d in complete),supervisors=states,dispatcher=dispatcher,
        sensitivity_thresholds=dict(max_cdf_difference=.1,target_mean_rms_difference_mm=.01,interpretation='Descriptive flags declared in audit request before diagnostic comparison; retain all results, not exclusion/convergence criteria.'),comparison_count=len(comparisons),flagged_comparisons=sum(c['flag_cdf'] or c['flag_prediction'] for c in comparisons),summary=summaries)
    save(REPORT/'continuous-summary.json',c);save(REPORT/'continuous-numerical-comparisons.json',dict(updated_utc=now(),comparisons=comparisons,thresholds=c['sensitivity_thresholds']))
    if summaries:
        fields=list(dict.fromkeys(k for r in summaries for k in r))
        with (REPORT/'continuous-summary.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(summaries)
    return c


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--watch',action='store_true');ap.add_argument('--interval',type=float,default=60);a=ap.parse_args()
    deadline=datetime.datetime.fromisoformat(read(REPORT/'protocol.json')['deadline_utc'])
    while True:
        try:
            d=main_audit();print(json.dumps(dict(updated_utc=d['updated_utc'],complete=d['complete'],integrity_pass=d['integrity_pass'],issues=len(d['issues']),continuous_completed=d['continuous']['completed'],pending=d['pending'])),flush=True)
        except Exception as e:
            save(REPORT/'evidence-audit-exception.json',dict(updated_utc=now(),type=type(e).__name__,message=str(e)))
            if a.watch and isinstance(e,(json.JSONDecodeError,FileNotFoundError)) and datetime.datetime.now(datetime.timezone.utc)<deadline:
                print(json.dumps(dict(transient_snapshot_error=str(e),time=now())),flush=True);time.sleep(min(60,max(1,a.interval)));continue
            raise
        if not a.watch or d['complete'] or datetime.datetime.now(datetime.timezone.utc)>=deadline:break
        time.sleep(min(60,max(1,a.interval)))
if __name__=='__main__':main()
