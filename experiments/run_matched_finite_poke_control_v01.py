"""SOURCE-only matched finite-NH versus finite-linear one-poke control.

The SAME uniform eight-state prior (known K, two E values/four layouts) is used
for both models. It differs from the continuous model's prior. All continuous
choices seal before catalogs; all64 finite choices seal before mapped trial
outcomes. No PDE, RNG, dataset access or prepare executes on import.
"""
import argparse,hashlib,json,os,resource,shutil,time
import numpy as np
from experiments import run_noisy_poke_benchmark_v01 as continuous
from experiments.campaign_finite_probe_reference_v01 import FiniteProbeReference
ROOT=continuous.ROOT;LOCAL=ROOT/'runs/operator-dev/matched-finite-poke-control-v01'
SOURCE='experiments/run_matched_finite_poke_control_v01.py';TEST='tests/test_matched_finite_poke_control_v01.py'
FINITE_SOURCE='experiments/campaign_finite_probe_reference_v01.py';FINITE_TEST='tests/test_finite_probe_reference_v01.py'
MODELS=('NH','linear');FIXED_ACTION=5
CANDIDATE_LABELS=tuple(s['id']for s in continuous.design.physics.measurement_specs()if s['role']=='candidate')
LIMITS=dict(CPU=17,threads=1,GPU=False,RSS_bytes=4*1024**3,wall_s=1800,disk_reserve_bytes=30*1024**3,maximum_writes_bytes=128*1024**2,external_owned_watchdog_required=True)
sha,record,read,write=continuous.sha,continuous.record,continuous.read,continuous.write


def sources():return {name:sha(ROOT/name)for name in(SOURCE,TEST,FINITE_SOURCE,FINITE_TEST)}

def guard(start):
    if set(os.sched_getaffinity(0))!={17}or os.environ.get('CUDA_VISIBLE_DEVICES')!=''or any(os.environ.get(k)!='1'for k in('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS')):raise ValueError('CPU17/noGPU/one-thread finitecontrol')
    if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024>=LIMITS['RSS_bytes']:raise MemoryError('Finitecontrol4GiB RSS')
    if time.monotonic()-start>=LIMITS['wall_s']:raise TimeoutError('Finitecontrol1800s cap')
    if shutil.disk_usage(ROOT).free<LIMITS['disk_reserve_bytes']+LIMITS['maximum_writes_bytes']:raise OSError('30GiB filesystemfree plus projectedwrites')
    if LOCAL.exists()and sum(p.stat().st_size for p in LOCAL.rglob('*')if p.is_file())>LIMITS['maximum_writes_bytes']:raise OSError('Finitecontrol128MiB outputcap')


def barrier(parent_sha,start):
    guard(start);release=read(LOCAL/'release.json')
    if not(release.get('approved')is True and release.get('parent_protocol_SHA256')==parent_sha and release.get('source_SHA256')==sources()and release.get('resources')==LIMITS):raise ValueError('External root source/resource/parent release before anycatalog/data')
    p=continuous.validate(parent_sha);seals=continuous.all_seals_gate(parent_sha);inputs=continuous.inputs_gate(parent_sha)
    if len(seals)!=32 or[v['index']for v in seals]!=list(range(32)):raise ValueError('All32 continuouschoices rootclosed before finitecatalogconstruction')
    return p,seals,inputs


def finite_catalog(p,start):
    """Called only after barrier; public uniform catalog, no trialidentity."""
    specs=continuous.design.physics.measurement_specs();points=continuous.design.physics.measurement_points();loads=[v['id']for v in continuous.design.physics.load_specs()]
    if len(specs)!=60 or len(loads)!=11 or points.shape!=(16,3):raise ValueError('Exact frozen60channel/11load/query16 schema')
    if p['objects']!=continuous.design.objects():raise ValueError('Same knownK/interior2E catalog')
    captures=[dict(state=i,label=label,capture=continuous.capture(p,i,label))for i in range(8)for label in loads]
    continuous.verify([v['capture']for v in captures]) # all88bytes before first modelarray
    result={name:[]for name in MODELS}
    for i in range(8):
        query={name:{}for name in MODELS}
        for entry in captures[11*i:11*(i+1)]:
            guard(start)
            with np.load(ROOT/entry['capture']['path'],allow_pickle=False)as z:
                if not np.array_equal(z['query_points_m'],points):raise ValueError('Same public query16 in both finitephysics')
                query['NH'][entry['label']]=z['query_displacement_m'].copy()
                query['linear'][entry['label']]=continuous.physics_stage.geometry.field_queries(12,z['linear_displacement_m'],points)
        for name in MODELS:
            row=[]
            for spec in specs:
                j=next(k for k,point in enumerate(points)if np.array_equal(point,spec['point_m']));row.append(float(query[name][spec['load']][j]@np.asarray(spec['projection'])))
            result[name].append(row)
    arrays={name:np.asarray(value)for name,value in result.items()}
    if any(value.shape!=(8,60)or not np.isfinite(value).all()for value in arrays.values()):raise ValueError('Complete finite NH/linear eight-state same60labels')
    return arrays,captures,[s['id']for s in specs]


def model(array):
    a=np.asarray(array,float)
    if a.shape!=(8,60)or not np.isfinite(a).all():raise ValueError('All8 uniformhypotheses/60channels, no filtering')
    return FiniteProbeReference(a[:,:4],a[:,4:12],a[:,12:],sigma_m=continuous.design.physics.SIGMA_M)


def choose(parent_sha):
    start=time.monotonic();p,seals,inputs=barrier(parent_sha,start)
    if(LOCAL/'finite-choice-seals.json').exists()or(LOCAL/'catalogs-LOCAL.npz').exists():raise FileExistsError('No finitecatalog/choice retry')
    arrays,captures,labels=finite_catalog(p,start);catalog=LOCAL/'catalogs-LOCAL.npz';np.savez_compressed(catalog,NH=arrays['NH'],linear=arrays['linear'],labels=np.array(labels));catalog_record=record(catalog)
    out=dict(completed=False,pid=os.getpid(),parent_protocol=record(continuous.LOCAL/'protocol.json'),parent_release=record(continuous.LOCAL/'release.json'),root_release=record(LOCAL/'release.json'),source_SHA256=sources(),continuous32_choice_seals=seals,continuous_choice_execution=record(continuous.LOCAL/'execution-choices.json'),continuous_choice_root_closure=record(continuous.LOCAL/'root-closure-choices.json'),inputs=record(continuous.LOCAL/'inputs.json'),catalog=catalog_record,source_captures=captures,choices=[],candidate_labels=list(CANDIDATE_LABELS),fixed_action=FIXED_ACTION,fixed_action_label=CANDIDATE_LABELS[FIXED_ACTION],prior='Uniform8,knownKsqrt150,4layouts×E2500/6500; SAME bothmodels; differscontinuousprior',no_trial_identity_or_mapped_outcome_used=True)
    try:
        for name in MODELS:
            ref=model(arrays[name])
            for index in range(32):
                guard(start);entry=inputs['prefixes'][index]
                with np.load(ROOT/entry['record']['path'],allow_pickle=False)as z:observed=continuous.design.prefix_only(z['observed_prefix'])
                fingerprint=continuous.design.fingerprint(observed)
                if entry['index']!=index or fingerprint!=entry['observation_fingerprint']:raise ValueError('Exact sharedobservedprefix binding')
                choice=ref.choose(observed) # ONLY model+observedprefix, no generatingstate
                out['choices'].append(dict(model=name,index=index,prefix_input=entry['record'],observation_fingerprint=fingerprint,choice=choice,fixed_action=FIXED_ACTION))
                out.update(wall_s=time.monotonic()-start,peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024);write(LOCAL/'finite-choices-partial-LOCAL.json',out)
        if[(v['model'],v['index'])for v in out['choices']]!=[(name,i)for name in MODELS for i in range(32)]:raise ValueError('Every64finitechoice retained')
        out['completed']=True;guard(start);write(LOCAL/'finite-choice-seals.json',out)
    except BaseException as error:out['failure']=dict(type=type(error).__name__,message=str(error));write(LOCAL/'finite-choices-partial-LOCAL.json',out);raise
    return record(LOCAL/'finite-choice-seals.json')


def choice_check(choice):
    """Recompute frozen helper's scalar refinement/tie predicates, no replay."""
    prior=choice['prior_variance_m2'];rows=choice['actions'];tiny=np.finfo(float).tiny
    if not np.isfinite(prior)or prior<0 or len(rows)!=8 or[v['action']for v in rows]!=list(range(8)):raise ValueError('All8finiteaction/prior witnesses')
    for row in rows:
        levels=row['levels']
        if len(levels)!=5 or[v['order']for v in levels]!=[8,16,32,64,128]or any(not np.isfinite(v['risk_m2'])or v['risk_m2']<0 or v['prior_variance_m2']!=prior or v.get('all_hypotheses_retained')is not True for v in levels):raise ValueError('All originalfivepositive quadraturelevels/hypotheses')
        error=abs(levels[-1]['risk_m2']-levels[-2]['risk_m2']);risk=levels[-1]['risk_m2'];rounding=128*np.finfo(float).eps*max(prior,tiny)
        qualified=bool(np.isfinite(risk)and risk>=0 and risk<=prior+error+rounding and error<=1e-4*max(prior,tiny))
        if row['risk_m2']!=risk or row['refinement_indicator_m2']!=error or row['qualified']is not qualified:raise ValueError('Original finitequadrature flags')
    qualified=all(v['qualified']for v in rows);best=min(rows,key=lambda v:(v['risk_m2'],v['action']));ties=[v['action']for v in rows if v['risk_m2']-best['risk_m2']<=v['refinement_indicator_m2']+best['refinement_indicator_m2']]
    if choice['qualified']is not qualified or choice['numerical_ties']!=ties or choice['selected_action']!=(min(ties)if qualified else None):raise ValueError('Exact finitechoice qualification/tie/selection')
    return qualified


def sealed_choices(parent_sha,start):
    p,seals,inputs=barrier(parent_sha,start);path=LOCAL/'finite-choice-seals.json';value=read(path)
    if not(value.get('completed')is True and 'failure'not in value and value['source_SHA256']==sources()and value['parent_protocol']==record(continuous.LOCAL/'protocol.json')and value['parent_release']==record(continuous.LOCAL/'release.json')and value['root_release']==record(LOCAL/'release.json')and value['continuous32_choice_seals']==seals and value['continuous_choice_execution']==record(continuous.LOCAL/'execution-choices.json')and value['continuous_choice_root_closure']==record(continuous.LOCAL/'root-closure-choices.json')and value['inputs']==record(continuous.LOCAL/'inputs.json')and value.get('no_trial_identity_or_mapped_outcome_used')is True):raise ValueError('Complete exact64finitechoice causal/source seal')
    continuous.base.absent(value['pid']) # root must reap chooser before separate evaluator
    continuous.verify(value['source_captures']);continuous.verify(value['catalog'])
    if[(v['model'],v['index'])for v in value['choices']]!=[(name,i)for name in MODELS for i in range(32)]:raise ValueError('All64 memberships without replacements')
    wall=value.get('wall_s')
    if type(wall)not in(int,float)or not np.isfinite(wall)or not 0<=wall<=LIMITS['wall_s']or value.get('candidate_labels')!=list(CANDIDATE_LABELS)or value.get('fixed_action')!=5 or value.get('fixed_action_label')!=CANDIDATE_LABELS[5]:raise ValueError('Frozen finitewall/action-label metadata')
    for row in value['choices']:
        choice_check(row['choice'])
        if row['prefix_input']!=inputs['prefixes'][row['index']]['record']or row['observation_fingerprint']!=inputs['prefixes'][row['index']]['observation_fingerprint']or row['fixed_action']!=5:raise ValueError('Everyfinitechoice sameprefix/fixedaction binding')
    return p,inputs,value


def mse_metrics(moment,truth):
    mean=np.asarray(moment['mean']);truth=np.asarray(truth);variance=np.asarray(moment['variance'])
    if mean.shape!=(48,)or truth.shape!=(48,)or variance.shape!=(48,)or not all(np.isfinite(x).all()for x in(mean,truth,variance))or np.any(variance<0):raise ValueError('Finite48target predictive moments/truth')
    return dict(MSE_m2=float(np.mean((mean-truth)**2)),posterior_variance_m2=float(variance.mean()))


def policy_metrics(before,counter,selected,qualified):
    if len(counter)!=8 or[v['action']for v in counter]!=list(range(8)):raise ValueError('All8 resetcounterfactuals')
    out=dict(before=before,fixed=counter[FIXED_ACTION],uniform_random={k:float(np.mean([v[k]for v in counter]))for k in('MSE_m2','posterior_variance_m2')})
    if qualified:
        if type(selected)is not int or not 0<=selected<8:raise ValueError('Qualified finitechoice selectedaction')
        out['adaptive']=counter[selected]
    elif selected is not None:raise ValueError('Unqualified choice cannotselect action')
    return out


def summarize(rows):
    if len(rows)!=64 or{(r['model'],r['index'])for r in rows}!={(name,i)for name in MODELS for i in range(32)}:raise ValueError('All64modeltrial memberships retained')
    for r in rows:
        if r['object_index']!=r['index']%8 or r['replicate']!=r['index']//8 or len(r['counterfactuals'])!=8 or[v['action']for v in r['counterfactuals']]!=list(range(8)):raise ValueError('All256 resetcounterfactuals/model and exact32trialidentities')
        if any(not np.isfinite(a['MSE_m2'])or a['MSE_m2']<0 for a in r['counterfactuals']):raise ValueError('Allfinite nonnegative counterfactual losses')
        if r['fixed_action']!=5 or r['policies']!=policy_metrics(r['before'],r['counterfactuals'],r['selected_action'],r['choice_qualified']):raise ValueError('Recomputed exactadaptive/fixed/uniform counterfactual policies')
    complete=all(r['choice_qualified']for r in rows);out=dict(retained_trials_per_model=32,retained_counterfactuals_per_model=256,qualified_choices=sum(r['choice_qualified']for r in rows),all64_retained=True,comparison_blocked=not complete,fixed_action=5,fixed_action_label=CANDIDATE_LABELS[5],finite_prior_differs_continuous_prior=True,population_or_calibration_claim=False)
    if not complete:return out
    objectmeans={}
    for name in MODELS:
        objects=[]
        for i in range(8):
            group=[r for r in rows if r['model']==name and r['object_index']==i]
            if len(group)!=4 or{r['replicate']for r in group}!=set(range(4)):raise ValueError('Equal4withinobjectnoise trials')
            objects.append({policy:float(np.mean([r['policies'][policy]['MSE_m2']for r in group]))for policy in('before','adaptive','fixed','uniform_random')})
        objectmeans[name]=objects
    out.update(equal_object_mean_MSE_m2={name:{policy:float(np.mean([r[policy]for r in objectmeans[name]]))for policy in('before','adaptive','fixed','uniform_random')}for name in MODELS},NH_minus_linear_MSE_m2={policy:float(np.mean([a[policy]-b[policy]for a,b in zip(objectmeans['NH'],objectmeans['linear'])]))for policy in('before','adaptive','fixed','uniform_random')},aggregation='Mean4noise perobject, then equalmean8objects; exactuniform8counterfactual MSE, never averageRMSE',scope='Matchedfinitepair isolates responsephysics withsame finiteprior/method/data; adaptiveeffect includes resultingmodel-dependentchoice. Finitevscontinuous differences also include prior/knownK restrictions, not an isolatedphysics comparison.')
    return out


def evaluate(parent_sha):
    start=time.monotonic();p,inputs,sealed=sealed_choices(parent_sha,start)
    # No trialidentity, poke-noise or mappedtruth numeric access before64seal.
    budget_start=start-sealed['wall_s'];guard(budget_start)
    target=LOCAL/'evaluation-results-LOCAL.json'
    if target.exists():raise FileExistsError('No finiteevaluation retry')
    with np.load(ROOT/sealed['catalog']['path'],allow_pickle=False)as z:arrays={name:z[name].copy()for name in MODELS}
    with np.load(ROOT/inputs['poke_noise']['path'],allow_pickle=False)as z:noise=z['poke_noise'].copy()
    if noise.shape!=(4,8,8)or not np.isfinite(noise).all():raise ValueError('Same frozen32×8 poke-noise values; no redraw')
    rows=[]
    try:
        for name in MODELS:
            ref=model(arrays[name])
            for index in range(32):
                guard(budget_start);trial=p['trials'][index];i,r=trial['object_index'],trial['replicate']
                if(trial['index'],i,r)!=(index,index%8,index//8):raise ValueError('Exact frozen trialidentity only after all64seal')
                choice=sealed['choices'][MODELS.index(name)*32+index]
                with np.load(ROOT/inputs['prefixes'][index]['record']['path'],allow_pickle=False)as z:observed=continuous.design.prefix_only(z['observed_prefix'])
                if continuous.design.fingerprint(observed)!=choice['observation_fingerprint']:raise ValueError('Same prefix at finiteupdate')
                base=ref.prefix_posterior(observed);truth=arrays['NH'][i,12:];before=mse_metrics(ref.moments(base),truth);counter=[];means=[];weights=[]
                for action in range(8):
                    outcome=float(arrays['NH'][i,4+action]+noise[r,i,action]);posterior=ref.after_one_poke(base,action,outcome) # fresh SAMEprefix, not previouspoke
                    moment=ref.moments(posterior);counter.append(dict(action=action,**mse_metrics(moment,truth),fresh_prefix_plus_ONE_scalar=True));means.append(moment['mean']);weights.append(np.exp(posterior))
                qualified=choice['choice']['qualified']is True;policies=policy_metrics(before,counter,choice['choice']['selected_action'],qualified)
                raw=LOCAL/'posteriors-LOCAL'/f'{name}-{index:02d}.npz';raw.parent.mkdir(exist_ok=True);np.savez_compressed(raw,mean_m=np.array(means),weights=np.array(weights),before_mean_m=ref.moments(base)['mean'])
                rows.append(dict(model=name,index=index,object_index=i,replicate=r,choice_qualified=qualified,choice_seal=record(LOCAL/'finite-choice-seals.json'),selected_action=choice['choice']['selected_action'],fixed_action=5,before=before,counterfactuals=counter,policies=policies,posterior_archive=record(raw)))
                write(LOCAL/'evaluation-partial-LOCAL.json',dict(completed=False,rows=rows))
        summary=summarize(rows);guard(budget_start);write(target,dict(completed=True,parent_protocol=record(continuous.LOCAL/'protocol.json'),finite_choices=record(LOCAL/'finite-choice-seals.json'),source_SHA256=sources(),rows=rows,summary=summary,wall_s=time.monotonic()-start,combined_owned_wall_s=time.monotonic()-budget_start,peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024));write(LOCAL/'compact-summary.json',summary)
    except BaseException as error:write(LOCAL/'evaluation-partial-LOCAL.json',dict(completed=False,rows=rows,failure=dict(type=type(error).__name__,message=str(error))));raise
    return record(target)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['choose','evaluate']);ap.add_argument('--parent-sha',required=True);a=ap.parse_args();print(json.dumps(choose(a.parent_sha)if a.action=='choose'else evaluate(a.parent_sha)))
