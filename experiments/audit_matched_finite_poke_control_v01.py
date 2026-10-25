"""Independent terminal audit of the matched uniform-eight finite control.

Source preparation has no data access. At execution the continuous32/finite64
choices, both external writer closures and ALL bound archives precede any NPZ.
Posterior arithmetic is direct Gaussian likelihood/logsumexp, not the producer
posterior helper. Quadrature risks remain source-bound producer witnesses.
"""
import argparse,os,resource,shutil,sys,time
from pathlib import Path
import numpy as np
from scipy.special import logsumexp
from experiments import run_matched_finite_poke_control_v01 as run
ROOT=run.ROOT;LOCAL=run.LOCAL;c=run.continuous
SOURCE='experiments/audit_matched_finite_poke_control_v01.py'
TEST='tests/test_audit_matched_finite_poke_control_v01.py'
OPERATOR=ROOT/'runs/operator-dev/noisy-poke-operator-v01'
LIMITS=dict(CPU=16,threads=1,GPU=False,RSS_bytes=4*1024**3,wall_s=1800,disk_reserve_bytes=30*1024**3,maximum_writes_bytes=16*1024**2,external_owned_watchdog_required=True)
TOLERANCES=dict(weight_absolute=2e-12,mean_absolute_m=2e-12,scalar_relative=1e-9,scalar_absolute_m2=1e-22)

def need(ok,message):
    if not ok:raise ValueError(message)
def sources():return {p:run.sha(ROOT/p)for p in(SOURCE,TEST)}
def guard(start):
    need(set(os.sched_getaffinity(0))=={16}and os.environ.get('CUDA_VISIBLE_DEVICES')==''and all(os.environ.get(k)=='1'for k in('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS')),'Audit CPU16/noGPU/one-thread')
    if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024>=LIMITS['RSS_bytes']:raise MemoryError('Audit4GiB RSS')
    if time.monotonic()-start>=LIMITS['wall_s']:raise TimeoutError('Audit1800s')
    if shutil.disk_usage(ROOT).free<LIMITS['disk_reserve_bytes']+LIMITS['maximum_writes_bytes']:raise OSError('Audit30GiB free plus projectedwrites')
def bound(rec):
    path=ROOT/rec['path'];need(run.record(path)==rec,'Exact bound archive bytes/path');return path

def close_numbers(actual,expected,*,atol=1e-22,rtol=1e-9):
    a=np.asarray(actual,float);b=np.asarray(expected,float)
    need(a.shape==b.shape and np.isfinite(a).all()and np.isfinite(b).all()and np.allclose(a,b,atol=atol,rtol=rtol),'Independent numerical reproduction')

def closure(phase,parent_sha,result):
    """Verify external owned wait/reap, not a claim about unified-exec."""
    rec=run.record(LOCAL/f'root-closure-{phase}.json');v=run.read(bound(rec));name='finite_'+phase
    need(all(v.get(k)is True for k in('root_reaped','root_owned_supervisor','external_subprocess_wait','all_owned_PID_and_groups_absent'))and v.get('root_exit_code')==0 and v.get('parent_protocol_SHA256')==parent_sha and v.get('source_SHA256')==run.sources()and v.get('result')==result,'Exact finite external root closure')
    approval=run.record(OPERATOR/'root-preapproval.json');need(v.get('operator_preapproval')==approval,'Exact operator preapproval')
    a=run.read(bound(approval));need(a.get('approved')is True and a.get('schema')=='noisy-poke-operator-preapproval-v01'and a.get('stage_caps',{}).get('finite_audit')==[16,1800,4*1024**3],'Approved operator procedure and external audit watchdog')
    pins=a['source_SHA256'];need(all(pins.get(k)==h for k,h in{**run.sources(),**sources(),**c.sources()}.items()),'Auditor/test and numerical procedure prepinned before execution')
    for path,h in pins.items():need(run.sha(ROOT/path)==h,'Current operator precommitted source bytes')
    stage=run.record(OPERATOR/'stages'/name/'execution.json');need(v.get('operator_stage_execution')==stage,'Canonical operator stage commitment');s=run.read(bound(stage))
    outputs=[result]if phase=='choose'else[result,run.record(LOCAL/'compact-summary.json')]
    need(s.get('phase')==name and s.get('parent_protocol_SHA256')==parent_sha and all(s.get(k)is True for k in('completed','reaped'))and s.get('exit_code')==s.get('cleanup_exit_code')==0 and 'failure'not in s and s.get('source_SHA256')==pins and s.get('operator_preapproval')==approval and s.get('outputs')==outputs and s.get('result')==result,'Exact successful operator stage/result/source')
    resources=dict(CPU=17,wall_s=1800,RSS_bytes=4*1024**3)
    need(s.get('resources')==resources and a.get('stage_caps',{}).get(name)==[17,1800,4*1024**3],'Frozen finite owned resource cap')
    for key,upper in(('wall_s',1800),('peak_RSS_bytes',4*1024**3)):
        x=s.get(key);need(type(x)in(int,float)and np.isfinite(x)and 0<=x<=upper,'Actual finite resource witness')
    cmd=s.get('command',[]);need(len(cmd)==10 and Path(cmd[3]).resolve()==Path(sys.executable).resolve()and cmd[:3]==['taskset','-c','17']and cmd[4:]==['-u','-m',run.__name__,phase,'--parent-sha',parent_sha],'Exact finite writer command')
    identity=s['child_identity'];pid=identity['pid'];owned=s['observed_owned']
    need(type(pid)is int and pid>1 and identity['PGID']==pid and type(identity.get('starttime'))is int and len(owned)==1 and owned[0]['pid']==pid and owned[0]['PGID']==pid and owned[0]['starttime']==identity['starttime']and v.get('owned_pids')==[pid],'One exact owned writer/session; no unexpected descendants')
    c.base.absent(pid)
    for entry in outputs:bound(entry)
    return dict(record=rec,stage=stage,preapproval=approval,pid=pid)

def metadata_barrier(parent_sha):
    """All causal memberships, closures and archive hashes; NO deserialization."""
    release=run.read(LOCAL/'release.json')
    need(release.get('approved')is True and release.get('parent_protocol_SHA256')==parent_sha and release.get('source_SHA256')==run.sources()and release.get('resources')==run.LIMITS,'External finite release exact source/parent/resources')
    p=c.validate(parent_sha);seals=c.all_seals_gate(parent_sha);inputs=c.inputs_gate(parent_sha)
    need(len(seals)==32 and[v['index']for v in seals]==list(range(32)),'All32 rootclosed continuous choices')
    choice_rec=run.record(LOCAL/'finite-choice-seals.json');evaluation_rec=run.record(LOCAL/'evaluation-results-LOCAL.json')
    choose_closed=closure('choose',parent_sha,choice_rec);evaluate_closed=closure('evaluate',parent_sha,evaluation_rec)
    need(release.get('root_owned_operator_preapproval')==choose_closed['preapproval']==evaluate_closed['preapproval'],'Same prepinned operator release/closures')
    seal=run.read(bound(choice_rec));value=run.read(bound(evaluation_rec))
    need(seal.get('completed')is True and 'failure'not in seal and seal.get('pid')==choose_closed['pid']and seal.get('source_SHA256')==run.sources()and seal.get('parent_protocol')==run.record(c.LOCAL/'protocol.json')and seal.get('parent_release')==run.record(c.LOCAL/'release.json')and seal.get('root_release')==run.record(LOCAL/'release.json')and seal.get('continuous32_choice_seals')==seals and seal.get('continuous_choice_execution')==run.record(c.LOCAL/'execution-choices.json')and seal.get('continuous_choice_root_closure')==run.record(c.LOCAL/'root-closure-choices.json')and seal.get('inputs')==run.record(c.LOCAL/'inputs.json')and seal.get('no_trial_identity_or_mapped_outcome_used')is True,'Complete finite64 causal seal')
    need(seal.get('candidate_labels')==list(run.CANDIDATE_LABELS)and seal.get('fixed_action')==5 and seal.get('fixed_action_label')==run.CANDIDATE_LABELS[5],'Exact fixed5/candidate IDs')
    memberships=[(name,i)for name in run.MODELS for i in range(32)]
    need([(v['model'],v['index'])for v in seal['choices']]==memberships,'Every64 finitechoice membership')
    for row in seal['choices']:
        index=row['index'];run.choice_check(row['choice'])
        need(row['prefix_input']==inputs['prefixes'][index]['record']and row['observation_fingerprint']==inputs['prefixes'][index]['observation_fingerprint']and row['fixed_action']==5,'Shared original prefix observation/fixedaction')
    loads=[v['id']for v in c.design.physics.load_specs()];captures=[dict(state=i,label=label,capture=c.capture(p,i,label))for i in range(8)for label in loads]
    need(len(captures)==88 and seal['source_captures']==captures and seal['catalog']==run.record(LOCAL/'catalogs-LOCAL.npz'),'Exact physical88/catalog commitments')
    need(value.get('completed')is True and 'failure'not in value and value.get('source_SHA256')==run.sources()and value.get('parent_protocol')==run.record(c.LOCAL/'protocol.json')and value.get('finite_choices')==choice_rec,'Complete sourcebound finiteevaluation')
    need([(v['model'],v['index'])for v in value['rows']]==memberships,'All64 evaluator rows; no filtering')
    for key in('wall_s','combined_owned_wall_s'):
        x=value.get(key);need(type(x)in(int,float)and np.isfinite(x)and 0<=x<=1800,'Finite evaluation activewall cap')
    need(np.isfinite(seal.get('wall_s',np.nan))and 0<=seal['wall_s']<=1800 and abs(value['combined_owned_wall_s']-value['wall_s']-seal['wall_s'])<.1,'Combined choose/evaluate owned1800s')
    for row in value['rows']:
        name,index=row['model'],row['index'];expected=run.record(LOCAL/'posteriors-LOCAL'/f'{name}-{index:02d}.npz')
        need(row.get('posterior_archive')==expected and row.get('choice_seal')==choice_rec,'Canonical everyposterior/seal commitment');bound(expected)
    # Require producer's independent continuous terminal audit as the operator does.
    audit_rec=run.record(c.LOCAL/'independent-terminal-review.json');audit=run.read(bound(audit_rec))
    need(audit.get('audit_passed')is True and audit.get('protocol')==run.record(c.LOCAL/'protocol.json')and audit.get('source_SHA256')=={k:run.sha(ROOT/k)for k in(c.AUDIT_SOURCE,c.AUDIT_TEST)},'Independently audited continuous parent source/protocol')
    c.verify(audit);c.verify(captures);bound(seal['catalog']);c.verify(inputs)
    return p,inputs,seal,value,dict(choose=choose_closed,evaluate=evaluate_closed,continuous_audit=audit_rec)

def reconstruct_catalog(p,check=lambda:None):
    """Independent projection of all88 physical fields, no solver/model helper."""
    points=c.design.physics.measurement_points();specs=c.design.physics.measurement_specs();loads=[v['id']for v in c.design.physics.load_specs()];arrays={k:np.empty((8,60))for k in run.MODELS}
    need(points.shape==(16,3)and len(specs)==60 and len(loads)==11 and p['objects']==c.design.objects(),'Same eight public hypotheses/query16/60labels')
    for state in range(8):
        queries={name:{}for name in run.MODELS}
        for label in loads:
            check()
            with np.load(bound(c.capture(p,state,label)),allow_pickle=False)as z:
                need(np.array_equal(z['query_points_m'],points),'Same public physical query16')
                queries['NH'][label]=z['query_displacement_m'].copy();queries['linear'][label]=c.physics_stage.geometry.field_queries(12,z['linear_displacement_m'],points)
                need(all(q[label].shape==(16,3)and np.isfinite(q[label]).all()for q in queries.values()),'Finite16x3 NH/linear public fields')
        for k,spec in enumerate(specs):
            matches=[j for j,point in enumerate(points)if np.array_equal(point,spec['point_m'])];need(len(matches)==1,'Unique authored query coordinate')
            for name in run.MODELS:arrays[name][state,k]=queries[name][spec['load']][matches[0]]@np.asarray(spec['projection'])
    return arrays,[s['id']for s in specs]

def direct_posterior(catalog,observed,action=None,outcome=None):
    a=np.asarray(catalog,float);obs=np.asarray(observed,float);sigma=c.design.physics.SIGMA_M
    need(a.shape==(8,60)and obs.shape==(4,)and np.isfinite(a).all()and np.isfinite(obs).all(),'Uniform8 complete finite hypothesis schema')
    logs=-.5*np.sum(((a[:,:4]-obs)/sigma)**2,axis=1)
    if action is not None:
        need(type(action)is int and 0<=action<8 and np.isfinite(outcome),'Exactly ONE reset scalar action');logs-=.5*((a[:,4+action]-outcome)/sigma)**2
    weights=np.exp(logs-logsumexp(logs));mean=weights@a[:,12:];variance=weights@((a[:,12:]-mean)**2)
    return dict(weights=weights,mean=mean,variance=variance)

def moment_metrics(moment,truth):
    truth=np.asarray(truth,float);need(truth.shape==(48,)and np.isfinite(truth).all(),'Complete clean48 heldtargets')
    mean=moment['mean'];variance=moment['variance'];sd=np.sqrt(variance)
    return dict(MSE_m2=float(np.mean((mean-truth)**2)),posterior_variance_m2=float(np.mean(variance)),moment_band_2SD_mean_width_m=float(np.mean(4*sd)),moment_band_2SD_inclusion_fraction=float(np.mean(np.abs(truth-mean)<=2*sd)))

def check_metrics(saved,computed):
    for key in('MSE_m2','posterior_variance_m2'):close_numbers(saved[key],computed[key])

def independent_summary(rows):
    need(len(rows)==64 and{(v['model'],v['index'])for v in rows}=={(m,i)for m in run.MODELS for i in range(32)},'Fixed64 comparison denominator')
    qualified=sum(v['qualified']for v in rows);out=dict(retained_trials_per_model=32,retained_counterfactuals_per_model=256,qualified_choices=qualified,all64_retained=True,comparison_blocked=qualified!=64,fixed_action=5,fixed_action_label=run.CANDIDATE_LABELS[5],finite_prior_differs_continuous_prior=True,population_or_calibration_claim=False)
    if qualified!=64:return out,{}
    policies=('before','adaptive','fixed','uniform_random');aggregate={};diag={}
    for model in run.MODELS:
        objects=[]
        for obj in range(8):
            group=[v for v in rows if v['model']==model and v['object_index']==obj]
            need(len(group)==4 and{v['replicate']for v in group}==set(range(4)),'Equal fourrepeats/eightobjects')
            objects.append({key:float(np.mean([v['policies'][key]['MSE_m2']for v in group]))for key in policies})
        aggregate[model]={key:float(np.mean([o[key]for o in objects]))for key in policies}
        group=[v for v in rows if v['model']==model]
        diag[model]={key:{**{metric:float(np.mean([v['policies'][key][metric]for v in group]))for metric in('posterior_variance_m2','moment_band_2SD_mean_width_m','moment_band_2SD_inclusion_fraction')},'variance_down_error_up_trials':sum(v['policies'][key]['posterior_variance_m2']<v['policies']['before']['posterior_variance_m2']and v['policies'][key]['MSE_m2']>v['policies']['before']['MSE_m2']for v in group)}for key in policies}
    out.update(equal_object_mean_MSE_m2=aggregate,NH_minus_linear_MSE_m2={key:aggregate['NH'][key]-aggregate['linear'][key]for key in policies})
    return out,diag

def audit(parent_sha):
    start=time.monotonic();guard(start);target=LOCAL/'independent-terminal-review.json';need(not target.exists(),'No audit overwrite')
    p,inputs,seals,result,closures=metadata_barrier(parent_sha)
    # Every raw input/physical88/posterior64 byte commitment and closed choice
    # memberships are now verified. No numerical data were opened above.
    arrays,labels=reconstruct_catalog(p,lambda:guard(start))
    with np.load(bound(seals['catalog']),allow_pickle=False)as z:
        need(set(z.files)=={'NH','linear','labels'}and tuple(str(v)for v in z['labels'])==tuple(labels),'Exact finitecatalog schema/labels')
        for name in run.MODELS:need(np.array_equal(z[name],arrays[name]),'Independent reconstructed NH/linear catalog equality')
    with np.load(bound(inputs['poke_noise']),allow_pickle=False)as z:
        need(set(z.files)=={'poke_noise'},'Exact single shared poke-noise archive');noise=z['poke_noise'].copy()
    need(noise.shape==(4,8,8)and np.isfinite(noise).all(),'All256 bound noisy scalar outcomes; no redraw')
    rows=[];worst=dict(weight_absolute=0.,mean_absolute_m=0.)
    for entry in result['rows']:
        guard(start);name,index=entry['model'],entry['index'];obj,rep=index%8,index//8;choice=seals['choices'][run.MODELS.index(name)*32+index];trial=p['trials'][index]
        need(trial==dict(index=index,object_index=obj,replicate=rep)and(entry['object_index'],entry['replicate'])==(obj,rep),'Same frozen32 identities afterall64seals')
        with np.load(bound(inputs['prefixes'][index]['record']),allow_pickle=False)as z:
            need(set(z.files)=={'observed_prefix','prefix_noise'},'Exact acquired prefixschema');observed=z['observed_prefix'].copy();prefix_noise=z['prefix_noise'].copy()
        need(observed.shape==prefix_noise.shape==(4,)and np.isfinite(observed).all()and np.isfinite(prefix_noise).all()and np.array_equal(observed,arrays['NH'][obj,:4]+prefix_noise)and c.design.fingerprint(observed)==choice['observation_fingerprint'],'Shared NH truth/prefixnoise/fingerprint binding')
        base=direct_posterior(arrays[name],observed);truth=arrays['NH'][obj,12:];before=moment_metrics(base,truth);counter=[];moments=[]
        for action in range(8):
            moment=direct_posterior(arrays[name],observed,action,float(arrays['NH'][obj,4+action]+noise[rep,obj,action]));moments.append(moment);counter.append(moment_metrics(moment,truth))
        with np.load(bound(entry['posterior_archive']),allow_pickle=False)as z:
            need(set(z.files)=={'mean_m','weights','before_mean_m'},'Exact lean512updates archive')
            expected_mean=np.array([v['mean']for v in moments]);expected_weight=np.array([v['weights']for v in moments])
            close_numbers(z['mean_m'],expected_mean,atol=2e-12,rtol=0);close_numbers(z['weights'],expected_weight,atol=2e-12,rtol=0);close_numbers(z['before_mean_m'],base['mean'],atol=2e-12,rtol=0)
            worst['weight_absolute']=max(worst['weight_absolute'],float(np.max(abs(z['weights']-expected_weight))));worst['mean_absolute_m']=max(worst['mean_absolute_m'],float(np.max(abs(z['mean_m']-expected_mean))))
        qualified=run.choice_check(choice['choice']);selected=choice['choice']['selected_action']
        need(entry['choice_qualified']is qualified and entry['selected_action']==selected and entry['fixed_action']==5,'Independent sealedchoice status')
        check_metrics(entry['before'],before);need(len(entry['counterfactuals'])==8,'All8 resetposterior records')
        for action,(saved,computed)in enumerate(zip(entry['counterfactuals'],counter)):
            need(saved['action']==action and saved.get('fresh_prefix_plus_ONE_scalar')is True,'All8 freshprefix singlepoke flags');check_metrics(saved,computed)
        policies=dict(before=before,fixed=counter[5],uniform_random={k:float(np.mean([v[k]for v in counter]))for k in before})
        if qualified:policies['adaptive']=counter[selected]
        need(set(entry['policies'])==set(policies),'No adaptive comparison from unresolvedchoice')
        for key,metrics in policies.items():check_metrics(entry['policies'][key],metrics)
        rows.append(dict(model=name,index=index,object_index=obj,replicate=rep,qualified=qualified,policies=policies))
    summary,diagnostics=independent_summary(rows);saved=result['summary'];need(set(saved)==set(summary)|({'aggregation','scope'}if not summary['comparison_blocked']else set()),'Exact complete/blocked summary schema')
    for key,value in summary.items():
        if key in('equal_object_mean_MSE_m2','NH_minus_linear_MSE_m2'):
            def compare(a,b):
                if isinstance(b,dict):need(set(a)==set(b),'Exact metricmemberships');[compare(a[k],b[k])for k in b]
                else:close_numbers(a,b)
            compare(saved[key],value)
        else:need(saved[key]==value,'Exact summarydenominator/classification')
    need(run.read(LOCAL/'compact-summary.json')==saved,'Compactsummary exact producer result')
    out=dict(audit_passed=True,source_SHA256=sources(),source_numerical_SHA256=run.sources(),parent_protocol=run.record(c.LOCAL/'protocol.json'),finite_release=run.record(LOCAL/'release.json'),finite_choices=run.record(LOCAL/'finite-choice-seals.json'),source_evaluation=run.record(LOCAL/'evaluation-results-LOCAL.json'),catalog=seals['catalog'],source_captures=seals['source_captures'],all64_posterior_archives=[v['posterior_archive']for v in result['rows']],closed_stages=closures,all32continuous_and64finite_choice_barriers_verified=True,retained_objects=8,retained_noise_repeats_per_object=4,retained_trials_per_model=32,retained_reset_updates_per_model=256,independently_reconstructed_catalogs=2,independently_recomputed_posterior_updates=512,maximum_reproduction_errors=worst,summary=summary,descriptive_momentband_diagnostics=diagnostics,quadrature_choice_risks_are_producer_witnesses_scalar_flags_recomputed=True,momentband_scope='Mean±2SD descriptive finite-prior moment bands; no nominal coverage/calibration claim. Variance-down/error-up is not alone miscalibration proof.',scope='Same uniform8 knownK prior and identical NH outcomes isolate matchedfinite responsephysics. Continuous comparison also changes prior/knownK and is not isolated physics. Exact uniform8counterfactual losses; mean4withinobject then equal8object. Raw arrays LOCAL.',tolerances=TOLERANCES,audit_resources=LIMITS,audit_wall_s=time.monotonic()-start,audit_peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
    guard(start);run.write(target,out);return out

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--parent-sha',required=True);args=ap.parse_args();print(__import__('json').dumps(audit(args.parent_sha)))
