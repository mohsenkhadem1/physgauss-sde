"""Independent v02 finite recovery audit; original chooser/new evaluator lineage.

No rechoice, posterior-helper replay, RNG or PDE. Same original finite prior,
noise and512 resetupdates. Only8ULP prior-witness consistency is versioned.
"""
import argparse,resource,sys,time
from pathlib import Path
import numpy as np
from experiments import run_matched_finite_poke_control_v02 as run
from experiments import audit_matched_finite_poke_control_v01 as legacy
ROOT=run.ROOT;LOCAL=run.LOCAL;c=run.continuous;OPERATOR=run.OPERATOR
SOURCE='experiments/audit_matched_finite_poke_control_v02.py';TEST='tests/test_audit_matched_finite_poke_control_v02.py'
LIMITS=dict(legacy.LIMITS);TOLERANCES=dict(legacy.TOLERANCES)
need,guard,close_numbers=legacy.need,legacy.guard,legacy.close_numbers
reconstruct_catalog,direct_posterior,moment_metrics=legacy.reconstruct_catalog,legacy.direct_posterior,legacy.moment_metrics
check_metrics,independent_summary=legacy.check_metrics,legacy.independent_summary

def sources():return {name:run.sha(ROOT/name)for name in(SOURCE,TEST)}
def bound(rec):
    path=ROOT/rec['path'];need(run.record(path)==rec,'Exact v02 bound original/new bytes');return path

def closure(parent_sha,result):
    phase='evaluate'
    """Verify external owned wait/reap, not a claim about unified-exec."""
    rec=run.record(LOCAL/f'root-closure-{phase}.json');v=run.read(bound(rec));name='finite_'+phase
    need(all(v.get(k)is True for k in('root_reaped','root_owned_supervisor','external_subprocess_wait','all_owned_PID_and_groups_absent'))and v.get('root_exit_code')==0 and v.get('parent_protocol_SHA256')==parent_sha and v.get('source_SHA256')==run.sources()and v.get('result')==result,'Exact finite external root closure')
    approval=run.record(OPERATOR/'root-preapproval.json');need(v.get('operator_preapproval')==approval,'Exact operator preapproval')
    a=run.read(bound(approval));need(a.get('approved')is True and a.get('schema')=='noisy-poke-recovery-operator-preapproval-v02'and a.get('stage_caps',{}).get('finite_audit')==[16,1800,4*1024**3],'Approved operator procedure and external audit watchdog')
    pins=a['source_SHA256'];need(all(pins.get(k)==h for k,h in{**run.sources(),**sources()}.items()),'Auditor/test and numerical procedure prepinned before execution')
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
    release=run.release_gate(parent_sha);p,inputs,seal,origin=run.original_choices_gate(parent_sha)
    need(release.get('origin')==origin and release.get('continuous_audit')==run.continuous_admission(parent_sha),'Release binds original choices and independent NEW continuous audit')
    choice_rec=run.record(run.old.LOCAL/'finite-choice-seals.json');evaluation_rec=run.record(LOCAL/'evaluation-results-LOCAL.json');closed=closure(parent_sha,evaluation_rec)
    need(release.get('root_owned_operator_preapproval')==closed['preapproval'],'Same NEW operator authority for release/evaluator')
    value=run.read(bound(evaluation_rec))
    need(value.get('completed')is True and 'failure'not in value and value.get('source_SHA256')==run.sources()and value.get('original_chooser_source_SHA256')==run.old.sources()and value.get('recovery_origin')==origin and value.get('verifier_rule')==run.PRIOR_RULE and value.get('parent_protocol')==run.record(c.LOCAL/'protocol.json')and value.get('finite_choices')==choice_rec,'Honest originalchooser/newproducer context')
    memberships=[(name,i)for name in run.MODELS for i in range(32)]
    need([(v['model'],v['index'])for v in value['rows']]==memberships,'All64 evaluator rows; no filtering')
    for key in('wall_s','combined_owned_wall_s'):
        x=value.get(key);need(type(x)in(int,float)and np.isfinite(x)and 0<=x<=1800,'Unchanged finite activewall bound')
    need(abs(value['combined_owned_wall_s']-value['wall_s']-seal['wall_s'])<.1,'Originalchoose plusNEWevaluate combined1800s')
    for row in value['rows']:
        name,index=row['model'],row['index'];expected=run.record(LOCAL/'posteriors-LOCAL'/f'{name}-{index:02d}.npz')
        need(row.get('posterior_archive')==expected and row.get('choice_seal')==choice_rec,'Every canonical NEW posterior and ORIGINALchoice commitment');bound(expected)
    c.verify(seal['source_captures']);bound(seal['catalog']);c.verify(inputs)
    return p,inputs,seal,value,dict(original_choose=dict(record=origin['original_choose_closure'],stage=origin['original_choose_stage'],preapproval=origin['original_operator_preapproval']),new_evaluate=closed,continuous_audit=release['continuous_audit'])

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
    out=dict(audit_passed=True,source_SHA256=sources(),source_numerical_SHA256=run.sources(),original_chooser_source_SHA256=run.old.sources(),verifier_rule=run.PRIOR_RULE,sealed_choices_reused_without_modification=True,parent_protocol=run.record(c.LOCAL/'protocol.json'),finite_release=run.record(LOCAL/'release.json'),finite_choices=run.record(run.old.LOCAL/'finite-choice-seals.json'),source_evaluation=run.record(LOCAL/'evaluation-results-LOCAL.json'),catalog=seals['catalog'],source_captures=seals['source_captures'],all64_posterior_archives=[v['posterior_archive']for v in result['rows']],closed_stages=closures,all32continuous_and64finite_choice_barriers_verified=True,retained_objects=8,retained_noise_repeats_per_object=4,retained_trials_per_model=32,retained_reset_updates_per_model=256,independently_reconstructed_catalogs=2,independently_recomputed_posterior_updates=512,maximum_reproduction_errors=worst,summary=summary,descriptive_momentband_diagnostics=diagnostics,quadrature_choice_risks_are_producer_witnesses_scalar_flags_recomputed=True,momentband_scope='Mean±2SD descriptive finite-prior moment bands; no nominal coverage/calibration claim. Variance-down/error-up is not alone miscalibration proof.',scope='Same uniform8 knownK prior and identical NH outcomes isolate matchedfinite responsephysics. Continuous comparison also changes prior/knownK and is not isolated physics. Exact uniform8counterfactual losses; mean4withinobject then equal8object. Raw arrays LOCAL.',tolerances=TOLERANCES,audit_resources=LIMITS,audit_wall_s=time.monotonic()-start,audit_peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
    guard(start);run.write(target,out);return out

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--parent-sha',required=True);args=ap.parse_args();print(__import__('json').dumps(audit(args.parent_sha)))
