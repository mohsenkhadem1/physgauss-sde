"""Versioned finite verifier recovery; immutable v01 choices, no choose/RNG/PDE.

Only prior-witness arithmetic consistency changes: finite nonnegative FP64
values within eight ULPs. Risk/refinement/tie gates use original top prior.
New evaluation/archive lineage explicitly differs from the original chooser.
"""
import argparse,os,resource,shutil,time
from pathlib import Path
import numpy as np
from experiments import run_matched_finite_poke_control_v01 as old
from experiments import audit_matched_finite_poke_control_v01 as old_audit
from experiments import run_noisy_poke_continuation_v02 as continued
from experiments import audit_noisy_poke_continuation_v02 as continued_audit
ROOT=old.ROOT;LOCAL=ROOT/'runs/operator-dev/matched-finite-poke-control-v02'
SOURCE='experiments/run_matched_finite_poke_control_v02.py';TEST='tests/test_matched_finite_poke_control_v02.py'
AUDIT_SOURCE='experiments/audit_matched_finite_poke_control_v02.py';AUDIT_TEST='tests/test_audit_matched_finite_poke_control_v02.py'
OPERATOR=ROOT/'runs/operator-dev/noisy-poke-recovery-operator-v02'
MODELS=old.MODELS;FIXED_ACTION=old.FIXED_ACTION;CANDIDATE_LABELS=old.CANDIDATE_LABELS
LIMITS=dict(old.LIMITS)
PRIOR_RULE=dict(schema='finite-prior-witness-FP64-ULP-v02',maximum_ULP_distance=8,physical_absolute_tolerance=0.,zero_signs_equivalent=True,risk_refinement_tie_gates_unchanged=True)
continuous=old.continuous
sha,record,read,write=old.sha,old.record,old.read,old.write
model,mse_metrics,policy_metrics,summarize=old.model,old.mse_metrics,old.policy_metrics,old.summarize

def sources():
    names=set(old.sources())|set(old_audit.sources())|set(continued.sources())|{SOURCE,TEST,AUDIT_SOURCE,AUDIT_TEST}
    return {n:sha(ROOT/n)for n in sorted(names)}
def guard(start):
    if set(os.sched_getaffinity(0))!={17}or os.environ.get('CUDA_VISIBLE_DEVICES')!=''or any(os.environ.get(k)!='1'for k in('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS')):raise ValueError('Finite recovery CPU17/noGPU/one-thread')
    if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024>=LIMITS['RSS_bytes']:raise MemoryError('Finite recovery4GiB')
    if time.monotonic()-start>=LIMITS['wall_s']:raise TimeoutError('Finite recovery combined1800s')
    if shutil.disk_usage(ROOT).free<LIMITS['disk_reserve_bytes']+LIMITS['maximum_writes_bytes']:raise OSError('30GiBfree plus writes')
    if LOCAL.exists()and sum(p.stat().st_size for p in LOCAL.rglob('*')if p.is_file())>LIMITS['maximum_writes_bytes']:raise OSError('Finite recovery128MiBwrites')

def prior_witness_ULPs(a,b):
    if any(isinstance(v,(bool,np.bool_))or not isinstance(v,(int,float,np.integer,np.floating))for v in(a,b)):return None
    a,b=float(a),float(b)
    if not np.isfinite([a,b]).all()or a<0 or b<0:return None
    if a==b:return 0 # Includes equivalent +0/-0; no negative variance admitted.
    return abs(int(np.float64(a).view(np.uint64))-int(np.float64(b).view(np.uint64)))
def prior_witness_close(a,b):
    distance=prior_witness_ULPs(a,b);return distance is not None and distance<=PRIOR_RULE['maximum_ULP_distance']

def choice_check(choice):
    """Recompute frozen helper's scalar refinement/tie predicates, no replay."""
    prior=choice['prior_variance_m2'];rows=choice['actions'];tiny=np.finfo(float).tiny
    if not np.isfinite(prior)or prior<0 or len(rows)!=8 or[v['action']for v in rows]!=list(range(8)):raise ValueError('All8finiteaction/prior witnesses')
    for row in rows:
        levels=row['levels']
        if len(levels)!=5 or[v['order']for v in levels]!=[8,16,32,64,128]or any(not np.isfinite(v['risk_m2'])or v['risk_m2']<0 or not prior_witness_close(v['prior_variance_m2'],prior) or v.get('all_hypotheses_retained')is not True for v in levels):raise ValueError('All originalfivepositive quadraturelevels/hypotheses')
        error=abs(levels[-1]['risk_m2']-levels[-2]['risk_m2']);risk=levels[-1]['risk_m2'];rounding=128*np.finfo(float).eps*max(prior,tiny)
        qualified=bool(np.isfinite(risk)and risk>=0 and risk<=prior+error+rounding and error<=1e-4*max(prior,tiny))
        if row['risk_m2']!=risk or row['refinement_indicator_m2']!=error or row['qualified']is not qualified:raise ValueError('Original finitequadrature flags')
    qualified=all(v['qualified']for v in rows);best=min(rows,key=lambda v:(v['risk_m2'],v['action']));ties=[v['action']for v in rows if v['risk_m2']-best['risk_m2']<=v['refinement_indicator_m2']+best['refinement_indicator_m2']]
    if choice['qualified']is not qualified or choice['numerical_ties']!=ties or choice['selected_action']!=(min(ties)if qualified else None):raise ValueError('Exact finitechoice qualification/tie/selection')
    return qualified


def original_choices_gate(parent_sha):
    """Original metadata/closure/64choice checks only; never deserialize NPZ."""
    release=read(old.LOCAL/'release.json')
    if not(release.get('approved')is True and release.get('parent_protocol_SHA256')==parent_sha and release.get('source_SHA256')==old.sources()and release.get('resources')==old.LIMITS):raise ValueError('Exact original finite release before metadata reuse')
    p=continuous.validate(parent_sha);seals=continuous.all_seals_gate(parent_sha);inputs=continuous.inputs_gate(parent_sha)
    if len(seals)!=32 or[v['index']for v in seals]!=list(range(32)):raise ValueError('Original32 closed continuous choices')
    choice_record=record(old.LOCAL/'finite-choice-seals.json');closed=old_audit.closure('choose',parent_sha,choice_record);value=read(old.LOCAL/'finite-choice-seals.json')
    if not(value.get('completed')is True and 'failure'not in value and value['pid']==closed['pid']and value['source_SHA256']==old.sources()and value['parent_protocol']==record(continuous.LOCAL/'protocol.json')and value['parent_release']==record(continuous.LOCAL/'release.json')and value['root_release']==record(old.LOCAL/'release.json')and value['continuous32_choice_seals']==seals and value['continuous_choice_execution']==record(continuous.LOCAL/'execution-choices.json')and value['continuous_choice_root_closure']==record(continuous.LOCAL/'root-closure-choices.json')and value['inputs']==record(continuous.LOCAL/'inputs.json')and value.get('no_trial_identity_or_mapped_outcome_used')is True):raise ValueError('Exact immutable original64 causal/source seal')
    if[(v['model'],v['index'])for v in value['choices']]!=[(name,i)for name in MODELS for i in range(32)]:raise ValueError('Every original64choice retained')
    if value.get('candidate_labels')!=list(CANDIDATE_LABELS)or value.get('fixed_action')!=5 or value.get('fixed_action_label')!=CANDIDATE_LABELS[5]or not np.isfinite(value.get('wall_s',np.nan))or not 0<=value['wall_s']<=LIMITS['wall_s']:raise ValueError('Original action/activecost metadata')
    maximum=exact_mismatches=0
    for row in value['choices']:
        choice_check(row['choice'])
        if row['prefix_input']!=inputs['prefixes'][row['index']]['record']or row['observation_fingerprint']!=inputs['prefixes'][row['index']]['observation_fingerprint']or row['fixed_action']!=5:raise ValueError('Original sharedprefix/fixedaction binding')
        for action in row['choice']['actions']:
            for level in action['levels']:
                distance=prior_witness_ULPs(level['prior_variance_m2'],row['choice']['prior_variance_m2']);maximum=max(maximum,distance);exact_mismatches+=distance>0
    loads=[v['id']for v in continuous.design.physics.load_specs()];captures=[dict(state=i,label=label,capture=continuous.capture(p,i,label))for i in range(8)for label in loads]
    if value['source_captures']!=captures or value['catalog']!=record(old.LOCAL/'catalogs-LOCAL.npz'):raise ValueError('Exact original88 physical/catalog commitments')
    continuous.verify(captures);continuous.verify(value['catalog']);continuous.verify(inputs)
    origin=dict(original_choices=choice_record,original_choose_closure=record(old.LOCAL/'root-closure-choose.json'),original_choose_stage=closed['stage'],original_operator_preapproval=closed['preapproval'],original_finite_release=record(old.LOCAL/'release.json'),original_inputs=record(continuous.LOCAL/'inputs.json'),original_catalog=value['catalog'],parent_protocol=record(continuous.LOCAL/'protocol.json'),verifier_rule=PRIOR_RULE,maximum_prior_ULP_distance=maximum,nonidentical_prior_witnesses=exact_mismatches)
    return p,inputs,value,origin

def continuous_admission(parent_sha):
    p,phases,seals,inputs,entries=continued_audit.metadata_barrier(parent_sha)
    path=continued.LOCAL/'independent-terminal-review.json';a=read(path);ctx=continued.execution_context();summary=read(continued.LOCAL/'summary-LOCAL.json')
    if not(a.get('audit_passed')is True and a.get('parent_protocol')==record(continuous.LOCAL/'protocol.json')and a.get('parent_source_SHA256')==p['source_SHA256']==continuous.sources()and a.get('protocol')==ctx['execution_protocol']and a.get('release')==ctx['execution_release']and a.get('execution_source_SHA256')==ctx['execution_source_SHA256']and a.get('source_SHA256')==continued_audit.sources()and a.get('source_summary')==record(continued.LOCAL/'summary-LOCAL.json')and a.get('closed_phases')==dict(zip(('choices','checkpoint','remaining'),continued_audit.phase_records()))and a.get('all32_trial_records')==[v['record']for v in entries]and all(summary.get(k)==v for k,v in ctx.items())):raise ValueError('Exact independent NEW continuous audit/source/mixed-lineage prerequisite')
    continuous.verify(a);return record(path)

def release_gate(parent_sha):
    r=read(LOCAL/'release.json');approval=record(OPERATOR/'root-preapproval.json');a=read(OPERATOR/'root-preapproval.json')
    if not(r.get('schema')=='matched-finite-poke-recovery-release-v02'and r.get('approved')is True and r.get('parent_protocol_SHA256')==parent_sha and r.get('source_SHA256')==sources()and r.get('resources')==LIMITS and r.get('verifier_rule')==PRIOR_RULE and r.get('root_owned_operator_preapproval')==approval):raise ValueError('Exact root finite recovery release before any outcome data')
    if a.get('approved')is not True or a.get('schema')!='noisy-poke-recovery-operator-preapproval-v02'or any(a.get('source_SHA256',{}).get(n)!=h for n,h in sources().items()):raise ValueError('ROOT preapproved complete versioned finite source/audit graph')
    for name,h in a['source_SHA256'].items():
        if sha(ROOT/name)!=h:raise ValueError('Changed root-preapproved source')
    return r

def admission(parent_sha,start):
    guard(start);r=release_gate(parent_sha);p,inputs,value,origin=original_choices_gate(parent_sha)
    if r.get('origin')!=origin or r.get('continuous_audit')!=continuous_admission(parent_sha):raise ValueError('Root release binds original choices and NEW continuous audit')
    return p,inputs,value,origin

def evaluate(parent_sha):
    start=time.monotonic();p,inputs,sealed,origin=admission(parent_sha,start)
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
                rows.append(dict(model=name,index=index,object_index=i,replicate=r,choice_qualified=qualified,choice_seal=record(old.LOCAL/'finite-choice-seals.json'),selected_action=choice['choice']['selected_action'],fixed_action=5,before=before,counterfactuals=counter,policies=policies,posterior_archive=record(raw)))
                write(LOCAL/'evaluation-partial-LOCAL.json',dict(completed=False,rows=rows))
        summary=summarize(rows);guard(budget_start);write(target,dict(completed=True,parent_protocol=record(continuous.LOCAL/'protocol.json'),finite_choices=record(old.LOCAL/'finite-choice-seals.json'),source_SHA256=sources(),original_chooser_source_SHA256=sealed['source_SHA256'],recovery_origin=origin,verifier_rule=PRIOR_RULE,rows=rows,summary=summary,wall_s=time.monotonic()-start,combined_owned_wall_s=time.monotonic()-budget_start,peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024));write(LOCAL/'compact-summary.json',summary)
    except BaseException as error:write(LOCAL/'evaluation-partial-LOCAL.json',dict(completed=False,rows=rows,failure=dict(type=type(error).__name__,message=str(error))));raise
    return record(target)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['evaluate']);ap.add_argument('--parent-sha',required=True);a=ap.parse_args();print(__import__('json').dumps(evaluate(a.parent_sha)))
