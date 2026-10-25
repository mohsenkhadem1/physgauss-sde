"""Independent terminal scalar/provenance audit of fixed36 DEVELOPMENT.

No model, PDE, GPU, NPZ or posterior replay. Risk/mass/refinement/tie/adequacy
arithmetic is recomputed from saved scalars. Category target-moment agreement,
within/between variance and exact coefficient/PDE parity are bound producer
witnesses: full reference moment arrays are deliberately not archived here.
"""
import argparse,copy
import numpy as np
from experiments import run_fixture_balanced_decision_screen_v01 as run
from experiments import audit_layout_grouped_reference_cost_v01 as refs_audit
from experiments import audit_layout_grouped_allactions_v01 as action_audit
ROOT=run.ROOT;LOCAL=run.LOCAL
SOURCE='experiments/audit_fixture_balanced_decision_screen_v01.py';TEST='tests/test_audit_fixture_balanced_decision_screen_v01.py'
EXPECTED_PROTOCOL=None # Bound explicitly to prospective protocol before actual audit.
def need(v,message):
    if not v:raise ValueError(message)
def bound(rec):
    p=ROOT/rec['path'];need(run.record(p)==rec,'Exact committed artifact bytes');return p
def finite_nonnegative(x):return isinstance(x,(int,float))and not isinstance(x,bool)and np.isfinite(x)and x>=0

def reference_metadata(value):
    d=value['diagnostics'];need(d.get('independent_adaptive_per_posterior_required')is True,'Mandatory independent adaptive per posterior flag')
    valid=True
    for key in((512,512),(512,256),(256,512)):
        final=[v for v in d.get('levels',[])if(v.get('outer'),v.get('inner'))==key];need(len(final)==1,'One exact required final-pair reference level')
        status=final[0].get('status');need(status in('computed','unresolved'),'Known required final-pair level status')
        if status=='unresolved':need(isinstance(final[0].get('reason'),str)and bool(final[0]['reason'].strip()),'Preserved unresolved numerical reason')
        valid=valid and status=='computed'and isinstance(final[0].get('log_evidence'),(int,float))and np.isfinite(final[0]['log_evidence'])
    if value.get('qualified')is True or value.get('strict',{}).get('passed')is True:need(valid,'Qualified reference requires computed finite final-pair logZ')
    return bool(valid)

def independent_adequacy(rows):
    # Deliberately independent arithmetic, with all missing cases in36denominator.
    need(len(rows)==36 and all(type(r.get('index'))is int for r in rows)and{r['index']for r in rows}==set(range(36)),'All36 unique memberships')
    counts=[0]*8;ties_count=[0]*8;qualified=[];tied=0
    for spec in run.design.worklist():
        row=next(r for r in rows if r['index']==spec['index']);need(row['case']==spec,'Exact immutable case membership');ranking=row.get('ranking')
        if not isinstance(ranking,dict)or ranking.get('qualified')is not True:continue
        need(ranking['all8_retained']is True,'All8 candidates');tie=ranking['numerical_ties'];winner=ranking['unique_best_action'];need(tie and len(set(tie))==len(tie)and all(type(i)is int and 0<=i<8 for i in tie),'Numerical ties in fixed catalog')
        if winner is None:
            need(len(tie)>1,'No arbitrary withholding unique winner');tied+=1
            for i in tie:ties_count[i]+=1
        else:need(tie==[winner],'Unique winner matches numerical tie');counts[winner]+=1
        qualified.append(row)
    out=dict(planned_cases=36,qualified_cases=len(qualified),unresolved_cases=36-len(qualified),unique_ranking_cases=sum(counts),tied_cases=tied,winner_counts=counts,tie_counts=ties_count,ranking_heterogeneity_observed=sum(c>0 for c in counts)>1,universal_unique_ranking=bool(len(qualified)==36 and sum(counts)==36 and sum(c>0 for c in counts)==1),all36_retained=True,no_continuous_NH_or_active_advantage_admission=True,scope=run.design.summarize(rows)['scope'],adequacy_gates=run.design.ADEQUACY,adequacy_qualified=False,comparison_blocked=len(qualified)<36)
    if len(qualified)<36:return out
    ordered=sorted(qualified,key=lambda r:r['index']);risk=np.array([[a['risk']for a in r['ranking']['actions']]for r in ordered]);fixed=int(np.argmin(risk.mean(axis=0)));ambiguity=regret=0;g=run.design.ADEQUACY
    for row in ordered:
        screen=row['posterior_screen'];actions=row['ranking']['actions'];best=min(actions,key=lambda a:(a['risk'],a['action']));a=actions[fixed];prior=screen['prior_variance'];fraction=screen['between_layout_fraction']-screen['between_layout_fraction_indicator']
        ambiguity+=int(screen['maximum_layout_probability']+screen['layout_TV']<=g['maximum_layout_probability']and fraction>=g['minimum_between_layout_fraction']and max(0,fraction)*prior>=g['minimum_between_layout_RMS_SD_m']**2)
        regret+=int(a['risk']-best['risk']-a['uncertainty']-best['uncertainty']>=g['minimum_regret_fraction_of_prior']*prior)
    distinct=sum(c>=g['minimum_winner_fraction']*36 for c in counts);out.update(distinct_winners_with_minimum_fraction=distinct,ambiguity_and_between_layout_cases=ambiguity,fixed_action_mean_risk_selected_from_all36_development=True,strong_fixed_regret_cases=regret,adequacy_qualified=bool(distinct>=g['minimum_distinct_winners']and ambiguity>=g['minimum_ambiguous_fraction']*36 and regret>=g['minimum_regret_case_fraction']*36),indicators_are_not_rigorous_error_bounds=True)
    return out

def unchanged_reference(old,new):
    expected={(512,128),(512,256),(512,512),(128,512),(256,512)}
    a={(v['outer'],v['inner']):v for v in old['levels']};b={(v['outer'],v['inner']):v for v in new['levels']}
    need(len(old['levels'])==len(new['levels'])==5 and set(a)==set(b)==expected,'Exact unchanged-density five level identities')
    errors=[];passed=True
    for key,v in a.items():
        if v['status']=='computed':
            if b[key]['status']!='computed':passed=False;continue
            error=abs(v['log_evidence']-b[key]['log_evidence']);passed=passed and refs_audit.finite_le(error,1e-8);errors.append(error)
    return dict(passed=bool(passed),maximum_logZ_difference=max(errors,default=None),gate=1e-8,only_unchanged_prefix_density=True)

def unchanged_decision(old,new):
    a,b=np.asarray(old['log_evidence']),np.asarray(new['log_evidence']);p,q=np.asarray(old['layout_probability']),np.asarray(new['layout_probability'])
    need(all(v.shape==(4,)and np.isfinite(v).all()for v in(a,b,p,q)),'All4 unchanged density evidences/probabilities')
    logz=float(np.max(abs(a-b)));rho=float(np.max(abs(p-q)))
    return dict(passed=bool(logz<=1e-8 and rho<=1e-10),maximum_logZ_difference=logz,maximum_layout_probability_difference=rho,logZ_gate=1e-8,layout_probability_gate=1e-10,only_unchanged_prefix_density=True)

def check_case(row,spec,metadata,digest,origin):
    index=spec['index'];old_witness=origin['old36_witnesses'][index];need(old_witness['index']==index,'Exact prior case witness');directory=LOCAL/f'case-{index:02d}';need(type(row['index'])is int and row['index']==index and row['case']==spec and row['protocol_SHA256']==digest and row['prefix']==metadata['prefixes'][index]['record'],'Exact case/input/protocol lineage')
    artifacts=row.get('partial_artifacts',[]);need(len({r['path']for r in artifacts})==len(artifacts),'Unique partial artifacts');by={str(bound(r)):r for r in artifacts}
    for name in by:need(Path(name).parent==directory,'Case-only artifact membership')
    reference_pass=[];reference_count=0;decisions={};decision_count=0
    for k in range(4):
        path=directory/f'category-{k}-LOCAL.json'
        if str(path)not in by:continue
        value=run.read(path);reference_metadata(value);check=refs_audit.reference_checks(value);need(value['old_reference']==old_witness['categories'][k],'Exact old category reference');before=run.read(bound(value['old_reference']));same=unchanged_reference(before['diagnostics'],value['diagnostics']);need(value['unchanged_prefix']==same,'Independent unchanged reference evidence witness');reference_pass.append(bool(check['passed']and same['passed']));reference_count+=1
    for outer,amp in run.DECISION_LEVELS:
        path=directory/f'decision-{outer}-{amp}-LOCAL.json'
        if str(path)not in by:continue
        value=run.read(path);witness=refs_audit.decision_checks(dict(reference_checks=value['checks']));rho=np.asarray(value['metadata']['layout_probability']);screen=value['posterior_screen'];need(rho.shape==(4,)and np.isfinite(rho).all()and np.all(rho>=0)and abs(float(rho.sum())-1)<=32*np.finfo(float).eps,'Normalized categorical posterior witnesses')
        need(screen['maximum_layout_probability']==float(rho.max())and screen['layout_TV']==value['checks']['layout_TV']and all(finite_nonnegative(screen[v])for v in('prior_variance','between_layout_fraction','maximum_layout_probability','layout_TV'))and screen['prior_variance']>0 and screen['between_layout_fraction']<=1+32*np.finfo(float).eps,'Finite producer posterior moments and rho maximum')
        need(value['old_decision']==old_witness['decisions'][decision_count],'Exact old decision bank');before=run.read(bound(value['old_decision']));same=unchanged_decision(before['metadata'],value['metadata']);need(value['unchanged_prefix']==same,'Independent unchanged decision evidence witness');decisions[outer,amp]=dict(value=value,passed=bool(witness['passed']and same['passed']));decision_count+=1
    jobs=row.get('jobs',[]);ranking=row.get('ranking');rank_verified=False
    if ranking is not None:
        declared=row.get('categories',[]);need(len(declared)==4 and[v['layout']for v in declared]==list(range(4))and all(v['record']==by[str(directory/f"category-{v['layout']}-LOCAL.json")]and v['qualified']is reference_pass[v['layout']]for v in declared),'Exact case category records/statuses')
        declared=row.get('decision_levels',[]);need(len(declared)==5 and[(v['outer'],v['amplitude'])for v in declared]==list(run.DECISION_LEVELS)and all(v['record']==by[str(directory/f"decision-{v['outer']}-{v['amplitude']}-LOCAL.json")]and v['checks']==decisions[v['outer'],v['amplitude']]['value']['checks']and v['unchanged_prefix']==decisions[v['outer'],v['amplitude']]['value']['unchanged_prefix']for v in declared),'Exact case decision records/checks')
        need(reference_count==4 and decision_count==5,'No ranking without every category/bank witness');bank_ok=all(decisions[v]['passed']for v in run.old.VARIANTS)and all(decisions[v]['value']['unchanged_prefix']['passed']for v in run.DECISION_LEVELS);category_ok=all(reference_pass)
        raw=copy.deepcopy(ranking);raw['qualified']=all(a['qualified']for a in raw['actions']);raw['unique_best_action']=raw['numerical_ties'][0]if raw['qualified']and len(raw['numerical_ties'])==1 else None
        result=action_audit.check_rows(jobs,dict(outcome_relative=1e-4,latent_relative=1e-3),raw);expected=bool(result['qualified']and bank_ok and category_ok);need(ranking['qualified']is expected and row['qualified']is expected and row['status']==('qualified'if expected else'unresolved'),'Whole reference/bank/action admission conjunction')
        if not expected:need(ranking['unique_best_action']is None,'Unresolved case cannot select winner')
        final=decisions[512,64]['value']['posterior_screen'];s=dict(final);s['between_layout_fraction_indicator']=abs(s['between_layout_fraction']-decisions[512,32]['value']['posterior_screen']['between_layout_fraction'])+abs(s['between_layout_fraction']-decisions[256,64]['value']['posterior_screen']['between_layout_fraction']);need(row['posterior_screen']==s,'Recomputed between-layout refinement indicator')
        for item in jobs:
            if(item['job']['outer'],item['job']['amplitude'])==(512,64):need(item['result']['prior_target_variance']==s['prior_variance'],'Same full prior target risk in all8 actions')
        rank_verified=True
    else:need(row['status']=='unresolved','No ranking implies unresolved retained case')
    # The category+decision arrays are producer scalar witnesses, not full moment reconstruction.
    return dict(reference_count=reference_count,references_qualified=sum(reference_pass),decision_count=decision_count,utility_jobs=len(jobs),rank_verified=rank_verified)

from pathlib import Path

def check_builder(metadata,p):
    """Recompute scalar equilibrium gates; bound, never replay, PDE witnesses."""
    need(metadata['prefixes']==p['origin']['old36_prefixes'],'Immutable exact old36 prefix bytes and memberships')
    need(metadata.get('old_negative_preserved')is True and metadata.get('no_new_NH')is True,'New held objective preserves old negative and LINEAR scope')
    need(metadata.get('new_held_equilibria')==72,'All72 paired held equilibria')
    rows=metadata.get('new_held_validations',[]);need(len(rows)==36 and{v['index']for v in rows}==set(range(36)),'All36 paired held-check memberships')
    ids=[v['id']for v in run.balanced.load_specs()[9:]]
    maximum=0.
    for row in rows:
        need(row['load_ids']==ids and row.get('exact_clamp_zeros')is True,'Exact two new held loads/clamp witness')
        residual=row.get('free_residual_l2_N',[]);need(len(residual)==2 and all(refs_audit.finite_le(x,1e-10)for x in residual),'Both held original residual gates')
        need(refs_audit.finite_le(row.get('maximum_modal_parity_m'),2e-12),'New held modal/direct2pm')
        maximum=max(maximum,row['maximum_modal_parity_m'])
        reaction=row.get('equilibrium',[]);need(len(reaction)==2,'Both held reactions retained')
        for j,v in enumerate(reaction):
            for key in('free_residual_N','tolerance_N','force_balance_error_N','force_balance_tolerance_N','moment_balance_error_Nm','moment_balance_tolerance_Nm'):
                need(finite_nonnegative(v.get(key)),'Finite reaction scalar witness')
            passed=bool(v['free_residual_N']<=v['tolerance_N'] and v['force_balance_error_N']<=v['force_balance_tolerance_N'] and v['moment_balance_error_Nm']<=v['moment_balance_tolerance_Nm'] and v.get('exact_clamp')is True)
            need(passed and v.get('passed')is True and v['free_residual_N']<=1e-10,'Recomputed known-load reaction/equilibrium gate')
    need([m['layout']for m in metadata['models']]==list(range(4)),'All4 modal categories')
    for k,m in enumerate(metadata['models']):
        need(m['old_model']==p['origin']['old36_models'][k]['record'],'Hash-bound old model reference')
        v=m['first12_invariance'];need(v.get('all_eigenvalues_byte_identical')is True and v.get('first12_labels_identical')is True and refs_audit.finite_le(v.get('maximum_full_prior_response_difference_bound_m'),2e-14),'All-prior first12 modal response witness')
    bound(metadata['held_response_record']) # byte-only; never deserialize actual held target array
    diagnostic=metadata['closed36_target_separation'];need(diagnostic.get('diagnostic_only_not_gate')is True and diagnostic.get('not_assumed_equal')is True,'Balance remains descriptive, never admission')
    return dict(new_held_equilibria_verified=72,maximum_new_held_modal_parity_m=maximum,old36_prefix_bytes_identical=True,held_balance_is_descriptive_only=True,PDE_and_coefficient_arrays_replayed=False)

def root_closure(p,e,digest):
    closure=run.read(LOCAL/'root-closure.json');expected=[e['controller_pid']]
    if e.get('builder'):expected.append(e['builder']['pid'])
    expected.extend(v['pid']for v in e['workers'])
    need(closure.get('root_exit_code')==0 and closure.get('root_reaped')is True and closure.get('all_owned_PID_and_groups_absent')is True,'Root externally reaped/closed')
    need(closure.get('protocol_SHA256')==digest and closure.get('execution_SHA256')==run.sha(LOCAL/'execution.json') and closure.get('source_SHA256')==p['source_SHA256'],'Exact root closure source/protocol/execution')
    need(len(closure['owned_pids'])==len(expected)and set(closure['owned_pids'])==set(expected),'Exact all5 owned root PID identities')
    for pid in expected:run.absent(pid)
    return run.record(LOCAL/'root-closure.json')

def audit(digest):
    target=LOCAL/'independent-terminal-review.json';need(not target.exists(),'No terminal audit overwrite');need(isinstance(digest,str)and len(digest)==64 and all(c in'0123456789abcdef'for c in digest)and(EXPECTED_PROTOCOL is None or digest==EXPECTED_PROTOCOL),'Exact prospective audit protocol');p=run.validate(digest);e=run.read(LOCAL/'execution.json');need(e.get('terminal')is True,'Terminal controller required');need(e['protocol_SHA256']==digest and e['release']==run.record(LOCAL/'release.json'),'Controller source/protocol/release');run.absent(e['controller_pid'])
    closure=root_closure(p,e,digest);technical_clean=bool(e.get('completed')is True and'failure'not in e);builder_checks={};metadata=None;checks=[];rows=[dict(index=s['index'],case=s)for s in p['worklist']];bindings=[]
    builder=e.get('builder')
    if builder:
        run.absent(builder['pid']);need(builder.get('reaped')is True,'Builder reaped');technical_clean=technical_clean and builder.get('exit_code')==0 and builder.get('cleanup_exit_code')==0
        if builder.get('result'):
            bpath=bound(builder['result']);metadata=run.read(bpath);need(metadata['pid']==builder['pid']and metadata['protocol_SHA256']==digest and metadata['release']==run.record(LOCAL/'release.json'),'Builder context');need(metadata.get('completed')is True and len(metadata['models'])==4 and len(metadata['prefixes'])==36,'Complete builder prerequisite');run.verify_records(metadata)
            need([m['layout']for m in metadata['models']]==list(range(4))and[v['index']for v in metadata['prefixes']]==list(range(36)),'All4 models/all36 prefixes')
            for m in metadata['models']:
                d=m['all_mode_diagnostics'];need(d['all_boundary_modes_retained']is True and d['interior_basis_truncated']is False and d['foundation_threshold_applied']is False and d['exact_particular_1_over_E_term']is True and m['maximum_prefix_parity_m']<=2e-12,'Allmode/particular and bound producer prefix parity')
            builder_checks=check_builder(metadata,p)
            need(metadata['wall_s']<=p['resources']['builder_wall_s']and metadata['peak_RSS_bytes']<=p['resources']['builder_RSS_bytes'],'Builder resource caps');bindings.append(builder['result'])
    seen=set()
    for worker in e['workers']:
        slot=worker['slot'];need(type(slot)is int and slot in range(3)and slot not in seen and worker['physical_UUID']==p['devices'][slot],'Unique exact worker slot/UUID');seen.add(slot);run.absent(worker['pid']);need(worker.get('reaped')is True,'Worker cleanup/reap');technical_clean=technical_clean and worker.get('exit_code')==0 and worker.get('cleanup_exit_code')==0
        if not worker.get('result'):continue
        need(metadata is not None,'Worker requires complete builder');v=run.read(bound(worker['result']));need(v['pid']==worker['pid']and v['slot']==slot and v['physical_UUID']==worker['physical_UUID']and v['protocol_SHA256']==digest and v['release']==run.record(LOCAL/'release.json')and v['builder']==builder['result'],'Worker exact launched context');need([q['index']for q in v['queue']]==p['split'][slot],'All12 assigned queue slots retained')
        need(v['wall_s']<=p['resources']['worker_wall_s']and v['peak_RSS_bytes']<=p['resources']['worker_RSS_bytes']and worker.get('peak_RSS_bytes',0)<=p['resources']['worker_RSS_bytes']and worker.get('peak_owned_VRAM_bytes',0)<=p['resources']['owned_VRAM_bytes'],'Owned worker bounds');technical_clean=technical_clean and v.get('completed')is True and'failure'not in v
        case_seen=set()
        for item in v['cases']:
            index=item['index'];need(type(index)is int and index in p['split'][slot]and index not in case_seen,'Unique assigned case');case_seen.add(index);path=bound(item['record']);need(path==LOCAL/f'case-{index:02d}/result-LOCAL.json','Exact case path');row=run.read(path);check=check_case(row,p['worklist'][index],metadata,digest,p['origin']);need(item['status']==row['status'],'Case status journal');rows[index]=row;checks.append(check);bindings.append(item['record'])
        if v.get('completed')is True:need(case_seen==set(p['split'][slot]),'Completed worker keeps all12 cases')
        bindings.append(worker['result'])
    computed=independent_adequacy(rows)
    if e.get('completed')is True:need(len(seen)==3 and len(checks)==36 and e['adequacy']==computed,'Recomputed all36 adequate/unresolved declaration')
    need(e['wall_s']<=p['resources']['global_wall_s']and e.get('peak_aggregate_RSS_bytes',0)<=p['resources']['aggregate_RSS_bytes'],'Global resource caps')
    summary=dict(integrity_passed=True,terminal=True,execution_clean=technical_clean,planned_cases=36,retained_cases=36,completed_case_records=len(checks),planned_category_references=144,completed_category_references=sum(c['reference_count']for c in checks),qualified_category_references=sum(c['references_qualified']for c in checks),planned_decision_banks=180,completed_decision_banks=sum(c['decision_count']for c in checks),planned_utility_jobs=1728,completed_utility_jobs=sum(c['utility_jobs']for c in checks),independently_recomputed_rankings=sum(c['rank_verified']for c in checks),adequacy=computed,builder_checks=builder_checks,producer_moment_witnesses_only=True,full_reference_moment_arrays_independently_reconstructed=False,wall_s=e['wall_s'],peak_aggregate_RSS_bytes=e.get('peak_aggregate_RSS_bytes',0),no_model_PDE_GPU_NPZ_or_posterior_replay=True)
    run.write(target,dict(summary=summary,source_SHA256={s:run.sha(ROOT/s)for s in(SOURCE,TEST)},protocol=run.record(LOCAL/'protocol.json'),release=run.record(LOCAL/'release.json'),execution=run.record(LOCAL/'execution.json'),root_closure=closure,scientific_scalar_bindings=bindings));return summary

def inventory():
    review=run.read(LOCAL/'independent-terminal-review.json');need(review['summary']['integrity_passed'],'Independent integrity audit');directory=ROOT/'reports/fixture-balanced-decision-screen-v01';directory.mkdir(parents=True,exist_ok=True);files=[]
    for path,value in((directory/'summary.json',review['summary']),(directory/'source-provenance.json',dict(protocol=review['protocol'],release=review['release'],audit_source_SHA256=review['source_SHA256'],execution=review['execution'],root_closure=review['root_closure']))):
        need(not path.exists(),'No public report overwrite');run.write(path,value);files.append(run.record(path))
    files.extend(run.record(ROOT/s)for s in(SOURCE,TEST));path=LOCAL/'publication-inventory-completed.json';need(not path.exists(),'No inventory overwrite');run.write(path,dict(files=files,raw_coordinates_observations_category_weights_evidences_action_risk_vectors_included=False));return run.record(path)
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['audit','inventory']);ap.add_argument('--sha');a=ap.parse_args();print(audit(a.sha)if a.action=='audit'else inventory())
