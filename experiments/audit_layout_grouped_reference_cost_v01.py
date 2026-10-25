"""Prospective scalar/provenance audit; never runs a reference/model or NPZ.

Original fixed-grid failure stays immutable. This procedure can classify only
one-action LINEAR development convergence, never admit all8 ranks or physics.
"""
import argparse,json
from pathlib import Path
import numpy as np
from experiments import run_layout_grouped_reference_cost_v01 as run
ROOT=run.ROOT;LOCAL=run.LOCAL
SOURCE='experiments/audit_layout_grouped_reference_cost_v01.py';TEST='tests/test_audit_layout_grouped_reference_cost_v01.py'
EXPECTED_PROTOCOL='7250194040bc23d9866c6647c4df84b941df07c003fbd567bfd5f1970e9f4200'
GATES=dict(logZ=.01,CDF=.005,quantile=.002,unit_mean=.001,target_absolute_m=2e-6,target_relative_SD=.01)

def need(value,message):
    if not value:raise ValueError(message)
def finite_le(value,limit):return isinstance(value,(float,int))and not isinstance(value,bool)and np.isfinite(value)and 0<=value<=limit
def close(a,b):return bool(np.allclose(a,b,rtol=1e-12,atol=1e-24,equal_nan=False))
def bound(record):
    path=ROOT/record['path'];need(run.record(path)==record,'Changed bound output/input bytes');return path

def reference_checks(row):
    d=row['diagnostics'];need(d.get('schema')=='conditional-amplitude-v02'and d.get('gates')==GATES,'Original frozen continuous-reference schema/gates')
    expected={(512,128),(512,256),(512,512),(128,512),(256,512)};levels=d.get('levels',[]);need(len(levels)==5 and{(v['outer'],v['inner'])for v in levels}==expected,'All original5reference levels retained')
    ok=row.get('qualified')is True;strict=True;max_field=0.
    for axis in('inner','outer'):
        history=d.get('successive',{}).get(axis,[]);need([(v['coarse'],v['fine'])for v in history]==[(128,256),(256,512)],'Unchanged two successive reference-axis witnesses')
        r=history[-1];ok=ok and r.get('passed')is True
        for key,limit in [('log_evidence_difference',.01),('maximum_CDF_difference',.005),('maximum_unit_quantile_difference',.002),('maximum_unit_mean_difference',.001)]:ok=ok and finite_le(r.get(key),limit)
        for field in('predictive_mean','predictive_SD'):
            value=r.get('maximum_'+field+'_difference_m');strict=strict and finite_le(value,2e-6)
            if isinstance(value,(float,int))and np.isfinite(value):max_field=max(max_field,value)
    a=d.get('independent_adaptive',{});ok=ok and a.get('passed')is True and a.get('epsrel')==[1e-6,1e-8]and a.get('CDF_checks_per_epsrel')==6
    for key,limit in [('log_evidence_difference',.01),('maximum_CDF_error',.005),('adaptive_refinement_log_evidence_difference',.01),('adaptive_refinement_maximum_CDF_difference',.005)]:ok=ok and finite_le(a.get(key),limit)
    errors=a.get('maximum_QUADPACK_relative_error_estimates',{});ok=ok and set(errors)=={'1e-06','1e-08'}and all(finite_le(v,1e-5)for v in errors.values());passed=bool(ok and strict)
    need(row['strict']['passed']is passed,'Reference original+strict2um admission matches scalar witnesses')
    return dict(passed=passed,maximum_field_axis_error_m=max_field,adaptive_passed=a.get('passed')is True)

def decision_checks(row):
    c=row['reference_checks'];need(len(c.get('per_category',[]))==4,'Every category decision quadrature comparison')
    passed=True;maximum_field=maximum_logz=0.
    for r in c['per_category']:
        ok=finite_le(r.get('log_evidence'),.01)and finite_le(r.get('mean_m'),2e-6)and finite_le(r.get('SD_m'),2e-6);need(r.get('passed')is bool(ok),'Category decision comparison flag');passed=passed and ok
        maximum_logz=max(maximum_logz,r['log_evidence']);maximum_field=max(maximum_field,r['mean_m'],r['SD_m'])
    passed=bool(passed and finite_le(c.get('mixture_mean_error_m'),2e-6)and finite_le(c.get('mixture_SD_error_m'),2e-6)and finite_le(c.get('layout_TV'),.01));need(c.get('passed')is passed,'Mixed48moment/evidence comparison flag')
    return dict(passed=passed,maximum_field_error_m=max(maximum_field,c['mixture_mean_error_m'],c['mixture_SD_error_m']),maximum_evidence_error=maximum_logz,layout_TV=c['layout_TV'])

def outcome_checks(row):
    levels=row['levels'];need(len(levels)==2 and[v['order']for v in levels]==[8,16],'Fixed independent outcome refinement8/16')
    c,f=levels;error=abs(c['expected_target_variance']-f['expected_target_variance'])+f['omitted_risk_bound'];passed=bool(error<=1e-4*f['prior_target_variance'])
    need(close(error,row['outcome_check']['error'])and row['outcome_check']['passed']is passed and row['outcome_check']['gate_relative_to_prior']==1e-4,'Independent outcome error/flag calculation')
    for v in levels:need(finite_le(v['expected_target_variance'],v['prior_target_variance'])and close(v['variance_reduction'],v['prior_target_variance']-v['expected_target_variance']),'Positive target risk/reduction identity')
    return dict(passed=passed,error=error,final=f)

def risk_refinement_checks(utilities,reported):
    risk={(r['job']['outer'],r['job']['amplitude']):r['levels'][-1]for r in utilities};need(set(risk)=={(512,32),(256,64),(512,64)},'Complete ONEaction risk quadrature worklist')
    fine=risk[512,64];checks={}
    for axis,key in(('amplitude',(512,32)),('outer',(256,64))):
        error=abs(risk[key]['expected_target_variance']-fine['expected_target_variance']);relative=error/fine['prior_target_variance'];passed=bool(error<=1e-3*fine['prior_target_variance']);r=reported[axis]
        need(close(error,r['error'])and close(relative,r['relative_to_prior'])and r['passed']is passed,'Independent latent-bank risk refinement calculation');checks[axis]=dict(passed=passed,error_m2=error,relative_to_prior=relative)
    return checks

def audit():
    output=LOCAL/'independent-terminal-review.json';need(not output.exists(),'No terminal review overwrite')
    p=run.read(LOCAL/'protocol.json');r=run.read(LOCAL/'release.json');e=run.read(LOCAL/'root-execution.json')
    need(run.sha(LOCAL/'protocol.json')==EXPECTED_PROTOCOL and p['source_SHA256']==run.sources(),'Exact frozen source/protocol')
    need(r.get('approved')is True and r.get('protocol_SHA256')==EXPECTED_PROTOCOL and r.get('source_SHA256')==p['source_SHA256']and r.get('origin')==p['origin']and r.get('external_watchdog_600s_4GiB')is True,'Exact root release')
    run.old.verify_mechanics_receipt(p['origin']);need(e.get('reaped')is True and isinstance(e.get('cleanup_exit_code'),int)and e.get('protocol_SHA256')==EXPECTED_PROTOCOL,'Final externally owned/reaped execution')
    for pid in(e['controller_pid'],e['child_pid']):run.old.mechanical_screen.shared.old.absent(pid)
    w=run.read(bound(e['worker']));need(w.get('pid')==e['child_pid']and w.get('protocol_SHA256')==EXPECTED_PROTOCOL and w.get('release')==run.record(LOCAL/'release.json'),'Worker protocol/PID/release')
    need(e.get('wall_cap_s')==600 and e.get('RSS_cap_bytes')==4*1024**3,'Frozen external600s/4GiB cap')
    need(w.get('all8_action_ranks')=='PENDING_NOT_ADMITTED'and w.get('new_RNG_calls')==0 and w.get('nonlinear_PDE_solves')==0 and w.get('generating_or_held_truth_reads')==0,'Oneaction/same-data source scope')
    refs=[];decisions=[];utilities=[];bindings=[]
    need(len(w.get('prefix_categories',[]))==4 and{v['layout']for v in w['prefix_categories']}=={s['id']for s in run.old.mechanics.LAYOUTS},'All four categories retained including failures')
    for k,entry in enumerate(w['prefix_categories']):
        if entry['status']=='pending':continue
        path=bound(entry['record']);need(path==LOCAL/f'prefix-layout-{k}-diagnostics-LOCAL.json','Fixed category output path');value=run.read(path);check=reference_checks(value);need(entry['status']==('qualified'if check['passed']else'unresolved'),'Category status consistent');refs.append(check);bindings.append(entry['record'])
    seen=set()
    for entry in w['decision_levels']:
        key=(entry['outer'],entry['amplitude']);need(key in run.DECISION_LEVELS and key not in seen,'Fixed unique decision quadratures');seen.add(key);path=bound(entry['record']);need(path==LOCAL/f'decision-{key[0]}-{key[1]}-LOCAL.json','Fixed decision output path');value=run.read(path);need(value['reference_checks']==entry['checks'],'Decision checkpoint equality');decisions.append(dict(outer=key[0],amplitude=key[1],**decision_checks(value)));bindings.append(entry['record'])
    seen=set()
    for entry in w['utility']:
        job=entry['job'];key=(job['outer'],job['amplitude']);need(job in p['utility_worklist']and key not in seen,'Unique predeclared singleaction jobs');seen.add(key);path=bound(entry['record']);need(path==LOCAL/f'utility-{key[0]}-{key[1]}-LOCAL.json','Fixed utility output path');value=run.read(path);need(value==dict(job=job,levels=entry['levels'],outcome_check=entry['check']),'Utility checkpoint equality');outcome_checks(value);utilities.append(value);bindings.append(entry['record'])
    complete=bool(w.get('completed')is True);refinements={};qualified=False
    if complete:
        need(len(refs)==4 and len(decisions)==5 and len(utilities)==3,'Completed cost keeps every planned category/bank/job')
        refinements=risk_refinement_checks(utilities,w['decision_risk_refinements']);required={(512,32),(256,64),(512,64)};qualified=all(v['passed']for v in refs)and all(v['passed']for v in decisions if(v['outer'],v['amplitude'])in required)and all(outcome_checks(v)['passed']for v in utilities)and all(v['passed']for v in refinements.values())
        need(w['qualified']is bool(qualified),'Whole numerical admission consistent')
    else:need(w.get('qualified')is False,'Incomplete pilot cannot be numerically qualified')
    clean=bool(e.get('completed')is True and e.get('exit_code')==0 and e.get('cleanup_exit_code')==0 and'failure'not in e);need(e.get('qualified')is bool(clean and qualified),'External technical failure cannot imply qualification')
    summary=dict(completed=True,integrity_passed=True,execution_clean=clean,continuous_categories_qualified=sum(v['passed']for v in refs),categories_retained=4,decision_banks_completed=len(decisions),oneaction_jobs_completed=len(utilities),oneaction_numerically_qualified=bool(clean and qualified),all8_ranks_pending=True,policy_use_blocked=True,original_grid_negative_preserved=True,scope='Same saved prefix LINEAR DEVELOPMENT oneaction/reference cost only; no nonlinear/spatial/calibration/active-advantage admission',external_wall_s=e['wall_s'],peak_RSS_bytes=max(e.get('peak_RSS_bytes',0),w.get('peak_RSS_bytes',0)),maximum_continuous_field_error_m=max([v['maximum_field_axis_error_m']for v in refs]or[0.]),maximum_decision_moment_error_m=max([v['maximum_field_error_m']for v in decisions]or[0.]),risk_refinements=refinements,no_model_reference_or_array_replay=True)
    run.write(output,dict(summary=summary,source_SHA256={name:run.sha(ROOT/name)for name in(SOURCE,TEST)},protocol=run.record(LOCAL/'protocol.json'),release=run.record(LOCAL/'release.json'),execution=run.record(LOCAL/'root-execution.json'),worker=e['worker'],scientific_scalar_bindings=bindings));return summary

def publication_inventory():
    """Only after terminal audit: compact aggregate flags/errors and hashes."""
    path=LOCAL/'independent-terminal-review.json';review=run.read(path);summary=review['summary']
    need(summary.get('completed')is True and summary.get('integrity_passed')is True,'Completed independent scalar audit required')
    report=ROOT/'reports/layout-grouped-reference-cost-v01';report.mkdir(parents=True,exist_ok=True)
    safe_keys=('completed','integrity_passed','execution_clean','continuous_categories_qualified','categories_retained','decision_banks_completed','oneaction_jobs_completed','oneaction_numerically_qualified','all8_ranks_pending','policy_use_blocked','original_grid_negative_preserved','scope','external_wall_s','peak_RSS_bytes','maximum_continuous_field_error_m','maximum_decision_moment_error_m','risk_refinements','no_model_reference_or_array_replay')
    safe={key:summary[key]for key in safe_keys};files=[]
    for name,value in[('summary.json',safe),('source-provenance.json',dict(source_SHA256=review['source_SHA256'],protocol=review['protocol'],release=review['release'],execution=review['execution'],worker=review['worker'],independent_review=run.record(path),raw_observations_and_category_values_LOCAL=True))]:
        target=report/name;need(not target.exists(),'No public artifact overwrite');run.write(target,value);files.append(run.record(target))
    files+=[run.record(ROOT/SOURCE),run.record(ROOT/TEST)]
    inventory=LOCAL/'publication-inventory-completed.json';need(not inventory.exists(),'No inventory overwrite');run.write(inventory,dict(status='TERMINAL_ONEACTION_ONLY',files=files,no_raw_publication=True));return run.record(inventory)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['audit','inventory'],nargs='?',default='audit');args=p.parse_args();print(json.dumps(audit()if args.action=='audit'else publication_inventory()))
