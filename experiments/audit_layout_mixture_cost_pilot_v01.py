"""Independent scalar/byte audit of the failed finite-grid development pilot.

No model, parameter quadrature, FE response or observation array is evaluated.
The original negative is immutable and cannot become a policy admission here.
"""
import json,os
from pathlib import Path
import numpy as np
from experiments import run_layout_mixture_cost_pilot_v01 as original
ROOT=original.ROOT;LOCAL=original.LOCAL
SOURCE='experiments/audit_layout_mixture_cost_pilot_v01.py';TEST='tests/test_audit_layout_mixture_cost_pilot_v01.py'
EXPECTED_PROTOCOL='f4527a99103ee1230f069aa1e857dde0b404d9d37ed15981ccdf94e74f2e6e6e'
EXPECTED_PIDS=(484606,484607)
sha,record=original.sha,original.record

def require(value,message):
    if not value:raise ValueError(message)
def close(a,b):return bool(np.allclose(a,b,rtol=1e-12,atol=1e-24,equal_nan=False))
def numerical_checks(worker,levels,actions):
    require(worker.get('completed')is True and worker.get('qualified')is False and'failure'not in worker,'Completed original numerical negative required')
    work=[dict(order=n,action=a)for n in(9,17,33)for a in range(8)]
    require(worker['all_worklist']==work and len(worker['completed_actions'])==24,'ALL24 jobs retained')
    require(len(levels)==3 and len(actions)==24,'Exact three parameter levels and all8 actions')
    winners=[];largest_outcome=0.;summaries=[]
    for n,summary in zip((9,17,33),levels):
        require(summary['parameter_order']==n and summary['hypotheses']==4*n*n and summary['outcome_qualified']is True,'Fixed four-layout bank and outcome refinement')
        rho=np.asarray(summary['layout_probability']);logs=np.asarray(summary['log_evidence']);require(rho.shape==(4,)and np.isfinite(rho).all()and np.all(rho>=0)and abs(rho.sum()-1)<1e-14,'All normalized category posterior masses')
        independent=np.exp(logs-max(logs));independent/=independent.sum();require(close(rho,independent),'Normalized original equal-category evidences')
        risks=[];errors=[]
        for a in range(8):
            value=actions[(n,a)];require(value['action']==a and value['qualified']is True and len(value['levels'])==2,'Every action outcome control passed')
            c,f=value['levels'];require(c['order']==8 and f['order']==16,'Original outcome orders8/16')
            error=abs(f['expected_target_variance']-c['expected_target_variance'])+f['omitted_risk_bound'];require(close(error,value['successive_error'])and error<=1e-4*f['prior_target_variance'],'Independent outcome convergence calculation')
            require(0<=f['expected_target_variance']<=f['prior_target_variance']and close(f['variance_reduction'],f['prior_target_variance']-f['expected_target_variance']),'Finite Bayes risk bookkeeping')
            risks.append(f['expected_target_variance']);errors.append(error);largest_outcome=max(largest_outcome,error)
        risks=np.asarray(risks);errors=np.asarray(errors);best=int(np.argmin(risks));ties=np.flatnonzero(risks-risks[best]<=errors+errors[best]).tolist()
        require(close(risks,summary['expected_target_risk'])and close(errors,summary['risk_error_witness']),'Independent all-action risk aggregation')
        require(summary['ties']==ties and summary['unique_best_action']==(best if len(ties)==1 else None),'Numerical tie/winner semantics')
        require(close(summary['fixed_action_regret'],risks-risks[best])and close(summary['uniform_random_expected_regret'],np.mean(risks)-risks[best]),'Finite predicted regret, not held-truth score')
        winners.append(summary['unique_best_action']);summaries.append(summary)
    require(worker['levels']==levels,'Saved scalar levels identical to original worker summary')
    checks=[]
    for c,f,saved in zip(levels,levels[1:],worker['parameter_refinements']):
        risk=float(np.max(abs(np.asarray(c['expected_target_risk'])-f['expected_target_risk'])));tv=float(np.sum(abs(np.asarray(c['layout_probability'])-f['layout_probability']))/2);scale=max(c['total_target_variance'],f['total_target_variance']);passed=risk<=1e-3*scale and tv<=.01
        require(close(risk,saved['maximum_risk_difference'])and close(tv,saved['layout_probability_TV'])and saved['passed']is bool(passed),'Independent parameter-refinement witness')
        checks.append(dict(coarse=c['parameter_order'],fine=f['parameter_order'],maximum_risk_difference=risk,relative_to_prior_variance=risk/scale,layout_TV=tv,passed=bool(passed)))
    require(len(checks)==2 and checks[-1]['passed']is False,'Original final grid failure must remain failed')
    return dict(parameter_refinements=checks,all24_outcome_controls_passed=True,maximum_outcome_error=largest_outcome,unique_winners_by_level=winners,winner_changes=sum(a!=b for a,b in zip(winners,winners[1:])),last_between_layout_variance_fraction=levels[-1]['between_layout_variance_fraction'],last_maximum_layout_probability=levels[-1]['maximum_layout_probability'])

def audit():
    require(not(LOCAL/'independent-negative-review.json').exists(),'No audit overwrite')
    p=json.loads((LOCAL/'protocol.json').read_text());r=json.loads((LOCAL/'release.json').read_text());w=json.loads((LOCAL/'execution.json').read_text());e=json.loads((LOCAL/'root-execution.json').read_text())
    require(sha(LOCAL/'protocol.json')==EXPECTED_PROTOCOL and p['source_SHA256']==original.sources(),'Exact frozen original protocol/source closure')
    for path,digest in p['source_SHA256'].items():require(sha(ROOT/path)==digest,'All frozen source bytes unchanged')
    require(r['approved']is True and r['protocol_SHA256']==EXPECTED_PROTOCOL and r['source_SHA256']==p['source_SHA256']and r['mechanics_admission']==p['mechanics_admission']and r['external_watchdog_120s_4GiB']is True,'Original exact root release')
    original.verify_mechanics_receipt(p['mechanics_admission'])
    require(e.get('completed')is True and e.get('reaped')is True and e.get('exit_code')==0 and e.get('cleanup_exit_code')==0 and e.get('protocol_SHA256')==EXPECTED_PROTOCOL and e['controller_pid']==EXPECTED_PIDS[0]and e['child_pid']==EXPECTED_PIDS[1],'Original clean external owned closure')
    for pid in EXPECTED_PIDS:original.mechanical_screen.shared.old.absent(pid)
    require(e['wall_cap_s']==120 and e['RSS_cap_bytes']==4*1024**3 and e['wall_s']<=120 and max(e['peak_RSS_bytes'],w['peak_RSS_bytes'])<4*1024**3,'Frozen resource budget respected')
    require(w['pid']==e['child_pid']and w['protocol_SHA256']==EXPECTED_PROTOCOL and w['release']==record(LOCAL/'release.json'),'Worker identity/protocol/release')
    require(w['new_RNG_calls']==0 and w['nonlinear_physics_solves']==0 and w['finite_element_modal_builds']==4 and w['generating_future_truth_evaluations']==0,'Original source-bound development scope')
    require(record(ROOT/w['prefix_observation']['path'])==w['prefix_observation'],'Saved prefix bytes hashed, never opened')
    actions={};pins={}
    for item in w['completed_actions']:
        n,a=item['order'],item['action'];path=LOCAL/f'order-{n}-action-{a}-LOCAL.json'
        require(record(path)==item['result']and(n,a)not in actions,'All exact unique action commitments');actions[(n,a)]=json.loads(path.read_text());pins[str(path.relative_to(ROOT))]=sha(path)
    require(set(actions)=={(n,a)for n in(9,17,33)for a in range(8)},'No omitted development actions')
    levels=[]
    for n in(9,17,33):
        path=LOCAL/f'order-{n}-summary-LOCAL.json';levels.append(json.loads(path.read_text()));pins[str(path.relative_to(ROOT))]=sha(path)
    checked=numerical_checks(w,levels,actions)
    for name in('protocol.json','release.json','execution.json','root-execution.json'):pins[str((LOCAL/name).relative_to(ROOT))]=sha(LOCAL/name)
    result=dict(completed=True,integrity_passed=True,numerical_qualified=False,policy_use_blocked=True,original_negative_preserved=True,scope='One deterministic finite LINEAR development case. No continuous posterior, physical spatial/nonlinear admission or adaptive advantage.',all_jobs_retained=24,controller_worker_closed=True,external_wall_s=e['wall_s'],peak_RSS_bytes=max(e['peak_RSS_bytes'],w['peak_RSS_bytes']),no_model_or_array_replay=True,source_SHA256={SOURCE:sha(ROOT/SOURCE),TEST:sha(ROOT/TEST)},input_SHA256=pins,checks=checked)
    original.write(LOCAL/'independent-negative-review.json',result);return result

if __name__=='__main__':print(json.dumps(audit()))
