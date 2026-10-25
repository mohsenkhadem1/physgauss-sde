"""Manufactured scalar records only; no actual model/observations."""
from copy import deepcopy
import numpy as np
import pytest
from experiments import audit_layout_mixture_cost_pilot_v01 as m

def fixture():
    levels=[];actions={}
    for n,k in[(9,0),(17,0),(33,2)]:
        risk=np.ones(8)*2e-9;risk[k]=1e-9 if n<33 else .5e-9;error=np.zeros(8)
        levels.append(dict(parameter_order=n,hypotheses=4*n*n,outcome_qualified=True,layout_probability=[.25]*4,log_evidence=[0.]*4,expected_target_risk=risk.tolist(),risk_error_witness=error.tolist(),ties=[k],unique_best_action=k,fixed_action_regret=(risk-risk[k]).tolist(),uniform_random_expected_regret=float(np.mean(risk)-risk[k]),total_target_variance=3e-9,between_layout_variance_fraction=.5,maximum_layout_probability=.25))
        for a in range(8):
            row=dict(expected_target_variance=risk[a],prior_target_variance=3e-9,variance_reduction=3e-9-risk[a],omitted_risk_bound=0.)
            actions[n,a]=dict(action=a,qualified=True,successive_error=0.,levels=[dict(row,order=8),dict(row,order=16)])
    refinements=[]
    for c,f in zip(levels,levels[1:]):
        diff=max(abs(np.array(c['expected_target_risk'])-f['expected_target_risk']));refinements.append(dict(maximum_risk_difference=diff,layout_probability_TV=0.,passed=bool(diff<=3e-12)))
    worker=dict(completed=True,qualified=False,all_worklist=[dict(order=n,action=a)for n in(9,17,33)for a in range(8)],completed_actions=list(range(24)),levels=levels,parameter_refinements=refinements)
    return worker,levels,actions

def test_independent_negative_and_winner_change_retained():
    w,l,a=fixture();r=m.numerical_checks(w,l,a);assert r['unique_winners_by_level']==[0,0,2]and r['winner_changes']==1 and not r['parameter_refinements'][-1]['passed']

def test_omitted_action_or_false_pass_refused():
    w,l,a=fixture();bad=dict(a);bad.pop((33,7))
    with pytest.raises(ValueError):m.numerical_checks(w,l,bad)
    w['qualified']=True
    with pytest.raises(ValueError,match='negative'):m.numerical_checks(w,l,a)

def test_parameter_gate_or_risk_tamper_refused():
    w,l,a=fixture();w['parameter_refinements'][-1]['passed']=True
    with pytest.raises(ValueError,match='parameter-refinement'):m.numerical_checks(w,l,a)
    w,l,a=fixture();l[2]['uniform_random_expected_regret']=1.
    with pytest.raises(ValueError,match='regret'):m.numerical_checks(w,l,a)
