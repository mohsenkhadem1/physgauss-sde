import numpy as np
import pytest
from scripts.physical_refinement_v15 import development_policy as module


def test_update_rebuilds_prior_and_rejects_duplicate():
    prior=np.array([.3,.7]);initial=np.log([.6,.2]);terms={'a':np.log([.1,.9]),'b':np.log([.8,.4])}
    expected=prior*np.exp(initial+terms['a']+terms['b']);expected/=expected.sum()
    np.testing.assert_allclose(module.update(prior,initial,terms,['a','b']),expected)
    np.testing.assert_allclose(module.update(prior,initial,terms,['b','a']),expected)
    with pytest.raises(ValueError):module.update(prior,initial,terms,['a','a'])


def test_selector_cannot_read_unselected_likelihood(monkeypatch):
    class Unreadable:
        def __array__(self,*a,**kw):raise AssertionError('Unselected observation consumed')
    calls=[]
    def choose(weights,queries,design,policy,**kw):
        calls.append(weights.copy());return ('a' if not kw['used'] else 'b'),{'policy':policy}
    monkeypatch.setattr(module,'choose',choose)
    terms={'a':np.log([.1,.9]),'b':np.log([.8,.4]),'unused':Unreadable()}
    history,weights,diagnostics=module.select_sequence(np.array([.5,.5]),np.zeros(2),terms,[],None,'physical_risk',seed=7,fantasies=8)
    assert history==['a','b'] and len(diagnostics)==2
    np.testing.assert_allclose(calls[0],[.5,.5]);np.testing.assert_allclose(calls[1],[.1,.9])
    np.testing.assert_allclose(weights,[.08/.44,.36/.44])


def test_fixed_policy_is_camera_track_conditional_and_stable():
    rows=[]
    for camera in ('left','oblique'):
        for track in ('V','VF'):
            for pair,loss in [(('view_right','probe_a'),2),(('view_right','probe_b'),1 if track=='VF' else 3),
                              (('view_right','view_left'),4),(('probe_a','probe_b'),5)]:
                rows.append(dict(initial_camera=camera,track=track,sequence=pair,metrics={'normalized_design_mse':loss}))
    result=module.fit_fixed(rows)
    assert result[('left','V','mixed')]==('view_right','probe_a')
    assert result[('left','VF','mixed')]==('view_right','probe_b')
    assert len(result)==12


def test_random_scores_average_errors_not_predictions():
    result=module.average_metrics([{'metrics':{'normalized_design_mse':1.}}, {'metrics':{'normalized_design_mse':9.}}])
    assert result['normalized_design_mse']==5.


def test_followon_all_pair_baseline_and_probe_expectation():
    import itertools
    from scripts.physical_refinement_v15.development_policy_followon import additional_baselines
    identity=dict(world='development-0',initial_camera='left',track='V',repeat=0)
    names=['view_right','view_oblique','probe_a','probe_b','probe_c','probe_d']
    pairs=[dict(**identity,sequence=list(pair),metrics={'normalized_design_mse':float(i)}) for i,pair in enumerate(itertools.combinations(names,2))]
    rows,fixed=additional_baselines(dict(pair_rows=pairs,rows=[dict(**identity,policy='initial_only')]))
    assert fixed[0]['sequence']==['view_right','view_oblique']
    assert rows[0]['metrics']['normalized_design_mse']==0
    assert rows[1]['pair_count']==6
    assert rows[1]['metrics']['normalized_design_mse']==np.mean([r['metrics']['normalized_design_mse'] for r in pairs if all(x.startswith('probe') for x in r['sequence'])])


def test_collapsing_identical_shape_hypotheses_preserves_proper_scores():
    from scripts.physical_refinement_v15.predictive_scores import marginal_scores
    values=np.array([[0.,1.],[0.,1.],[2.,3.],[2.,3.]])
    weights=np.array([.1,.2,.3,.4]);target=np.array([1.,2.])
    a=marginal_scores(values,weights,target);b=marginal_scores(values[[0,2]],[.3,.7],target)
    for key in ('crps_mean','central90_coordinate_coverage','mean_interval_width'):assert np.isclose(a[key],b[key])


def test_real_shape_contract_for_proper_scores():
    from scripts.physical_refinement_v15.predictive_scores import marginal_scores
    shapes=np.zeros((2,3,81,3));targets=np.zeros((3,249));weights=np.array([.4,.6])
    score=marginal_scores(shapes.reshape(2,3,243),weights,targets[:,:243])
    assert score['crps_mean']==0 and score['central90_coordinate_coverage']==1
