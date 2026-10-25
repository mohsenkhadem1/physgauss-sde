import copy,pytest,numpy as np
from experiments import campaign_fixture_gaussian_decision_screen_v01 as h

def records():return [dict(index=s['index'],case=s,ranking=dict(qualified=True,all8_retained=True,numerical_ties=[0],unique_best_action=0))for s in h.worklist()]
def test_fixed36_center_timing_checkpoint_not_filter():
    rows=h.worklist();assert len(rows)==36 and rows[0]['E_Pa']==4000 and len({(r['layout'],r['E_Pa'],r['K_N_m'])for r in rows})==36
    assert h.summarize(records())['universal_unique_ranking']
def test_all36_failures_and_ties_preserved_in_denominator():
    data=records();data[0].pop('ranking');data[1]['ranking'].update(numerical_ties=[1,2],unique_best_action=None);data[2]['ranking'].update(numerical_ties=[3],unique_best_action=3)
    out=h.summarize(data);assert out['planned_cases']==36 and out['qualified_cases']==35 and out['tied_cases']==1 and out['ranking_heterogeneity_observed']and not out['universal_unique_ranking']
    with pytest.raises(ValueError,match='36'):h.summarize(data[:-1])
def test_policy_input_is_only_prefix_and_no_mutation():
    a=np.arange(4.);b=h.prefix_observation(a);b[0]=99;assert a[0]==0
    for x in(np.zeros(60),np.zeros((4,1)),[0,0,np.nan,0]):
        with pytest.raises(ValueError):h.prefix_observation(x)
def test_no_forced_winner_or_changed_case():
    data=records();data[0]['ranking']['numerical_ties']=[0,1]
    with pytest.raises(ValueError,match='winner'):h.summarize(data)
    data=records();data[0]['case']=dict(data[0]['case'],layout=3)
    with pytest.raises(ValueError,match='membership'):h.summarize(data)

def adequate_records():
    data=records()
    for i,row in enumerate(data):
        best=i%2;row['ranking'].update(numerical_ties=[best],unique_best_action=best,actions=[dict(action=j,risk=(.1 if j==best else .4)*1e-8,uncertainty=1e-14)for j in range(8)]);row['posterior_screen']=dict(prior_variance=1e-8,between_layout_fraction=.3,between_layout_fraction_indicator=.01,maximum_layout_probability=.6,layout_TV=.001)
    return data

def test_adequacy_positive_and_universal_or_microscopic_negative():
    data=adequate_records();assert h.adequacy(data)['adequacy_qualified']
    for row in data:row['posterior_screen']['prior_variance']=1e-14
    assert not h.adequacy(data)['adequacy_qualified']
    data=adequate_records();data[0].pop('ranking');out=h.adequacy(data);assert out['comparison_blocked']and not out['adequacy_qualified']and out['planned_cases']==36
