import numpy as np
from scripts.physical_refinement_v15.belief import Query,DesignResponses
from scripts.physical_refinement_v15.forecast import rollout
from scripts.physical_refinement_v15.posterior import Likelihood,posterior


def test_only_chosen_actual_outcomes_enter_updates():
    initial=Query('initial','rgbd',np.array([[0.],[1.]]),np.ones(1))
    queries=[Query(name,'rgbd',np.array([[0.],[value]]),np.ones(1)) for name,value in [('a',2.),('b',3.),('poison',9.)]]
    design=DesignResponses(np.array([[0.],[1.]]),np.ones(1))
    seen=[]
    def request(name):
        assert name!='poison'
        seen.append(name);return np.array([.3])
    snapshots,record=rollout(np.array([.5,.5]),initial,queries,design,request,policy='fixed',fixed_sequence=['b','a'])
    expected,_=posterior(np.array([.5,.5]),[Likelihood(q).log_density(np.array([.3])) for q in (initial,queries[1],queries[0])])
    assert seen==['initial','b','a'] and record['sequence']==['b','a']
    np.testing.assert_allclose(snapshots[-1],expected)
