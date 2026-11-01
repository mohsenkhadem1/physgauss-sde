import numpy as np
import pytest
from scripts.physical_refinement_v15.predictive_scores import marginal_scores


def test_crps_matches_explicit_pairwise_distribution():
    values=np.array([[-1.,2.],[1.,4.],[3.,7.]])
    weights=np.array([.2,.5,.3]);truth=np.array([0.,3.])
    expected=(weights@abs(values-truth)-.5*np.einsum('i,j,ijd->d',weights,weights,abs(values[:,None]-values[None]))).mean()
    assert np.isclose(marginal_scores(values,weights,truth)['crps_mean'],expected)
    point=marginal_scores(np.array([[2.,3.]]),np.ones(1),np.array([1.,5.]))
    assert point['crps_mean']==1.5 and point['mean_interval_width']==0


def test_probability_roundoff_and_invalid_shapes():
    assert marginal_scores(np.array([[1000.]]),np.array([1.000001]),np.array([1000.]))['crps_mean']==0
    for values,weights,target in [
        (np.ones((2,2)),np.array([.5,.5]),np.ones((1,2))),
        (np.ones((2,0)),np.array([.5,.5]),np.ones(0)),
        (np.ones((2,1)),np.array([[.5,.5]]),np.ones(1)),
        (np.ones((2,1)),np.array([np.nan,.5]),np.ones(1)),
    ]:
        with pytest.raises(ValueError):marginal_scores(values,weights,target)
