import numpy as np
import pytest
from scripts.compositional_neural_mechanics.local_uncertainty import covariance_from_jacobian,predictive_variance,rank_position_probes


def test_unobservable_stiffness_scale_keeps_prior_uncertainty():
    prior=np.eye(2);position_jacobian=np.array([[8.,0.],[3.,0.]])
    result=covariance_from_jacobian(position_jacobian,prior)
    assert result['covariance'][1,1]==1.
    target_force=np.array([[0.,2.]])
    ranks=rank_position_probes(result['covariance'],target_force,{'more_same_positions':position_jacobian,'known_load_positions':np.array([[0.,3.]])})
    assert ranks[0]['action']=='known_load_positions'
    assert ranks[1]['expected_reduction']==0.
    assert np.isclose(ranks[0]['expected_mean_variance_after'],.4)


def test_discrepancy_is_not_removed_by_latent_information():
    result=covariance_from_jacobian(np.eye(2)*1e6,np.eye(2))
    v=predictive_variance(np.eye(2),result['covariance'],.002**2)
    assert np.all(v>=.002**2)
    with pytest.raises(np.linalg.LinAlgError):covariance_from_jacobian(np.eye(2),np.zeros((2,2)))
    with pytest.raises(ValueError,match='Symmetric'):predictive_variance(np.eye(2),np.array([[1,1],[0,1]]))
    with pytest.raises(ValueError,match='semidefinite'):predictive_variance(np.eye(2),np.diag([1.,-1.]))
    with pytest.raises(ValueError,match='scalar'):predictive_variance(np.eye(2),np.eye(2),np.ones((2,1)))
