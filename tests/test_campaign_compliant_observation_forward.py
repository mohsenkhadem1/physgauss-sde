import numpy as np
import torch
from experiments import campaign_compliant_continuous_reference as base
from experiments import campaign_compliant_observation_forward as observation

torch.set_num_threads(2)


def test_observation_only_matches_frozen_full_forward_random_prior_and_posterior():
    rng=np.random.default_rng(2026121104)
    unit=np.vstack([rng.uniform(size=(32,4)),np.zeros((1,4)),np.ones((1,4)),
        np.column_stack([rng.uniform(.39,.62,32),rng.uniform(size=(32,3))])])
    theta=base.physical(unit,(0,1,2,3));short=observation.ObservationOnlyForward();full=base.ExactVectorForward()
    acquired=base.original.selected_observations(full(theta))
    np.testing.assert_allclose(short.observations(theta),acquired,atol=2e-15,rtol=2e-11)
    actual=base.original.selected_observations(full(base.original.TRUTH[None]))[0]
    np.testing.assert_allclose(short.log_likelihood(unit,(0,1,2,3),actual),full.log_likelihood(unit,(0,1,2,3),actual),atol=1e-9,rtol=1e-12)


def test_chunk_boundary_and_future_controls_do_not_change_acquired_data():
    rng=np.random.default_rng(2026121105);unit=rng.uniform(size=(1027,2));theta=base.physical(unit,(0,3))
    model=observation.ObservationOnlyForward();a=model.observations(theta)
    model.forces['A'][24:]=1e9 # synthetic future corruption only on fresh object
    b=model.observations(theta)
    np.testing.assert_array_equal(a,b)
    np.testing.assert_allclose(a,base.original.selected_observations(base.ExactVectorForward()(theta)),atol=2e-15,rtol=2e-11)
