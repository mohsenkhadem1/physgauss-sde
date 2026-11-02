import numpy as np
import pytest
from scripts.compositional_neural_mechanics.model_discrepancy import summarize_errors,held_commands
from scripts.compositional_neural_mechanics.mechanics import GeometrySpec
from scripts.compositional_neural_mechanics.unknown_action_campaign import held_commands as frozen_commands


def test_equal_case_mse_signed_bias_and_variance_decomposition():
    e=np.broadcast_to(np.array([[-1.,2.],[3.,-4.],[5.,6.],[-7.,8.]])[:,:,None,None],(4,2,26,3)).copy()*1e-4
    r=summarize_errors(e)
    np.testing.assert_allclose(r['discrepancy_variance_m2'],np.mean(np.arange(1,9)**2)*1e-8)
    np.testing.assert_allclose(r['centered_coordinate_variance_m2']+r['signed_mean_m']**2,r['discrepancy_variance_m2'])
    with pytest.raises(ValueError):summarize_errors(e[:3])
    e[0,0,0,0]=np.nan
    with pytest.raises(ValueError):summarize_errors(e)


def test_public_command_shapes_match_frozen_study():
    for g in (GeometrySpec(),GeometrySpec((.18,.07,.09),taper=.2,arch=.01)):
        np.testing.assert_array_equal(held_commands(g),frozen_commands(g))
