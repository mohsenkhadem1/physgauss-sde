import numpy as np
from scripts.physical_refinement_v15.quadrature import scale_rule, gradient_rule, geometry_cases, hypotheses


def test_loguniform_scale_integrates_known_prior_moments():
    expected = (1.5-.75)/np.log(2.)
    for n in (25,49,97):
        s,w=scale_rule(n)
        assert np.isclose(w@s, expected, atol=1e-13)
        assert np.isclose(w@np.log(s), .5*(np.log(.75)+np.log(1.5)), atol=1e-13)


def test_nested_geometry_reuses_states_and_weights_do_not_count_grid_density_as_prior():
    coarse = {r['name']: r for r in geometry_cases(5)}
    fine = {r['name']: r for r in geometry_cases(9)}
    assert len(coarse)==20 and len(fine)==36
    assert all(fine[k]==v for k,v in coarse.items())
    rows,weights=hypotheses(5,25)
    assert len(rows)==500 and np.isclose(weights.sum(),1.)
    g,w=gradient_rule(5)
    assert w[0]==w[-1]==w[1]/2
    assert np.isclose(w@g,0.)
