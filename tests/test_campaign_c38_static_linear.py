import numpy as np
import pytest
import torch
from scipy.sparse.linalg import spsolve
from experiments.campaign_c38_supported_grading import SupportedBoundaryGradedGeometry
from experiments.campaign_c38_static_family import compile_known_forces
from experiments.campaign_c38_static_linear import fit_reference_basis
from experiments.campaign_newton_stable_material import StableP2Block


def sparse_reference(m):
    cpu = StableP2Block((4, 2, 2), young=m.young, poisson=m.poisson,
                        density=m.density, gravity=0., traction=0., order=3)
    for key in ('rest', 'tets', 'elements', 'faces'):
        np.testing.assert_array_equal(getattr(cpu, key), getattr(m, key))
    for key in ('bary', 'weights', 'N', 'load', 'mask', 'free'):
        setattr(cpu, key, getattr(m, key).copy())
    cpu.check_bary = np.concatenate([m.check_bary, m.bary])
    return cpu


@pytest.mark.parametrize('support', ['both_ends', 'separated_bottom_patches'])
def test_independent_sparse_rest_tangent_force_superposition_and_E_scaling(support):
    torch.set_num_threads(2)
    m = SupportedBoundaryGradedGeometry((4, 2, 2), 100000., support, 0.)
    old_load = m.load.copy()
    basis = fit_reference_basis(m, support)
    H = sparse_reference(m).elastic(m.rest, tangent=True)[2]
    f = compile_known_forces(m.rest, m.tets, m.faces, m.elements, -1.3)['total_N']
    expected = np.zeros_like(m.rest)
    expected.ravel()[m.free] = spsolve(H[m.free][:, m.free], f.ravel()[m.free])
    np.testing.assert_allclose(basis.displacement(100000., -1.3), expected, atol=2e-12, rtol=1e-9)
    np.testing.assert_array_equal(basis.displacement(100000., -1.3)[m.mask], 0.)
    np.testing.assert_array_equal(m.load, old_load)
    np.testing.assert_allclose(basis.displacement(150000., -1.3), expected * (2/3), atol=2e-12, rtol=1e-9)
    m_high = SupportedBoundaryGradedGeometry((4, 2, 2), 150000., support, 0.)
    H_high = sparse_reference(m_high).elastic(m_high.rest, tangent=True)[2]
    np.testing.assert_allclose(H_high.toarray(), 1.5 * H.toarray(), atol=1e-10, rtol=1e-11)
    with pytest.raises(ValueError):
        basis.displacement(-1., 0.)
    with pytest.raises(ValueError):
        basis.displacement(100000., 3.)
    with pytest.raises(ValueError):
        basis.gravity_response_m[0, 0] = 1.


def test_small_load_nonlinear_limit_is_a_separate_control():
    torch.set_num_threads(2)
    support = 'both_ends'
    m = SupportedBoundaryGradedGeometry((4, 2, 2), 100000., support, 0.)
    basis = fit_reference_basis(m, support)
    scale = 1e-4
    cpu = sparse_reference(m)
    cpu.load *= scale
    # Independent CPU sparse nonlinear Newton solver; the control intentionally
    # scales the applied body load, not stiffness or damping.
    answer = cpu.solve(maximum_iterations=15, force_tolerance_N=1e-11)
    nonlinear = answer['positions'] - m.rest
    np.testing.assert_allclose(nonlinear, scale * basis.gravity_response_m, atol=2e-11, rtol=2e-4)
