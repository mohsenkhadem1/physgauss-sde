import numpy as np
import pytest
from scripts.position_ood.force_laws import ForceLawFixture


LAWS = [
    dict(kind='anisotropic_harmonic', center=[.12, 0, -.05],
         stiffness_N_per_m=[[20, 3, 0], [3, 35, 2], [0, 2, 8]]),
    dict(kind='two_gaussian', components=[
        dict(center=[.07, -.035, -.06], width_m=.045, amplitude_J=.07),
        dict(center=[.17, .035, -.09], width_m=.055, amplitude_J=-.09)])]


@pytest.mark.parametrize('law', LAWS)
def test_force_law_energy_gradient_and_tangent(law):
    # The field calculation is independent of the elasticity backend.
    model = object.__new__(ForceLawFixture)
    model.force_law = law
    rng = np.random.default_rng(295001)
    x = rng.uniform([0, -.04, -.04], [.24, .04, .04], (19, 3))
    model.volume_weights = rng.uniform(.5, 1.5, len(x))
    model.volume_weights /= model.volume_weights.sum()
    direction = rng.normal(size=x.shape)
    step = 1e-6
    energy, grad, hessian = model.spatial_field(x, tangent=True)
    ep, gp, _ = model.spatial_field(x+step*direction)
    em, gm, _ = model.spatial_field(x-step*direction)
    assert np.isfinite(energy)
    np.testing.assert_allclose((ep-em)/(2*step), np.sum(grad*direction), rtol=1e-7, atol=1e-9)
    np.testing.assert_allclose((gp-gm).ravel()/(2*step), hessian@direction.ravel(), rtol=2e-7, atol=1e-8)
    np.testing.assert_allclose(hessian.toarray(), hessian.toarray().T, atol=1e-13)


def test_harmonic_force_changes_under_rigid_translation():
    model = object.__new__(ForceLawFixture)
    model.force_law = LAWS[0]
    model.volume_weights = np.full(3, 1/3)
    x = np.array([[.04, 0, 0], [.12, 0, 0], [.20, 0, 0]])
    shift = np.array([0, .04, 0])
    _, before, _ = model.spatial_field(x)
    _, after, _ = model.spatial_field(x+shift)
    np.testing.assert_allclose((after-before).sum(0), np.asarray(LAWS[0]['stiffness_N_per_m'])@shift)


@pytest.mark.parametrize('law', LAWS)
def test_force_law_equilibrium_and_reaction_balance(law):
    model = ForceLawFixture(2, E=20000., Kleft=5000., Kright=5000.,
                            layout='crossed', force_law=law, device='cpu')
    model.linear_solver = 'cpu_multigrid'
    state = model.solve(force_tolerance_N=1e-9)
    result = model.sampled(state)
    assert result['residual_N'] < 1e-9
    assert result['reaction_balance_N'] < 1e-7
    assert result['minimum_J'] > .1
    assert np.linalg.norm(result['spatial_force_N']) > .01
