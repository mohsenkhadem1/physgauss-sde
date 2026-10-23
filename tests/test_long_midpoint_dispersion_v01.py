"""Prospective CPU-only production midpoint control; no FEM/CUDA imports/calls.

Use the SAME nonlinear-midpoint entry point with an exactly soluble oscillator.
This exercises its target, effective timestep and q/v endpoint reconstruction.
"""
import numpy as np
import pytest
import torch

from experiments.campaign_p2_smooth_midpoint import midpoint_step, smooth_bump
from experiments.campaign_p2_transient import P2State
from experiments.diagnose_long_midpoint_dispersion_v01 import midpoint_clocks, duhamel_clocks


class ExactOscillator:
    def __init__(self, omega):
        self.omega = omega
        self.rest = torch.zeros((1, 3), dtype=torch.float64)
        self.mask = torch.zeros_like(self.rest, dtype=torch.bool)
        self.minimum_J = .1

    def mass(self, x):
        return x

    def evaluate(self, x):
        return dict(energy=.5*self.omega**2*(x*x).sum(), gradient=self.omega**2*x, minimum_J=1.)

    def solve(self, *, force, initial, target, effective_dt, force_tolerance):
        factor = 1/effective_dt**2
        u = (force + factor*target)/(self.omega**2 + factor)
        cache = self.evaluate(u)
        residual = cache['gradient'] + factor*(u-target) - force
        return dict(displacement=u, cache=cache, residual=residual, iterations=1, history=[])


@pytest.mark.parametrize('omega', [2.7, 42.672902889331645])
def test_production_midpoint_matches_independent_cayley_and_duhamel(omega):
    reference = duhamel_clocks(omega)
    errors = []
    for n in (32, 64, 128, 256, 512):
        model = ExactOscillator(omega)
        state = P2State(model.rest.clone(), model.rest.clone(), 0.)
        saved = [[0., 0.]]
        for k in range(n):
            force = torch.zeros_like(model.rest)
            force[0, 2] = smooth_bump((k+.5)/n)
            state, ledger = midpoint_step(model, state, 1/n, force)
            assert abs(ledger['ledger_identity_defect_J']) < 1e-13
            if (k+1) % (n//8) == 0:
                saved.append([float(state.u[0, 2]), float(state.v[0, 2])])
        values = np.array(saved)
        np.testing.assert_allclose(values, midpoint_clocks(omega, n), rtol=1e-10, atol=1e-13)
        errors.append(abs(values-reference).max(0))
    # Only the resolved regime promises the familiar factor4. The structural
    # frequency control intentionally includes the original underresolved steps.
    assert np.all(np.array(errors[-2])/errors[-1] > 3.8)
