import sys
from pathlib import Path
import unittest
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'experiments'))
from beam_boundary import candidates, observation_design, ExactBoundaryPosterior, response
from beam_active import State, action_pool, action_models, update, predictive, hypothetical_update, target_risk
from beam_adaptive import AdaptiveForcePosterior, expected_risk_grid


class AdaptiveBeamTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bounds = candidates()[::8]
        cls.design = observation_design('sparse')

    def test_single_timescale_reproduces_original_model(self):
        original = ExactBoundaryPosterior(self.bounds, self.design)
        adaptive = AdaptiveForcePosterior(self.bounds, self.design, [2.])
        y = np.random.default_rng(4).normal(size=(4, len(self.design)))
        for a, b in zip(original.fit(y), adaptive.fit(y)):
            np.testing.assert_allclose(a, b, atol=1e-12)
        np.testing.assert_allclose(original.cov, adaptive.cov, atol=1e-14)

    def test_joint_evidence_and_sequential_updates_match_dense_gaussians(self):
        engine = AdaptiveForcePosterior(self.bounds, self.design, [.35, .8, 2.], length_weights=[1., 3., 2.])
        y = np.random.default_rng(5).normal(size=len(self.design))
        w, m = engine.fit(y); state = State(w[0], m[0], engine.cov.copy())
        models = engine.expand_models(action_models(self.bounds, action_pool()))
        selected = [5, 12, 3]; observed = [2., -.7, -4.]
        for a, value in zip(selected, observed):
            state = update(state, models[0][a], models[1][a], models[2][a], value)
        scores = []
        for c in range(len(engine.prior)):
            h = np.vstack([engine.h[c], models[0][selected, c]])
            obs = np.r_[y, np.array(observed)-models[1][selected, c]]
            k = engine.prior[c]; s = h@k@h.T+.25**2*np.eye(len(obs))
            mean = k@h.T@np.linalg.solve(s, obs)
            cov = k-k@h.T@np.linalg.solve(s, h@k)
            np.testing.assert_allclose(state.mean[c], mean, atol=1e-10)
            np.testing.assert_allclose(state.cov[c], cov, atol=1e-12)
            scores.append(np.log(engine.prior_weights[c])-.5*(obs@np.linalg.solve(s, obs)+np.linalg.slogdet(s)[1]))
        weights = np.exp(np.array(scores)-max(scores)); weights /= weights.sum()
        np.testing.assert_allclose(state.weights, weights, atol=1e-10)

    def test_full_hyperparameter_uncertainty_is_retained(self):
        engine = AdaptiveForcePosterior(self.bounds[:1], self.design, [.35, 2.])
        w, m = engine.fit(np.zeros(len(self.design)))
        state = State(w[0], m[0], engine.cov)
        spatial = np.stack([response(self.bounds[0], np.array([1.]))]*2)
        expected = np.sum(w[0]*np.mean(np.diagonal(engine.cov, axis1=1, axis2=2), axis=1)*spatial[:, 0]**2)
        self.assertAlmostEqual(target_risk(state, spatial), expected, places=14)
        self.assertGreater(expected, np.min(np.mean(np.diagonal(engine.cov, axis1=1, axis2=2), axis=1)*spatial[:, 0]**2))
        np.testing.assert_allclose(engine.length_probabilities(w), w)

    def test_adaptive_grid_against_independent_dense_integration(self):
        engine = AdaptiveForcePosterior(self.bounds, self.design, [.35, .8, 2.])
        y = np.random.default_rng(8).normal(size=len(self.design))
        w, m = engine.fit(y); state = State(w[0], m[0], engine.cov)
        models = engine.expand_models(action_models(self.bounds, action_pool()))
        spatial = np.stack([response(b, np.array([.05, .2, .85, 1.])) for b in self.bounds])[engine.boundary_index]
        for a in [1, 5, 12]:
            score, diagnostics = expected_risk_grid(state, models[0][a], models[1][a], models[2][a], spatial, 1e-7)
            mu, var, _ = predictive(state, models[0][a], models[1][a], models[2][a])
            grid = np.linspace(np.min(mu-10*np.sqrt(var)), np.max(mu+10*np.sqrt(var)), 20001)
            density = np.exp(-.5*(grid[:, None]-mu)**2/var)/np.sqrt(2*np.pi*var)@state.weights
            ww, mm, cc = hypothetical_update(state, models[0][a], models[1][a], models[2][a], grid)
            second = np.sum(ww*np.mean(mm*mm+np.diagonal(cc, axis1=1, axis2=2), axis=2)*np.mean(spatial**2, axis=1), axis=1)
            pred = np.einsum('qc,qct,cx->qtx', ww, mm, spatial, optimize=True)
            integral = np.trapezoid(density*(second-np.mean(pred**2, axis=(1, 2))), grid)
            self.assertLess(abs(score-integral)/target_risk(state, spatial), 1e-7)
            self.assertLess(diagnostics['mass_error'], 1e-7)
            self.assertLessEqual(score, target_risk(state, spatial)+1e-12)

    def test_uninformative_observation_and_noiseless_timescale_prior_input(self):
        engine = AdaptiveForcePosterior(self.bounds, self.design, [.35, .8, 2.])
        w, m = engine.fit(np.zeros(len(self.design))); state = State(w[0], m[0], engine.cov)
        updated = update(state, np.zeros_like(m[0]), np.zeros(len(w[0])), .25**2, .2)
        np.testing.assert_allclose(updated.weights, state.weights, atol=1e-15)
        # There is no true-timescale argument to update or selection.
        with self.assertRaises(ValueError):
            AdaptiveForcePosterior(self.bounds, self.design, [0.])


if __name__ == '__main__': unittest.main()
