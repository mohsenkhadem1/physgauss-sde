import sys
from pathlib import Path
import unittest
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'experiments'))
from beam_boundary import candidates, observation_design, ExactBoundaryPosterior, response
from beam_active import (State, action_pool, action_models, update, predictive,
                         expected_risk, target_risk, select_action)


class ActiveBeamTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.boundaries = candidates()[::5]
        cls.engine = ExactBoundaryPosterior(cls.boundaries, observation_design('sparse'))
        cls.models = action_models(cls.boundaries, action_pool())
        cls.spatial = np.stack([response(b, np.array([.05, .2, .85, 1.])) for b in cls.boundaries])

    def initial(self):
        y = np.random.default_rng(5).normal(size=len(self.engine.design))
        w, m = self.engine.fit(y)
        return y, State(w[0], m[0], self.engine.cov.copy())

    def test_mixed_sequential_updates_equal_independent_dense_conditioning(self):
        y, state = self.initial()
        h, offset, variance = self.models
        selected = [5, 11, 1]
        observations = [.8, -2., -.3]
        for a, value in zip(selected, observations):
            state = update(state, h[a], offset[a], variance[a], value)
        logp = []
        for c in range(len(self.boundaries)):
            design = np.vstack([self.engine.h[c], h[selected, c]])
            obs = np.r_[y, np.array(observations)-offset[selected, c]]
            s = design@self.engine.prior@design.T + .25**2*np.eye(len(obs))
            mean = self.engine.prior@design.T@np.linalg.solve(s, obs)
            cov = self.engine.prior-self.engine.prior@design.T@np.linalg.solve(s, design@self.engine.prior)
            np.testing.assert_allclose(state.mean[c], mean, atol=2e-11)
            np.testing.assert_allclose(state.cov[c], cov, atol=2e-12)
            logp.append(-.5*(obs@np.linalg.solve(s, obs)+np.linalg.slogdet(s)[1]))
        w = np.exp(np.array(logp)-max(logp)); w /= w.sum()
        np.testing.assert_allclose(state.weights, w, atol=2e-11)

    def test_known_boundary_utility_matches_analytic_variance_reduction(self):
        _, state = self.initial()
        state.weights[:] = 0; state.weights[0] = 1
        h, offset, var = self.models
        for a in [1, 5, 11]:
            mu, _, _ = predictive(state, h[a], offset[a], var[a])
            posterior = update(state, h[a], offset[a], var[a], mu[0])
            exact = target_risk(posterior, self.spatial)
            self.assertAlmostEqual(expected_risk(state, h[a], offset[a], var[a], self.spatial), exact, places=13)
            self.assertLessEqual(exact, target_risk(state, self.spatial)+1e-14)

    def test_probe_updates_support_weights_without_changing_conditional_original_forces(self):
        _, state = self.initial()
        h, offset, var = self.models
        posterior = update(state, h[11], offset[11], var[11], -3.)
        np.testing.assert_array_equal(posterior.mean, state.mean)
        np.testing.assert_allclose(posterior.cov, state.cov, rtol=1e-12, atol=1e-17)
        self.assertGreater(np.linalg.norm(posterior.weights-state.weights), 1e-5)
        # Original load location is irrelevant to a calibrated probe.
        models = action_models(candidates()[:2], [action_pool()[11]])
        np.testing.assert_allclose(models[1][0, 0], models[1][0, 1])

    def test_quadrature_matches_independent_numerical_integration(self):
        _, state = self.initial()
        from beam_active import hypothetical_update
        h, offset, var = self.models
        a = 5
        mu, variance, _ = predictive(state, h[a], offset[a], var[a])
        grid = np.linspace(np.min(mu-9*np.sqrt(variance)), np.max(mu+9*np.sqrt(variance)), 20001)
        density = np.sum(state.weights*np.exp(-.5*(grid[:, None]-mu)**2/variance)
                         /np.sqrt(2*np.pi*variance), axis=1)
        w, m, cov = hypothetical_update(state, h[a], offset[a], var[a], grid)
        norms = np.mean(self.spatial**2, axis=1)
        second = np.mean(m*m+np.diagonal(cov, axis1=1, axis2=2), axis=2)*norms
        pred = np.einsum('qc,qct,cx->qtx', w, m, self.spatial, optimize=True)
        risk = np.sum(w*second, axis=1)-np.mean(pred*pred, axis=(1, 2))
        integrated = np.trapezoid(density*risk, grid)
        gh = expected_risk(state, h[a], offset[a], var[a], self.spatial, 61)
        self.assertLess(abs(integrated-gh)/integrated, 1e-4)

    def test_policy_restriction_and_uninformative_action(self):
        _, state = self.initial()
        risk = expected_risk(state, np.zeros_like(state.mean), np.zeros(len(state.weights)), .25**2, self.spatial)
        self.assertAlmostEqual(risk, target_risk(state, self.spatial), places=13)
        selected, scores = select_action(state, self.models, self.spatial, [3, 9], 15)
        self.assertIn(selected, [3, 9]); self.assertEqual(set(scores), {3, 9})
        self.assertTrue(np.isfinite(list(scores.values())).all())


if __name__ == '__main__':
    unittest.main()
