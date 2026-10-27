import numpy as np
from scripts.phystwin.controlled_probing import normalize, utility


def test_posterior_normalization_extreme_likelihood():
    p = normalize(np.array([-10000., -10001., -10002.]))
    assert np.isclose(p.sum(), 1)
    assert np.allclose(p[1:] / p[:-1], np.exp(-1))


def test_uninformative_probe_preserves_target_variance():
    p = np.array([.2, .3, .5])
    targets = np.array([[0., 1.], [2., 4.], [-1., 2.]]) * .001
    probes = np.zeros((2, 3, 3))
    variance = np.mean(p @ (targets * targets) - (p @ targets) ** 2)
    assert np.allclose(utility(p, probes, targets, 7), variance, rtol=1e-10, atol=1e-15)


def test_resolving_probe_reduces_expected_variance():
    p = np.array([.5, .5])
    targets = np.array([[0., 0.], [.01, .01]])
    probes = np.array([[[0., 0., 0.], [.01, 0., 0.]]])
    assert abs(utility(p, probes, targets, 7)[0]) < 1e-12
