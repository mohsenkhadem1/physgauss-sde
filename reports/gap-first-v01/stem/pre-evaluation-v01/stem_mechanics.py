"""Translating-base Euler--Bernoulli reference; SI units throughout.

b = L**3/(3 EI), s = 1/k_base. The base translates but cannot rotate.
These two compliances, rather than arbitrary neural parameters, make the
observable and unobservable directions explicit.
"""
from __future__ import annotations
import numpy as np


def response_row(x, load, length, force):
    x, a = np.broadcast_arrays(np.asarray(x, float), np.asarray(load, float))
    lo, hi = np.minimum(x, a), np.maximum(x, a)
    bending = lo**2 * (3 * hi - lo) / (2 * length**3)
    return force * np.stack((bending, np.ones_like(bending)), axis=-1)


def displacement(x, load, length, force, compliance):
    return response_row(x, load, length, force) @ np.asarray(compliance)


def beam_fem(length, ei, base_stiffness, force, load_fraction, elements=48):
    """Independent assembled cubic Hermite beam, spring at translating base.

    Rotation at base is fixed; force must coincide with a mesh node. Supports
    have finite stiffness, not a penalty intended to imitate a fixed node.
    """
    n = elements + 1
    h = length / elements
    k = ei / h**3 * np.array([[12, 6*h, -12, 6*h],
                              [6*h, 4*h*h, -6*h, 2*h*h],
                              [-12, -6*h, 12, -6*h],
                              [6*h, 2*h*h, -6*h, 4*h*h]])
    K = np.zeros((2*n, 2*n))
    for i in range(elements):
        ix = np.arange(2*i, 2*i+4)
        K[np.ix_(ix, ix)] += k
    K[0, 0] += base_stiffness
    node = int(round(load_fraction * elements))
    if not np.isclose(node / elements, load_fraction):
        raise ValueError('Load must coincide with a node for reference check')
    f = np.zeros(2*n); f[2*node] = force
    free = np.delete(np.arange(2*n), 1)
    u = np.zeros(2*n)
    u[free] = np.linalg.solve(K[np.ix_(free, free)], f[free])
    return np.linspace(0, length, n), u[::2], u[1::2]


def qualify():
    L, F, width, thickness = .12, .02, .01, .008
    I = width * thickness**3 / 12
    records = []
    for name, b, s in [('A', .4, .1), ('B', .1, .4)]:
        ei = L**3 / (3*b)
        for frac in [1, .5, .75]:
            for ne in [12, 24, 48, 96]:
                x, u, rotation = beam_fem(L, ei, 1/s, F, frac, ne)
                truth = displacement(x, frac*L, L, F, [b, s])
                records.append(dict(specimen=name, load_fraction=frac,
                    elements=ne, max_error_m=float(abs(u-truth).max()),
                    peak_slope=float(abs(rotation).max()),
                    surface_strain_bound=float(F*frac*L*thickness/(2*ei)),
                    modulus_pa=ei/I))
    assert max(r['max_error_m'] for r in records) < 1e-8
    ratios = [float(displacement(.5*L, .5*L, L, F, p)/
                    displacement(L, L, L, F, p)) for p in [[.4,.1],[.1,.4]]]
    assert np.allclose(ratios, [.3,.825])
    return dict(model='linear Euler-Bernoulli with translating rotation-constrained base',
        length_m=L, width_m=width, thickness_m=thickness, first_force_n=F,
        half_load_ratios=ratios, half_load_ratio_between_specimens=ratios[1]/ratios[0],
        checks=records, scope='Independent beam FE verification, not nonlinear 3D/contact validation')
