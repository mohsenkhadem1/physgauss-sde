"""Numerical controls for two categories and a continuous log-uniform modulus.

The supplied log likelihood must use acquired public observations only. This
module draws no observations and opens no dataset. Agreement between these
integrators cannot validate a forward model or establish physical coverage.
"""
import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.integrate import quad_vec
from scipy.optimize import brentq


class StaticPosterior:
    def __init__(self, log_likelihood, *, young_bounds=(75000., 150000.),
                 categories=('both_ends', 'separated_bottom_patches'),
                 category_prior=(.5, .5), method='adaptive', order=128,
                 rtol=1e-9, atol=1e-12):
        self.bounds = np.asarray(young_bounds, dtype=float)
        self.categories = tuple(categories)
        self.prior = np.asarray(category_prior, dtype=float)
        if self.bounds.shape != (2,) or not np.isfinite(self.bounds).all() or not 0 < self.bounds[0] < self.bounds[1]:
            raise ValueError('Finite positive increasing modulus bounds required')
        if len(self.categories) != 2 or len(set(self.categories)) != 2 or self.prior.shape != (2,) or not np.isfinite(self.prior).all() or np.any(self.prior <= 0) or not np.isclose(self.prior.sum(), 1., rtol=0, atol=1e-15):
            raise ValueError('Two distinct categories with explicit positive normalized prior required')
        if method not in ('adaptive', 'legendre') or type(order) is not int or order < 2 or not np.isfinite([rtol, atol]).all() or min(rtol, atol) <= 0:
            raise ValueError('Explicit numerical method/order and positive tolerances required')
        self.log_likelihood = log_likelihood
        self.method, self.order, self.rtol, self.atol = method, order, rtol, atol
        self.errors = []
        self._nodes, self._weights = leggauss(order)
        # Shared shift is only arithmetic scaling; no likelihood or prior cut.
        self.shift = max(self._log(c, x) for c in self.categories for x in np.linspace(0, 1, 257))
        self.mass = self.integral(1.)
        self.Z = float(self.mass.sum())
        if not np.isfinite(self.Z) or self.Z <= 0:
            raise FloatingPointError('Unresolved normalizer; preserve failure')
        self.category_probabilities = self.mass / self.Z
        self.log_normalizer = float(np.log(self.Z) + self.shift)

    def young(self, unit):
        return self.bounds[0] * np.exp(float(unit) * np.log(self.bounds[1]/self.bounds[0]))

    def _log(self, category, unit):
        value = float(self.log_likelihood(category, self.young(unit)))
        if not np.isfinite(value):
            raise FloatingPointError('Finite supplied log likelihood required; no dropped nodes')
        return value

    def weight(self, unit):
        logs = np.array([self._log(c, unit) for c in self.categories]) - self.shift
        if np.any(logs > 700):
            raise FloatingPointError('Numerical scaling unresolved; no likelihood clipping')
        return self.prior * np.exp(logs)

    def integral(self, upper):
        if not np.isfinite(upper) or not 0 <= upper <= 1:
            raise ValueError('Unit log-modulus integration endpoint required')
        if upper == 0:
            return np.zeros(2)
        if self.method == 'adaptive':
            value, error, info = quad_vec(self.weight, 0., upper, epsrel=self.rtol,
                                         epsabs=self.atol, norm='max', limit=512,
                                         full_output=True)
            self.errors.append(float(error))
            if not info.success:
                raise FloatingPointError('Adaptive integral did not converge')
        else:
            nodes = upper * (self._nodes + 1) / 2
            value = np.array([self.weight(x) for x in nodes]).T @ (self._weights * upper/2)
        if not np.isfinite(value).all() or np.any(value < 0):
            raise FloatingPointError('Invalid positive posterior integral')
        return value

    def cdf(self, young_Pa):
        if not np.isfinite(young_Pa):
            raise ValueError('Finite material CDF query required')
        if young_Pa <= self.bounds[0]:
            return 0.
        if young_Pa >= self.bounds[1]:
            return 1.
        u = np.log(young_Pa/self.bounds[0])/np.log(self.bounds[1]/self.bounds[0])
        return float(self.integral(float(u)).sum()/self.Z)

    def quantile(self, probability):
        if not np.isfinite(probability) or not 0 < probability < 1:
            raise ValueError('Interior probability required')
        u = brentq(lambda x: self.cdf(self.young(x))-probability, 0., 1., xtol=1e-12)
        return self.young(u)


def compare_integrators(log_likelihood, *, orders=(64, 128), **kwargs):
    """Return discrepancies; physical execution must freeze its own pass gates.

    Quantiles integrate to each query endpoint, rather than using percentiles
    of finite quadrature atoms. No automatic order/budget escalation occurs.
    """
    if len(orders) != 2 or not 2 <= orders[0] < orders[1]:
        raise ValueError('Two prospectively fixed increasing orders required')
    adaptive = StaticPosterior(log_likelihood, method='adaptive', **kwargs)
    queries = np.geomspace(*adaptive.bounds, 17)
    probabilities = (.025, .5, .975)
    reference_cdf = np.array([adaptive.cdf(x) for x in queries])
    reference_q = np.array([adaptive.quantile(p) for p in probabilities])
    rows = []
    for order in orders:
        fixed = StaticPosterior(log_likelihood, method='legendre', order=order, **kwargs)
        rows.append(dict(order=order,
                         log_normalizer_difference=float(abs(fixed.log_normalizer-adaptive.log_normalizer)),
                         maximum_category_probability_difference=float(abs(fixed.category_probabilities-adaptive.category_probabilities).max()),
                         maximum_material_cdf_difference=float(abs(np.array([fixed.cdf(x) for x in queries])-reference_cdf).max()),
                         maximum_log_material_quantile_difference=float(abs(np.log(np.array([fixed.quantile(p) for p in probabilities])/reference_q)).max())))
    return dict(adaptive_category_probabilities=adaptive.category_probabilities.tolist(),
                adaptive_material_quantiles_Pa=reference_q.tolist(), comparisons=rows,
                adaptive_reported_absolute_errors=adaptive.errors,
                physical_calibration_established=False, pass_gate_not_chosen_by_helper=True)
