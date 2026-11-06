"""Exact finite-hypothesis Gaussian load inference and scalar acquisition.

For hypothesis j, y = H_j F + b_j + e and F ~ N(0,K).  Forces are integrated
analytically; hypothesis weights include the changing covariance determinant.
Exactness is conditional on this finite dictionary and affine observation model.
No true parameters, future targets or unacquired measurements enter planning.
"""
from dataclasses import dataclass
import numpy as np


def _array(x):
    x = np.asarray(x, dtype=np.float64)
    if not np.isfinite(x).all():
        raise ValueError('All inputs must be finite')
    return x


def _logsumexp(x, axis=-1):
    m = np.max(x, axis=axis, keepdims=True)
    return np.squeeze(m + np.log(np.exp(x-m).sum(axis=axis, keepdims=True)), axis=axis)


def _weights(w, n):
    w = np.ones(n)/n if w is None else _array(w)
    if w.shape != (n,) or np.any(w < 0) or w.sum() <= 0:
        raise ValueError('Prior weights must be nonnegative with positive sum')
    return w/w.sum()


@dataclass
class TargetMoments:
    component_means: np.ndarray
    component_variances: np.ndarray
    mean: np.ndarray
    variance: np.ndarray
    risk: float


@dataclass
class Posterior:
    weights: np.ndarray
    means: np.ndarray
    covariances: np.ndarray
    log_evidence: float = 0.0

    def target(self, T, offset=None):
        """Marginal moments for q=T_j F+offset_j; risk averages coordinates."""
        T = _array(T)
        if T.ndim == 2:
            T = np.broadcast_to(T, (len(self.weights),)+T.shape)
        if T.ndim != 3 or T.shape[0] != len(self.weights) or T.shape[2] != self.means.shape[1]:
            raise ValueError('T must be [hypothesis,target,force] or [target,force]')
        mu = np.einsum('hdf,hf->hd', T, self.means)
        if offset is not None:
            mu = mu + np.broadcast_to(_array(offset), mu.shape)
        var = np.einsum('hdf,hfg,hdg->hd', T, self.covariances, T, optimize=True)
        if np.min(var) < -1e-9:
            raise ValueError('Negative conditional target variance')
        var = np.maximum(var, 0)
        mean = self.weights @ mu
        # Centering avoids catastrophic cancellation for large reference offsets.
        variance = self.weights @ (var+(mu-mean)**2)
        return TargetMoments(mu, var, mean, variance, float(variance.mean()))

    def scalar_predictive(self, h, noise_variance, offset=None):
        h = _array(h)
        if h.ndim == 1:
            h = np.broadcast_to(h, self.means.shape)
        if h.shape != self.means.shape:
            raise ValueError('h must be [hypothesis,force] or [force]')
        noise = np.broadcast_to(_array(noise_variance), self.weights.shape)
        if np.any(noise <= 0):
            raise ValueError('Observation noise variance must be positive')
        cross = np.einsum('hfg,hg->hf', self.covariances, h)
        mean = np.einsum('hf,hf->h', h, self.means)
        if offset is not None:
            mean = mean+np.broadcast_to(_array(offset), mean.shape)
        variance = np.einsum('hf,hf->h', h, cross)+noise
        return mean, variance, cross

    def update_scalar(self, h, value, noise_variance, offset=None):
        mean, variance, cross = self.scalar_predictive(h, noise_variance, offset)
        residual = float(value)-mean
        logprior = np.full_like(self.weights, -np.inf)
        np.log(self.weights, out=logprior, where=self.weights > 0)
        logw = logprior-.5*(residual**2/variance+np.log(2*np.pi*variance))
        normalizer = float(_logsumexp(logw))
        means = self.means+cross*(residual/variance)[:, None]
        covariance = self.covariances-cross[:, :, None]*cross[:, None, :]/variance[:, None, None]
        covariance = .5*(covariance+covariance.transpose(0, 2, 1))
        return Posterior(np.exp(logw-normalizer), means, covariance, self.log_evidence+normalizer)

    def sample(self, n, seed=None, rng=None):
        """Return (hypothesis indices, paired load draws), preserving dependence."""
        if n < 1:
            raise ValueError('Positive sample count required')
        rng = np.random.default_rng(seed) if rng is None else rng
        indices = rng.choice(len(self.weights), size=n, p=self.weights)
        force = np.empty((n, self.means.shape[1]))
        for j in np.unique(indices):
            take = indices == j
            L = np.linalg.cholesky(self.covariances[j])
            force[take] = self.means[j]+rng.standard_normal((take.sum(), force.shape[1]))@L.T
        return indices, force


def fit_posterior(H, y, load_cov, noise_cov, prior_weights=None, offset=None):
    """Integrate F exactly, allowing H and known mean offsets to vary by hypothesis.

    noise_cov: positive scalar, diagonal [observation], shared [obs,obs], or
    hypothesis-specific [hypothesis,obs,obs]. load_cov: [force,force] or
    [hypothesis,force,force]. All covariances must be positive definite.
    """
    H, y, K = _array(H), _array(y), _array(load_cov)
    if H.ndim != 3 or y.shape != (H.shape[1],):
        raise ValueError('Expected H[hypothesis,observation,force], y[observation]')
    n, d, f = H.shape
    K = np.broadcast_to(K, (n, f, f))
    np.linalg.cholesky(K)
    R = _array(noise_cov)
    if R.ndim == 0:
        R = np.eye(d)*R
    elif R.ndim == 1:
        if R.shape != (d,):
            raise ValueError('Diagonal noise must match observation dimension')
        R = np.diag(R)
    R = np.broadcast_to(R, (n, d, d))
    np.linalg.cholesky(R)
    prior = _weights(prior_weights, n)
    KH = K @ H.transpose(0, 2, 1)
    C = H @ KH+R
    L = np.linalg.cholesky(C)
    residual = np.broadcast_to(y, (n,d)).copy()
    if offset is not None:
        residual -= np.broadcast_to(_array(offset), residual.shape)
    white = np.linalg.solve(L, residual[..., None])[..., 0]
    alpha = np.linalg.solve(L.transpose(0, 2, 1), white[..., None])[..., 0]
    means = np.einsum('hfd,hd->hf', KH, alpha)
    V = np.linalg.solve(L, KH.transpose(0, 2, 1))
    gain = np.linalg.solve(L.transpose(0, 2, 1), V).transpose(0, 2, 1)
    # Joseph form preserves tiny positive uncertainty at high signal/noise.
    # Direct K-V.T@V cancellation can otherwise create negative eigenvalues.
    remaining = np.eye(f)[None]-gain@H
    cov = remaining@K@remaining.transpose(0, 2, 1)+gain@R@gain.transpose(0, 2, 1)
    cov = .5*(cov+cov.transpose(0, 2, 1))
    logprior = np.full(n, -np.inf)
    np.log(prior, out=logprior, where=prior > 0)
    logw = logprior-.5*(np.sum(white**2, axis=1)+2*np.log(np.diagonal(L, axis1=1, axis2=2)).sum(1)+d*np.log(2*np.pi))
    evidence = float(_logsumexp(logw))
    return Posterior(np.exp(logw-evidence), means, cov, evidence)


def risks_after_values(posterior, T, h, noise_variance, values, offset=None,
                       target_offset=None, chunk_size=64, target_moments=None):
    """Conditional target risk for hypothetical scalar values, without state mutation.

    Uses matrix products, avoiding [outcome,hypothesis,target] allocation.
    """
    # Caller may reuse moments for the SAME posterior, target map and offset
    # across all candidates at one acquisition step. Recompute after updates.
    target = posterior.target(T, target_offset) if target_moments is None else target_moments
    T = _array(T)
    if T.ndim == 2:
        T = np.broadcast_to(T, (len(posterior.weights),)+T.shape)
    mu, variance, cross = posterior.scalar_predictive(h, noise_variance, offset)
    target_cross = np.einsum('hdf,hf->hd', T, cross)
    # Shift target coordinates globally for stable squared moments.
    qmu = target.component_means-target.mean
    gain = target_cross/variance[:, None]
    cond_var = target.component_variances-target_cross**2/variance[:, None]
    cond_var = np.maximum(cond_var, 0)
    c0 = np.mean(cond_var+qmu**2, axis=1)
    c1 = 2*np.mean(qmu*gain, axis=1)
    c2 = np.mean(gain**2, axis=1)
    logprior = np.full_like(posterior.weights, -np.inf)
    np.log(posterior.weights, out=logprior, where=posterior.weights>0)
    values = _array(values).reshape(-1)
    result = []
    for start in range(0, len(values), chunk_size):
        residual = values[start:start+chunk_size, None]-mu
        lw = logprior[None, :]-.5*(residual**2/variance+np.log(2*np.pi*variance))
        weights = np.exp(lw-_logsumexp(lw, axis=1)[:, None])
        qmean = weights@qmu+(weights*residual)@gain
        second = np.sum(weights*(c0+residual*c1+residual**2*c2), axis=1)
        result.append(np.maximum(second-np.mean(qmean**2, axis=1), 0))
    return np.concatenate(result)


def expected_risk_mc(posterior, T, h, noise_variance, offset=None, *,
                     n_samples=128, seed=0, target_offset=None, return_details=False,
                     target_moments=None):
    """Reproducible posterior-predictive MC; seed can be shared across actions.

    Planning only samples the predictive measurement distribution. Report MC SE;
    choose actions at the declared sample budget, and audit with a larger budget.
    """
    if n_samples < 2:
        raise ValueError('At least two predictive samples required')
    mu, variance, _ = posterior.scalar_predictive(h, noise_variance, offset)
    rng = np.random.default_rng(seed)
    index = rng.choice(len(mu), n_samples, p=posterior.weights)
    values = mu[index]+np.sqrt(variance[index])*rng.standard_normal(n_samples)
    risks = risks_after_values(posterior,T,h,noise_variance,values,offset,target_offset,
                              target_moments=target_moments)
    result = dict(expected_risk=float(risks.mean()), standard_error=float(risks.std(ddof=1)/np.sqrt(n_samples)), n_samples=n_samples, seed=seed)
    return result if return_details else result['expected_risk']


def expected_risk_gh(posterior,T,h,noise_variance,offset=None, *, order=31,target_offset=None):
    """Deterministic quadrature under every mixture component, for small audits."""
    mu, variance, _ = posterior.scalar_predictive(h,noise_variance,offset)
    nodes, weights = np.polynomial.hermite.hermgauss(order)
    values = (mu[:,None]+np.sqrt(2*variance[:,None])*nodes).ravel()
    mass = (posterior.weights[:,None]*weights[None,:]/np.sqrt(np.pi)).ravel()
    return float(mass@risks_after_values(posterior,T,h,noise_variance,values,offset,target_offset))


def marginal_quantiles(posterior,T,probabilities=(.05,.95),offset=None):
    """Equal-tail marginal Gaussian-mixture quantiles [probability,target].

    These are marginal coordinate intervals, not simultaneous field coverage.
    """
    from scipy.special import ndtr
    target = posterior.target(T,offset)
    p = _array(probabilities)
    if p.ndim != 1 or np.any((p<=0)|(p>=1)):
        raise ValueError('Quantiles must lie strictly between zero and one')
    mu, sd = target.component_means, np.sqrt(target.component_variances)
    active = posterior.weights>0
    mu,sd,w = mu[active],sd[active],posterior.weights[active]
    lower = np.min(mu-12*sd,axis=0)
    upper = np.max(mu+12*sd,axis=0)
    out = []
    for probability in p:
        lo,hi = lower.copy(),upper.copy()
        for _ in range(70):
            mid = (lo+hi)/2
            z = np.divide(mid-mu,sd,out=np.zeros_like(mu),where=sd>0)
            cdf = np.where(sd>0,ndtr(z),(mid>=mu).astype(float))
            value = w@cdf
            lo = np.where(value<probability,mid,lo)
            hi = np.where(value>=probability,mid,hi)
        out.append((lo+hi)/2)
    return np.stack(out)
