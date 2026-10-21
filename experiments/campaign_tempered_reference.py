"""Prior-preserving tempered SMC for the declared conditional physical chart.

This is an approximate numerical reference, not a convergence certificate.
Independent seeds, increased particle counts, mode coverage and predictive
stability must be checked before comparing it with amortized inference. The
caller supplies one common prior and log likelihood; no labels/files are read.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np


def normalized_log_weights(log_weights):
    x = np.asarray(log_weights, dtype=float)
    if x.ndim != 1 or np.isnan(x).any() or np.isposinf(x).any() or not np.isfinite(x).any():
        raise ValueError('At least one finite log weight and no NaN/+inf required')
    maximum = np.max(x)
    raw = np.exp(x-maximum)
    normalizer = maximum+np.log(raw.sum())
    return raw/raw.sum(), float(normalizer)


def systematic_resample(weights, rng):
    w = np.asarray(weights, dtype=float)
    if w.ndim != 1 or not len(w) or (w < 0).any() or not np.isfinite(w).all() or not np.isclose(w.sum(), 1):
        raise ValueError('Normalized finite probability vector required')
    cdf = np.cumsum(w)
    cdf[-1] = 1.
    return np.searchsorted(cdf, (rng.random()+np.arange(len(w)))/len(w), side='right')


def reflect_coordinates(x, lower, upper, periodic_indices=(9, 10)):
    """Symmetric box reflection, with periodic angles using a wrapped proposal."""
    lower, upper = np.asarray(lower), np.asarray(upper)
    width = upper-lower
    if (width <= 0).any():
        raise ValueError('Positive chart widths required')
    offset = np.asarray(x)-lower
    folded = np.mod(offset, 2*width)
    result = lower+np.minimum(folded, 2*width-folded)
    for i in periodic_indices:
        result[..., i] = lower[i]+np.mod(offset[..., i], width[i])
    return result


def _merge_states(first, second, mask):
    # PriorState-like dataclass contract: every field is batched on axis0.
    return type(first)(**{key: np.where(mask.reshape((len(mask),)+(1,)*(np.asarray(value).ndim-1)),
                            getattr(second, key), value)
                          for key,value in vars(first).items()})


def category_occupancy(state):
    counts = {}
    for key,value in vars(state).items():
        array = np.asarray(value)
        if key == 'coordinates' or array.ndim not in (1,2) or not np.issubdtype(array.dtype,np.integer):
            continue
        if array.ndim == 1:
            labels,frequency = np.unique(array,return_counts=True)
            counts[key] = {str(int(k)):int(v) for k,v in zip(labels,frequency)}
        else:
            rows,frequency = np.unique(array,axis=0,return_counts=True)
            order = np.argsort(-frequency,kind='stable')[:16]
            counts[key] = dict(distinct_rows=len(rows),
                top_joint_modes=[dict(values=rows[i].tolist(),count=int(frequency[i])) for i in order],
                per_column=[{str(int(k)):int(v) for k,v in zip(*np.unique(column,return_counts=True))}
                            for column in array.T])
    return counts


@dataclass(frozen=True)
class TemperedResult:
    state: object
    weights: np.ndarray
    log_likelihood: np.ndarray
    log_evidence_estimate: float
    stages: list
    seed: int

    def diagnostics(self):
        return dict(particles=len(self.weights), seed=self.seed,
            log_evidence_estimate=self.log_evidence_estimate, stages=self.stages,
            qualification='Final weights are uniform after resampling/rejuvenation; their ESS does not establish mixing. Check independent seeds, particle-count stability and modes.',
            target='Exact stipulated prior times supplied likelihood, approximated by tempered SMC; not proof of simulator fidelity or physical calibration')


def tempered_smc(prior, log_likelihood, *, particles=2048, seed=0,
                 ess_fraction=.8, rejuvenation_steps=6, proposal_scale=.08,
                 independence_probability=.25, max_stages=256,
                 periodic_indices=(9, 10)):
    """Sample p(theta|y) proportional to prior(theta)*exp(log_likelihood).

    Every stage resamples and applies invariant MH moves. Local symmetric
    proposals reflect in the intrinsic box (wrap periodic angles) and retain
    categorical variables; joint prior-independence proposals refresh all
    active/inactive/discrete variables, using the exact proposal cancellation.
    No learned-flow density slice or plug-in nuisance estimate is substituted.
    """
    if (isinstance(particles,bool) or int(particles) != particles or particles < 2 or
        not 0 < ess_fraction < 1 or not 0 < independence_probability <= 1 or
        not 0 < proposal_scale <= 1 or
        any(isinstance(x,bool) or int(x) != x or x < 1 for x in [rejuvenation_steps,max_stages])):
        raise ValueError('Invalid SMC configuration')
    rng = np.random.default_rng(seed)
    n = int(particles)
    state = prior.sample(n, rng)
    lp = prior.log_prob(state)
    if lp.shape != (n,) or not np.isfinite(lp).all():
        raise ValueError('Prior must sample inside its own finite intrinsic support')
    def likelihood(s):
        ll = np.asarray(log_likelihood(s),dtype=float)
        if ll.shape != (n,) or np.isnan(ll).any() or np.isposinf(ll).any():
            raise ValueError('Likelihood must return one finite/-inf log value per particle')
        return ll
    ll = likelihood(state)
    if not np.isfinite(ll).any():
        raise ValueError('No likelihood support sampled; reference has failed')
    beta, log_z, stages = 0., 0., []
    ancestry = np.arange(n)
    while beta < 1:
        if len(stages) >= max_stages:
            raise RuntimeError('SMC did not reach the posterior within max_stages')
        def weights_at(delta):
            # delta is strictly positive here, so -inf likelihood is safe.
            w, normalizer = normalized_log_weights(delta*ll)
            return w, normalizer-np.log(n), 1/np.sum(w*w)
        remaining = 1-beta
        weights, increment, ess = weights_at(remaining)
        delta = remaining
        if ess < ess_fraction*n:
            lo, hi = 0., remaining
            for _ in range(50):
                mid = (lo+hi)/2
                if weights_at(mid)[2] < ess_fraction*n:
                    hi = mid
                else:
                    lo = mid
            delta = hi
            weights, increment, ess = weights_at(delta)
        if delta < 1e-12:
            raise RuntimeError('Temperature stalled; likelihood support or particle diversity is insufficient')
        beta = min(1.,beta+delta)
        log_z += increment
        ids = systematic_resample(weights,rng)
        state, lp, ll = state.take(ids),lp[ids],ll[ids]
        ancestry = ancestry[ids]
        accepted = {'local':0,'independence':0}
        proposed = {'local':0,'independence':0}
        for _ in range(rejuvenation_steps):
            independent = rng.random(n) < independence_probability
            coords = reflect_coordinates(state.coordinates+rng.normal(size=state.coordinates.shape)*
                proposal_scale*(prior.upper-prior.lower), prior.lower,prior.upper,periodic_indices)
            local = state.with_coordinates(coords)
            candidate = _merge_states(local,prior.sample(n,rng),independent)
            new_lp = prior.log_prob(candidate)
            new_ll = likelihood(candidate)
            # Independent proposal q=prior: prior ratio cancels exactly.
            log_alpha = beta*(new_ll-ll)+np.where(independent,0.,new_lp-lp)
            accept = np.log(rng.random(n)) < np.minimum(0.,log_alpha)
            state = _merge_states(state,candidate,accept)
            lp, ll = np.where(accept,new_lp,lp),np.where(accept,new_ll,ll)
            for key,mask in [('local',~independent),('independence',independent)]:
                proposed[key] += int(mask.sum())
                accepted[key] += int((mask&accept).sum())
        stages.append(dict(beta=beta,delta_beta=delta,pre_resampling_ess=float(ess),
            log_evidence_increment=float(increment),unique_initial_ancestors=int(len(np.unique(ancestry))),
            category_occupancy=category_occupancy(state),
            accepted=accepted,proposed=proposed,
            acceptance={k:accepted[k]/proposed[k] if proposed[k] else None for k in proposed}))
    return TemperedResult(state,np.full(n,1/n),ll,float(log_z),stages,int(seed))
