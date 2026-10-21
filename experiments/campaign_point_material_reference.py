"""Stratified numerical reference for unknown point support and material.

Each mechanically distinct support mask retains its exact public-geometry
prior mass. Materials are drawn from their normalized physical marginal in
every stratum. The integration weights are p(mask)/N_mask, not uniform across
masks or across particles when stratum counts differ. This module never reads
truth, trajectories, observations or a boundary label.
"""
from __future__ import annotations

import numpy as np

try:
    from campaign_material_marginal import MaterialMarginalPrior
except ImportError:
    from experiments.campaign_material_marginal import MaterialMarginalPrior


def mask_probabilities(distribution):
    if (distribution.get('schema') != 'point-pin-normalized-prior-pushforward-v0.1'
            or distribution.get('boundary_index') != 0
            or not distribution.get('conditioned_on_BC')
            or distribution.get('probability_renormalized')
            or distribution.get('positive_area_dropped')):
        raise ValueError('Exact complete point-pin conditional prior required')
    pairs = [(int(m['node_id']), float(m['probability'])) for m in distribution['masks']]
    if len({k for k, _ in pairs}) != len(pairs) or not pairs:
        raise ValueError('Distinct nonempty global node categories required')
    probabilities = dict(sorted(pairs))
    p = np.array(list(probabilities.values()))
    # This checks floating exports of the exact rational sum, never repairs it.
    if (not np.isfinite(p).all() or (p <= 0).any()
            or abs(p.sum() - 1.) > 64*np.finfo(float).eps
            or distribution['total_probability_exact'] != '1'):
        raise ValueError('All positive prior masses and exact unit total required')
    return probabilities


def draw_stratified(prior, distribution, *, count_per_mask, seed):
    if type(prior) is not MaterialMarginalPrior:
        raise TypeError('Normalized material marginal required')
    if type(count_per_mask) is not int or count_per_mask < 1:
        raise ValueError('Positive integer count per mask required')
    probabilities = mask_probabilities(distribution)
    if prior.parent_prior_sha256 != distribution['prior_sha256']:
        raise ValueError('Material and mask marginals must share one full prior')
    rng = np.random.default_rng(seed)
    ids = np.repeat(list(probabilities), count_per_mask).astype(np.int64)
    state = prior.sample(len(ids), rng)
    return ids, prior.physical_coordinates(state)


def prefix_indices(geometry_ids, count_per_mask, probabilities):
    ids = np.asarray(geometry_ids)
    if type(count_per_mask) is not int or count_per_mask < 1:
        raise ValueError('Positive prefix count required')
    rows = []
    for node in sorted(probabilities):
        available = np.flatnonzero(ids == node)
        if len(available) < count_per_mask:
            raise ValueError('Insufficient samples in a declared positive-mass stratum')
        rows.append(available[:count_per_mask])
    if set(np.unique(ids)) != set(probabilities):
        raise ValueError('Unknown or missing category; no dropped-support fallback')
    return np.concatenate(rows)


def summarize(geometry_ids, physical, log_likelihood, probabilities):
    """Normalized joint posterior and evidence, retaining zero-weight categories.

    Finite samples in every positive-mass category are mandatory. A category
    whose likelihood underflows is kept with its log evidence and zero exported
    probability; an all-impossible likelihood raises rather than inventing mass.
    """
    ids = np.asarray(geometry_ids)
    x = np.asarray(physical, dtype=np.float64)
    ll = np.asarray(log_likelihood, dtype=np.float64)
    if (ids.ndim != 1 or ids.dtype.kind not in 'iu' or x.shape != (len(ids), 4)
            or ll.shape != (len(ids),) or not np.isfinite(x).all()
            or np.isnan(ll).any() or np.isposinf(ll).any()
            or set(np.unique(ids)) != set(probabilities)):
        raise ValueError('Complete categorical/material particles and valid log likelihoods required')
    p = np.asarray(list(probabilities.values()), dtype=float)
    if not np.isfinite(p).all() or (p <= 0).any() or abs(p.sum()-1.) > 64*np.finfo(float).eps:
        raise ValueError('Normalized strictly positive categorical masses required')
    log_base = np.empty(len(ids))
    counts = {}
    for node, mass in probabilities.items():
        rows = ids == node
        counts[node] = int(rows.sum())
        log_base[rows] = np.log(mass)-np.log(counts[node])
    maximum = float(np.max(ll+log_base))
    if not np.isfinite(maximum):
        raise ValueError('Every particle is impossible under the likelihood')
    unnormalized = np.exp(ll+log_base-maximum)
    log_evidence = maximum+np.log(unnormalized.sum())
    weights = unnormalized/unnormalized.sum()
    categories = []
    for node, mass in sorted(probabilities.items()):
        rows = ids == node
        local_max = float(np.max(ll[rows]))
        if np.isfinite(local_max):
            local = np.exp(ll[rows]-local_max)
            local /= local.sum()
            local_ess = float(1./np.square(local).sum())
            log_z = float(local_max+np.log(np.exp(ll[rows]-local_max).sum())-np.log(counts[node]))
        else:
            local_ess, log_z = None, None
        categories.append(dict(node_id=int(node), prior_probability=float(mass),
            posterior_probability=float(weights[rows].sum()), particles=counts[node],
            conditional_material_ESS=local_ess, conditional_log_evidence=log_z))
    mean = weights@x
    centered = x-mean
    covariance = centered.T@(weights[:, None]*centered)
    material = []
    for j, name in enumerate(['E_Pa', 'rho_kg_m3', 'nu', 'body_drag_s_inverse']):
        order = np.argsort(x[:, j], kind='stable')
        q = np.interp([.05, .5, .95], np.cumsum(weights[order]), x[order, j])
        material.append(dict(name=name, mean=float(mean[j]), std=float(np.sqrt(covariance[j,j])),
            quantile05=float(q[0]), median=float(q[1]), quantile95=float(q[2])))
    result = dict(particles=len(ids), ESS=float(1./np.square(weights).sum()),
        maximum_weight=float(weights.max()), log_evidence=float(log_evidence),
        material=material, weighted_physical_material_covariance=covariance.tolist(),
        support_categories=categories, categories_dropped=[],
        scope='Known point-pin type; unknown discrete support node and continuous material; not subnode center inference')
    return result, weights
