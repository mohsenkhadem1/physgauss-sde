"""Finite physical belief and query selection, isolated from actual query outcomes.

The mechanics/sensor adapters supply predictions under hypotheses. Selection
receives no truth parameters or unselected measurements. A design response bank
is distinct from the independently sealed evaluation responses.
"""
from dataclasses import dataclass
import numpy as np
from experiments.gaussian_active_design import condition, normalized_weights, predictive_risk, whitened
from scripts.physical_refinement_v15.risk import score_query


@dataclass(frozen=True)
class Query:
    name: str
    modality: str
    means: np.ndarray
    variance: np.ndarray
    visual_indices: tuple | None = None

    def checked(self, hypotheses):
        means = np.asarray(self.means, dtype=float)
        var = np.asarray(self.variance, dtype=float)
        if means.ndim != 2 or means.shape[0] != hypotheses or means.shape[1] == 0:
            raise ValueError('Query means must be hypotheses by sensor features')
        if not np.isfinite(means).all() or var.shape not in ((means.shape[1],), (means.shape[1], means.shape[1])):
            raise ValueError('Finite sensor features and explicit covariance required')
        # Accept correlated image-feature noise; counting many correlated pixels
        # as independent evidence would manufacture unjustified confidence.
        whitened(means, var)
        return means, var


@dataclass(frozen=True)
class DesignResponses:
    values: np.ndarray
    scales: np.ndarray
    role: str = 'development_design'

    def normalized(self, hypotheses):
        if self.role != 'development_design':
            raise ValueError('Evaluation responses cannot be acquisition-design inputs')
        values = np.asarray(self.values, dtype=float)
        scales = np.asarray(self.scales, dtype=float)
        if values.ndim != 2 or values.shape[0] != hypotheses or values.shape[1] == 0 or scales.shape != (values.shape[1],):
            raise ValueError('Explicit response-feature dimensions and unit scales required')
        if not np.isfinite(values).all() or not np.isfinite(scales).all() or np.any(scales <= 0):
            raise ValueError('Finite design response and positive prespecified unit scales required')
        return values / scales


def choose(weights, queries, design, policy, *, used=(), fixed_sequence=(), seed=0, fantasies=128):
    weights = normalized_weights(weights)
    names = [query.name for query in queries]
    if len(set(names)) != len(names):
        raise ValueError('Query names must be unique')
    available = [query for query in queries if query.name not in used]
    if not available:
        raise ValueError('No unobserved query remains')
    for query in available:
        query.checked(len(weights))
    if policy == 'fixed':
        name = next((name for name in fixed_sequence if name not in used), None)
        if name not in [q.name for q in available]:
            raise ValueError('Fixed sequence must cover the requested acquisition budget')
        return name, dict(policy=policy)
    if policy == 'random':
        return available[int(np.random.default_rng(seed).integers(len(available)))].name, dict(policy=policy)
    scores = {}
    diagnostics = {}
    if policy == 'visual_uncertainty':
        for query in available:
            means, var = query.checked(len(weights))
            indices = query.visual_indices
            if indices is None:
                if query.modality not in ('rgbd', 'rgbd_probe'):
                    continue
                indices = tuple(range(means.shape[1]))
            else:
                indices = tuple(indices)
            if not indices:
                continue
            if len(set(indices)) != len(indices) or any(not isinstance(i, (int, np.integer)) or i < 0 or i >= means.shape[1] for i in indices):
                raise ValueError('Invalid visual feature indices')
            visual_cov = var[list(indices)] if var.ndim == 1 else var[np.ix_(indices, indices)]
            visual_means, _ = whitened(means[:, indices], visual_cov)
            scores[query.name] = predictive_risk(visual_means, weights)
    elif policy == 'physical_risk':
        targets = design.normalized(len(weights))
        for query in available:
            means, var = query.checked(len(weights))
            result = score_query(targets, means, var, weights,
                                 fantasies=fantasies, seed=seed)
            scores[query.name] = result['expected_risk_reduction']
            diagnostics[query.name] = dict(standard_error=result['score_standard_error'],
                                           expected_effective_sample_size=result['expected_effective_sample_size'],
                                           fantasies=result['fantasies'], seed=result['seed'])
    else:
        raise ValueError('Unknown acquisition policy')
    if not scores:
        raise ValueError('No eligible query for this policy')
    # Stable declaration-order tie breaking, never case/truth-specific.
    selected = max(scores, key=scores.get)
    return selected, dict(policy=policy, estimated_scores=scores, monte_carlo_diagnostics=diagnostics,
                          interpretation='Model-based design score, not measured improvement or calibrated uncertainty')


def observe_once(weights, query, observation, *, previously_observed=()):
    if query.name in previously_observed:
        raise ValueError('Do not count an acquired observation twice')
    means, var = query.checked(len(weights))
    return condition(means, np.asarray(observation, dtype=float), var, weights)
