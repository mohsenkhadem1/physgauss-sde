"""Condition the frozen point/material PRIOR ray library on one new observation.

No cohort files, response generators, GPU models or physical labels are opened.
The C26 NPZs also contain an old experiment's likelihood and posterior weights;
those members are deliberately NEVER loaded. Only prior samples and their
observation-independent predicted 72-ray visibility/ranges are retained.

``load_reference_libraries`` verifies the method gate, three NPZ hashes and the
public exact support prior. ``condition_reference`` returns every seed/prefix
weight vector and summary, stability diagnostics, and the fixed first-seed
2048-particle empirical posterior for subsequent forward queries. Array-valued
results are LOCAL artifacts, not JSON publication payloads.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from types import MappingProxyType
from typing import Mapping

import numpy as np

try:
    from campaign_point_material_reference import mask_probabilities, prefix_indices, summarize
    from censored_range_observation import CensoredRangeModel, CensoredRanges
except ImportError:
    from experiments.campaign_point_material_reference import mask_probabilities, prefix_indices, summarize
    from experiments.censored_range_observation import CensoredRangeModel, CensoredRanges


ROOT = Path(__file__).resolve().parents[1]
METHOD_PATH = 'runs/operator-dev/point-prior-analysis-v01/method-protocol.json'
METHOD_SHA256 = 'ecbc02153336b36ddef33f0c6c01370b012f3ddbeb1d76bd5c71e15145dc3bff'
SUPPORT_PRIOR = dict(
    path='runs/operator-dev/unknown-point-oracle-inputs-v01/public/point-mask-prior.json',
    sha256='7c919bdf7718592c3dacef2402ceb94218a8897348d3a2dc90a183b967526bd0')


def _readonly(value):
    array = np.array(value, copy=True)
    array.flags.writeable = False
    return array


def _checked(root, record):
    path = (root / record['path']).resolve()
    digest = record['sha256']
    if not path.is_relative_to(root) or not isinstance(digest, str) or len(digest) != 64:
        raise ValueError('Repository-local path and SHA256 required')
    if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
        raise ValueError('Pinned artifact hash changed: ' + record['path'])
    return path


@dataclass(frozen=True)
class ReferenceLibrary:
    seed: int
    geometry_ids: np.ndarray                 # [55*1024], global mechanical node IDs
    physical_coordinates: np.ndarray         # [55*1024,4]: E, rho, nu, body drag
    predicted_visible: np.ndarray            # [55*1024,72]
    predicted_residual_m: np.ndarray         # [55*1024,72], absent entries are zero


@dataclass(frozen=True)
class ReferenceLibraries:
    references: tuple[ReferenceLibrary, ...]
    probabilities: Mapping[int, float]
    counts_per_mask: tuple[int, ...]
    sensor: Mapping[str, float]
    primary_seed: int
    predictive_particles: int
    resampling_seed: int
    method_sha256: str
    reference_records: tuple[Mapping[str, str], ...]
    support_prior_record: Mapping[str, str]


def load_reference_libraries(method_path=METHOD_PATH, method_sha256=METHOD_SHA256,
                             *, root=ROOT, support_prior_record=None):
    """Read only gate-selected prior libraries and the exact public mask prior.

    ``root`` and ``support_prior_record`` permit relocation/synthetic fixtures;
    their contents must still pass explicit hashes. The method's fixed ordered
    three seeds, both nested counts and first-seed resampling are never selected
    from a likelihood or diagnostic. This function does not follow ANY cohort,
    observation, truth, prior-posterior-summary or old-weight path in the gate.
    """
    root = Path(root).resolve()
    method_file = _checked(root, dict(path=str(method_path), sha256=method_sha256))
    method = json.loads(method_file.read_text())
    inference = method['base_inference']
    plan = inference['reference_plan']
    seeds, counts = plan['seeds'], plan['counts_per_mask']
    records = inference['library_references']
    if (len(seeds) != 3 or any(type(s) is not int for s in seeds) or len(set(seeds)) != 3
            or len(records) != 3 or len(counts) != 2
            or any(type(c) is not int or c < 1 for c in counts)
            or counts != sorted(set(counts))
            or inference['primary_predictive_reference_seed'] != seeds[0]
            or inference['posterior_predictive_particles'] != 2048
            or inference['posterior_resampling_seed'] != 20261115
            or plan['reuse_observation_independent_predicted_rays'] is not True):
        raise ValueError('Three ordered independent seeds, nested counts and fixed resampling required')
    prior_record = SUPPORT_PRIOR if support_prior_record is None else support_prior_record
    prior_file = _checked(root, prior_record)
    probabilities = mask_probabilities(json.loads(prior_file.read_text()))
    sensor = {name: method['working_sensor'][name] for name in (
        'range_std_m', 'visibility_error_probability', 'background_std_m', 'outlier_probability')}
    if not np.isfinite(list(sensor.values())).all() or sensor['range_std_m'] <= 0:
        raise ValueError('Finite frozen working sensor required')
    # Validate the complete model now, even before the first observation.
    _sensor_model(sensor)
    references = []
    for seed, record in zip(seeds, records):
        path = _checked(root, record)
        if path.name != f'reference-{seed}.npz':
            raise ValueError('Reference path must identify its declared seed')
        with np.load(path, allow_pickle=False) as data:
            # Explicit allowlist. In particular, DO NOT access log_likelihood or weights.
            geometry = data['geometry_ids']
            physical = data['physical_coordinates']
            visible = data['predicted_visible']
            residual = data['predicted_residual_m']
        expected_geometry = np.repeat(sorted(probabilities), max(counts))
        n = len(expected_geometry)
        if (geometry.dtype.kind not in 'iu' or geometry.shape != (n,)
                or not np.array_equal(geometry, expected_geometry)
                or physical.dtype != np.dtype('float64') or physical.shape != (n, 4)
                or not np.isfinite(physical).all()
                or (physical[:, [0, 1, 3]] <= 0).any()
                or ((physical[:, 2] <= 0) | (physical[:, 2] >= .5)).any()
                or visible.dtype != np.dtype('bool') or visible.shape != (n, 72)
                or residual.dtype != np.dtype('float64') or residual.shape != (n, 72)
                or not np.isfinite(residual[visible]).all()):
            raise ValueError('Complete ordered prior strata and finite float64 72-ray library required')
        references.append(ReferenceLibrary(seed, _readonly(geometry), _readonly(physical),
            _readonly(visible), _readonly(np.where(visible, residual, 0.))))
    return ReferenceLibraries(tuple(references), MappingProxyType(probabilities), tuple(counts),
        MappingProxyType(sensor), seeds[0], 2048, 20261115, method_sha256,
        tuple(MappingProxyType(dict(r)) for r in records), MappingProxyType(dict(prior_record)))


def _sensor_model(sensor):
    return CensoredRangeModel(np.eye(72) * sensor['range_std_m']**2,
        sensor['visibility_error_probability'], sensor['background_std_m'], sensor['outlier_probability'])


def systematic_resample(weights, count, seed):
    """Frozen C26 systematic convention, including its single uniform offset."""
    w = np.asarray(weights, dtype=np.float64)
    if (w.ndim != 1 or not len(w) or not np.isfinite(w).all() or (w < 0).any()
            or abs(w.sum() - 1.) > 64*np.finfo(float).eps
            or type(count) is not int or count < 1):
        raise ValueError('Normalized weights and a positive integer count required')
    cdf = np.cumsum(w)
    cdf[-1] = 1.
    return np.searchsorted(cdf,
        (np.random.default_rng(seed).random() + np.arange(count))/count, side='right')


def _stability(reports, seeds, counts):
    largest = [r for r in reports if r['count_per_mask'] == max(counts)]
    support = np.array([[s['posterior_probability'] for s in r['support_categories']] for r in largest])
    result = dict(minimum_largest_run_ESS=min(r['ESS'] for r in largest),
        largest_run_log_evidence_range=float(np.ptp([r['log_evidence'] for r in largest])),
        maximum_independent_seed_support_total_variation=float(max(
            .5*np.abs(a-b).sum() for a in support for b in support)),
        particle_refinement_by_seed=[], material=[],
        qualification='Monte Carlo integration diagnostics, not physical calibration or convergence proof')
    pairs = []
    for seed in seeds:
        pair = sorted((r for r in reports if r['seed'] == seed), key=lambda r: r['count_per_mask'])
        a, b = pair[0], pair[-1]
        pairs.append((a, b))
        pa = np.array([s['posterior_probability'] for s in a['support_categories']])
        pb = np.array([s['posterior_probability'] for s in b['support_categories']])
        result['particle_refinement_by_seed'].append(dict(seed=seed,
            support_total_variation=float(.5*np.abs(pa-pb).sum()),
            absolute_log_evidence_shift=float(abs(a['log_evidence']-b['log_evidence']))))
    for j in range(4):
        sd = float(np.mean([r['material'][j]['std'] for r in largest]))
        means = [r['material'][j]['mean'] for r in largest]
        quantiles = np.array([[r['material'][j][q] for q in ('quantile05', 'median', 'quantile95')]
                             for r in largest])
        shifts = [abs(a['material'][j]['mean']-b['material'][j]['mean']) for a, b in pairs]
        result['material'].append(dict(name=largest[0]['material'][j]['name'],
            independent_seed_mean_range_in_posterior_std=float(np.ptp(means)/sd) if sd else None,
            independent_seed_maximum_quantile_range_in_posterior_std=float(np.ptp(quantiles, axis=0).max()/sd) if sd else None,
            maximum_particle_refinement_mean_shift_in_posterior_std=float(max(shifts)/sd) if sd else None))
    return result


def condition_reference(libraries, observation):
    """Recompute new LL and p(mask)/N weights for ALL seeds and both prefixes.

    Returns a dict with ``seeds`` (new LL and per-prefix ``indices``, ``weights``,
    ``summary``), flattened JSON-ready ``summaries`` and ``stability``, and
    ``empirical_posterior`` containing ``reference_indices``, ``geometry_ids``,
    physical ``physical_coordinates`` and equal weights. Prefix weight indices
    are relative to that seed's full library. The empirical posterior always
    uses the declared first seed's largest count, never the best-scoring seed.
    """
    if not isinstance(libraries, ReferenceLibraries) or not isinstance(observation, CensoredRanges):
        raise TypeError('Verified reference libraries and one CensoredRanges observation required')
    if observation.visible.shape != (72,):
        raise ValueError('One frozen 72-ray observation required')
    model = _sensor_model(libraries.sensor)
    seeds, reports = [], []
    primary = None
    for library in libraries.references:
        predictions = CensoredRanges(library.predicted_visible, library.predicted_residual_m)
        ll = model.log_prob(predictions, observation)
        prefixes = []
        for count in libraries.counts_per_mask:
            rows = prefix_indices(library.geometry_ids, count, libraries.probabilities)
            report, weights = summarize(library.geometry_ids[rows], library.physical_coordinates[rows],
                ll[rows], libraries.probabilities)
            report = dict(seed=library.seed, count_per_mask=count, **report)
            reports.append(report)
            prefixes.append(dict(count_per_mask=count, indices=_readonly(rows),
                weights=_readonly(weights), summary=report))
        seeds.append(dict(seed=library.seed, log_likelihood=_readonly(ll), prefixes=prefixes))
        if library.seed == libraries.primary_seed:
            primary = (library, prefixes[-1])
    library, largest = primary
    selected = systematic_resample(largest['weights'], libraries.predictive_particles, libraries.resampling_seed)
    rows = largest['indices'][selected]
    empirical = dict(reference_seed=library.seed, count_per_mask=max(libraries.counts_per_mask),
        resampling_seed=libraries.resampling_seed, reference_indices=_readonly(rows),
        geometry_ids=_readonly(library.geometry_ids[rows]),
        physical_coordinates=_readonly(library.physical_coordinates[rows]),
        weights=_readonly(np.full(libraries.predictive_particles, 1./libraries.predictive_particles)))
    return dict(seeds=seeds, summaries=reports,
        stability=_stability(reports, [r.seed for r in libraries.references], libraries.counts_per_mask),
        empirical_posterior=empirical,
        method_sha256=libraries.method_sha256,
        old_likelihood_or_posterior_reused=False,
        scope='Known point-pin type, fixed public hardware; uncalibrated working-model posterior')
