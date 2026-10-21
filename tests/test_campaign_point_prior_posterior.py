"""Tiny analytic prior-library tests; no campaign response/label files opened."""
import hashlib
import json

import numpy as np
import pytest

from experiments.campaign_point_prior_posterior import (
    condition_reference, load_reference_libraries, systematic_resample)
from experiments.censored_range_observation import CensoredRanges


def record(path, root):
    return dict(path=str(path.relative_to(root)), sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def fixture_library(root, *, offsets=False):
    root.mkdir(exist_ok=True)
    distribution = dict(schema='point-pin-normalized-prior-pushforward-v0.1',
        boundary_index=0, conditioned_on_BC=True, probability_renormalized=False,
        positive_area_dropped=False, total_probability_exact='1',
        masks=[dict(node_id=4, probability=.8), dict(node_id=9, probability=.2)])
    prior = root/'prior.json'
    prior.write_text(json.dumps(distribution))
    seeds = [20261112, 20261113, 20261114]
    references = []
    for i, seed in enumerate(seeds):
        ids = np.repeat([4, 9], 3).astype(np.int64)
        physical = np.column_stack((np.arange(6)+1000.+i*100.,
            np.arange(6)+500., np.full(6, .3), np.full(6, .5)))
        visible = np.zeros((6, 72), dtype=bool)
        visible[:, 0] = True
        residual = np.zeros((6, 72), dtype=np.float64)
        if offsets:
            residual[ids == 9, 0] = .0002
        # Invalid object members prove neither old observation-dependent member
        # can be loaded under allow_pickle=False, even indirectly.
        path = root/f'reference-{seed}.npz'
        np.savez(path, geometry_ids=ids, physical_coordinates=physical,
            predicted_visible=visible, predicted_residual_m=residual,
            log_likelihood=np.array([{'OLD': 'DO NOT READ'}], dtype=object),
            weights=np.array([{'OLD': 'DO NOT READ'}], dtype=object))
        references.append(record(path, root))
    method = dict(base_inference=dict(library_references=references,
        reference_plan=dict(seeds=seeds, counts_per_mask=[1, 3],
                            reuse_observation_independent_predicted_rays=True),
        primary_predictive_reference_seed=seeds[0], posterior_predictive_particles=2048,
        posterior_resampling_seed=20261115), working_sensor=dict(range_std_m=.0002,
        visibility_error_probability=.005, background_std_m=.05, outlier_probability=0.))
    path = root/'method.json'
    path.write_text(json.dumps(method))
    return path, record(path, root)['sha256'], record(prior, root)


def load(root, fixture):
    path, digest, prior = fixture
    return load_reference_libraries(path, digest, root=root, support_prior_record=prior)


def observation(value=0.):
    v = np.zeros(72, dtype=bool)
    v[0] = True
    y = np.full(72, np.nan)
    y[0] = value
    return CensoredRanges(v, y)


def test_constant_likelihood_recovers_exact_prior_and_never_loads_old_weights(tmp_path, monkeypatch):
    fixture = fixture_library(tmp_path)
    original = np.lib.npyio.NpzFile.__getitem__
    accessed = []

    def allowlist(self, key):
        accessed.append(key)
        assert key in {'geometry_ids', 'physical_coordinates', 'predicted_visible', 'predicted_residual_m'}
        return original(self, key)

    monkeypatch.setattr(np.lib.npyio.NpzFile, '__getitem__', allowlist)
    libraries = load(tmp_path, fixture)
    result = condition_reference(libraries, observation())
    assert len(accessed) == 12
    assert result['old_likelihood_or_posterior_reused'] is False
    for seed in result['seeds']:
        for prefix in seed['prefixes']:
            count = prefix['count_per_mask']
            np.testing.assert_allclose(prefix['weights'], np.repeat([.8/count, .2/count], count))
            np.testing.assert_allclose([s['posterior_probability'] for s in prefix['summary']['support_categories']], [.8, .2])
        np.testing.assert_allclose(seed['log_likelihood'], seed['log_likelihood'][0])
    assert result['stability']['maximum_independent_seed_support_total_variation'] == 0
    assert len(result['empirical_posterior']['geometry_ids']) == 2048
    with pytest.raises(ValueError):
        libraries.references[0].physical_coordinates[0, 0] = 0.


def test_new_observation_bayes_factor_and_full_likelihood_recomputation(tmp_path):
    libraries = load(tmp_path, fixture_library(tmp_path, offsets=True))
    first = condition_reference(libraries, observation())
    second = condition_reference(libraries, observation(.0002))
    p_first = .8/(.8 + .2*np.exp(-.5))
    p_second = .8*np.exp(-.5)/(.8*np.exp(-.5) + .2)
    for result, expected in ((first, p_first), (second, p_second)):
        for report in result['summaries']:
            assert report['support_categories'][0]['posterior_probability'] == pytest.approx(expected)
    assert not np.array_equal(first['seeds'][0]['log_likelihood'], second['seeds'][0]['log_likelihood'])


def test_all_seeds_and_nested_prefix_indices_retained_fixed_first_seed_resample(tmp_path):
    libraries = load(tmp_path, fixture_library(tmp_path, offsets=True))
    result = condition_reference(libraries, observation())
    assert [s['seed'] for s in result['seeds']] == [20261112, 20261113, 20261114]
    assert [(r['seed'], r['count_per_mask']) for r in result['summaries']] == [
        (seed, count) for seed in [20261112, 20261113, 20261114] for count in [1, 3]]
    for s in result['seeds']:
        np.testing.assert_array_equal(s['prefixes'][0]['indices'], [0, 3])
        np.testing.assert_array_equal(s['prefixes'][1]['indices'], np.arange(6))
    empirical = result['empirical_posterior']
    assert empirical['reference_seed'] == 20261112 and empirical['resampling_seed'] == 20261115
    weights = result['seeds'][0]['prefixes'][1]['weights']
    # Independent specification of the original systematic CDF convention.
    cdf = np.cumsum(weights)
    cdf[-1] = 1.
    offset = np.random.default_rng(20261115).random()
    expected = np.searchsorted(cdf, (np.arange(2048)+offset)/2048, side='right')
    np.testing.assert_array_equal(empirical['reference_indices'], expected)
    np.testing.assert_array_equal(empirical['physical_coordinates'], libraries.references[0].physical_coordinates[expected])
    np.testing.assert_allclose(empirical['weights'], 1./2048)


@pytest.mark.parametrize('target', ['method', 'prior', 'reference-20261113'])
def test_hash_failure_before_reusing_any_corrupt_library(tmp_path, target):
    fixture = fixture_library(tmp_path)
    suffix = '.npz' if target.startswith('reference') else '.json'
    path = tmp_path/(target+suffix)
    path.write_bytes(path.read_bytes()+b' ')
    with pytest.raises(ValueError, match='hash changed'):
        load(tmp_path, fixture)


def test_missing_stratum_fails_without_prior_renormalization(tmp_path):
    path, _, prior = fixture_library(tmp_path)
    method = json.loads(path.read_text())
    reference = tmp_path/method['base_inference']['library_references'][0]['path']
    with np.load(reference, allow_pickle=False) as data:
        allowed = {key: data[key][:3] for key in (
            'geometry_ids', 'physical_coordinates', 'predicted_visible', 'predicted_residual_m')}
    np.savez(reference, **allowed)
    method['base_inference']['library_references'][0] = record(reference, tmp_path)
    path.write_text(json.dumps(method))
    with pytest.raises(ValueError, match='Complete ordered'):
        load(tmp_path, (path, record(path, tmp_path)['sha256'], prior))


def test_one_supplied_72_ray_observation_required_and_resampling_rejects_bad_weights(tmp_path):
    libraries = load(tmp_path, fixture_library(tmp_path))
    with pytest.raises(ValueError, match='72-ray'):
        condition_reference(libraries, CensoredRanges(np.ones(2, bool), np.zeros(2)))
    with pytest.raises(ValueError, match='Normalized'):
        systematic_resample([.2, .3], 10, 1)
