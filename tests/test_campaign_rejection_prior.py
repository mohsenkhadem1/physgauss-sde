from dataclasses import replace
import math
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'experiments'))
from isaac_dataset import geometry, compact_patch, face_ids
from campaign_physical_prior import FACE_BASES, FACE_NORMALS
from campaign_rejection_prior import (
    ConditionalRejectionPrior, RejectionPriorState, SamplingBudgetExceeded,
)


@pytest.fixture
def context():
    points, _, _, spec = geometry(dict(cells=[12, 6, 4], seed=531000,
        boundary_type='point_pin', force_type='push_pulse', load_patch_nodes=9, seconds=3.))
    return points, points[spec['load_nodes']], np.array(spec['load_patches'])


@pytest.fixture
def prior(context):
    return ConditionalRejectionPrior(*context)


def test_joint_sampler_retains_uniform_BC_material_marginals_and_finite_nuisances(prior):
    s, d = prior.sample_with_diagnostics(7000, np.random.default_rng(33))
    assert np.max(np.abs(np.bincount(s.boundary_type)/len(s)-.25)) < .02
    assert np.max(np.abs(np.bincount(s.force_type)/len(s)-.25)) < .02
    for values, lo, hi in [(s.coordinates[:, 0], math.log(1500), math.log(20000)),
                           (np.exp(s.coordinates[:, 1]), 400, 850),
                           (s.coordinates[:, 2], .28, .42),
                           (np.exp(s.coordinates[:, 3]), .35, .85)]:
        assert abs(np.mean((values-lo)/(hi-lo))-.5) < .02
    assert np.isfinite(prior.log_kernel(s)).all()
    assert d['completed'] and d['accepted'] == len(s)
    assert d['proposals'] >= len(s) and max(d['attempts_per_particle']) > 1
    assert np.all(prior.acceptance_lower_bound > 0)
    assert d['unconditional_worst_BC_failure_probability_upper_bound'] < 1e-12
    assert not hasattr(prior, 'log_prob')


def test_feasibility_matches_generator_stable_compact_patch_exactly(prior):
    rng = np.random.default_rng(51);s = prior.sample(350, rng)
    x = s.coordinates.copy();x[:, 4:8] = rng.uniform(-.18, .18, (len(s), 4))
    faces = rng.integers(6, size=len(s));s = replace(s.with_coordinates(x), support_face=faces)
    expected = []
    for i in range(len(s)):
        valid = True
        for j in range(2 if s.boundary_type[i] == 2 else 1):
            face = (s.support_face[i]+3*j) % 6
            center = prior.center+FACE_NORMALS[face]*prior.size/2+x[i, 4+2*j:6+2*j]@FACE_BASES[face]*prior.size
            try:
                patch = compact_patch(prior.reference_points,
                    face_ids(prior.reference_points, face % 3, -1 if face < 3 else 1),
                    center, [1, 16, 9, 16][s.boundary_type[i]])
                hardware = prior.reference_points[prior.handle_patch_node_ids].reshape(-1, 3)
                valid &= np.linalg.norm(prior.reference_points[patch, None]-hardware[None], axis=-1).min() > .027
            except ValueError:
                valid = False
        expected.append(valid)
    assert np.array_equal(prior.feasible_support(s), expected)
    assert not all(expected) and any(expected)


def test_actual_valid_point_outside_old_certified_atlas_remains_supported():
    # A knownTRAIN geometry excluded by the old8x8atlas, without reading files.
    points, _, _, spec = geometry(dict(cells=[12, 6, 4], seed=531008,
        boundary_type='point_pin', force_type='shear_oscillation', load_patch_nodes=9, seconds=3.))
    prior = ConditionalRejectionPrior(points, points[spec['load_nodes']], np.array(spec['load_patches']))
    s = prior.sample(1, np.random.default_rng(3));x = s.coordinates.copy()
    axis, sign = spec['support_faces'][0];face = int(axis+3*(sign > 0))
    axes = [a for a in range(3) if a != axis]
    x[0, 4:6] = ((np.array(spec['support_centers_m'][0])-prior.center)/prior.size)[axes]
    s = replace(s.with_coordinates(x), boundary_type=np.array([0]), support_face=np.array([face]))
    assert prior.feasible_support(s)[0]
    assert np.isfinite(prior.log_kernel(s)[0])
    uv = x[0, 4:6];ij = np.floor((uv+.18)/.36*prior._witness.grid).astype(int)
    cell = int(ij[0]*prior._witness.grid+ij[1])
    assert cell not in prior._witness.feasible_cells[0, face]


def test_inactive_second_support_is_not_a_rejection_constraint(prior):
    s = prior.sample(200, np.random.default_rng(34));s = s.take(np.flatnonzero(s.boundary_type != 2))
    x = s.coordinates.copy();x[:, 6:8] = 1000.
    assert prior.feasible_support(s.with_coordinates(x)).all()
    # Physical geometry ignores the inactive support, but its nuisance density
    # is still proper and zero outside the declared finite coordinate box.
    assert np.isneginf(prior.log_kernel(s.with_coordinates(x))).all()


def test_kernel_rho_drag_jacobians_and_BC_stratum_contract(prior):
    s = prior.sample(20, np.random.default_rng(21));x = s.coordinates.copy()
    x[:, 1] = math.log(800);x[:, 3] = math.log(.8)
    change = prior.log_kernel(s.with_coordinates(x))-prior.log_kernel(s)
    assert np.allclose(change, x[:, 1]+x[:, 3]-s.coordinates[:, 1]-s.coordinates[:, 3])
    assert np.array_equal(prior.normalization_stratum(s), s.boundary_type)
    assert not prior.metadata()['normalized_density_available']
    assert 'NOTuniform' in prior.metadata()['support_face_prior']


def test_exact_face_frequency_matches_independent_rejection_in_one_BC(prior):
    # Independent direct base-proposal counting gives feasible-area-weighted
    # face probabilities, which differ from the former uniform-face prior.
    rng = np.random.default_rng(80);base = prior.sample(14000, rng)
    x = base.coordinates.copy();x[:, 4:8] = rng.uniform(-.18, .18, (len(base), 4))
    faces = rng.integers(6, size=len(base));base = replace(base.with_coordinates(x),
        boundary_type=np.full(len(base), 1), support_face=faces)
    ok = prior.feasible_support(base)
    expected = np.bincount(faces[ok], minlength=6)/ok.sum()
    draws = prior.sample(14000, np.random.default_rng(81));ids = draws.boundary_type == 1
    actual = np.bincount(draws.support_face[ids], minlength=6)/ids.sum()
    assert np.max(abs(actual-expected)) < .04
    assert expected.max()-expected.min() > .07


def test_budget_failure_never_redraws_BC_or_returns_partial_batch(context):
    prior = ConditionalRejectionPrior(*context, max_attempts_per_particle=3)
    seen = []
    def reject(s):
        seen.append(s.boundary_type.copy())
        return np.zeros(len(s), dtype=bool)
    prior.feasible_support = reject
    with pytest.raises(SamplingBudgetExceeded) as caught:
        prior.sample(30, np.random.default_rng(41))
    assert len(seen) == 3 and all(np.array_equal(seen[0], b) for b in seen[1:])
    d = caught.value.diagnostics
    assert not d['completed'] and d['accepted'] == 0 and d['proposals'] == 90
    assert d['failed_particle_indices'] == list(range(30))


def test_decode_all_material_BC_load_fields_and_known_control(prior):
    s = prior.sample(300, np.random.default_rng(89));p = prior.decode(s)['physical']
    assert p.shape == (len(s), 25)
    assert np.allclose(p[:, :4], s.coordinates[:, :4])
    assert np.allclose(p[:, 10:16], prior.handle_centers_m.reshape(1, 6))
    assert np.allclose(p[s.boundary_type != 2, 4:7], p[s.boundary_type != 2, 7:10])
    twist = s.force_type == 3
    assert np.allclose(p[twist, 16:19]+p[twist, 19:22], 0)
    assert np.allclose(p[~twist, 19:22], 0)
    moving = s.boundary_type == 3
    assert np.allclose(p[~moving, 22:25], 0)
    assert np.allclose(np.linalg.norm(p[moving, 22:25], axis=-1), s.coordinates[moving, 11])
    known = {'force_type_index':2, 'peak_force_N':[0., 0., .02]}
    a = prior.decode(s, known);b = prior.decode(replace(s, force_type=(s.force_type+1) % 4), known)
    assert np.array_equal(a['physical'], b['physical'])
    assert np.allclose(a['physical'][:, 16:22], [0, 0, .02, 0, 0, 0])
    assert prior.hardware_descriptor()['attachment_geometry_known']


def test_zero_batches_reproducibility_and_missing_hardware_rejected(context, prior):
    a = prior.sample(20, np.random.default_rng(7));b = prior.sample(20, np.random.default_rng(7))
    assert np.array_equal(a.coordinates, b.coordinates)
    assert np.array_equal(a.support_face, b.support_face)
    assert len(a.take([1, 1, 0])) == 3
    empty = prior.sample(0)
    assert prior.log_kernel(empty).shape == (0,)
    assert prior.decode(empty)['physical'].shape == (0, 25)
    with pytest.raises(ValueError, match='mandatory'):
        ConditionalRejectionPrior(context[0], context[1], None)
    with pytest.raises(ValueError, match='budget'):
        ConditionalRejectionPrior(*context, max_attempts_per_particle=1.5)
