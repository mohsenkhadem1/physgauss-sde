"""No responses/recipes are opened: reference geometry is constructed directly."""
from dataclasses import replace
import math
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'experiments'))
from boundary_block import block_mesh
from campaign_physical_prior import (
    ConditionalPhysicalPrior, FACE_NORMALS, PriorState,
)


@pytest.fixture
def prior():
    return ConditionalPhysicalPrior(block_mesh()[0], [[.12, 0., .25], [-.04, -.06, .25]])


def test_material_marginals_are_simulator_uniform_not_old_flow(prior):
    state = prior.sample(16000, np.random.default_rng(14))
    x = state.coordinates
    for values, low, high in [(x[:, 0], math.log(1500), math.log(20000)),
                             (np.exp(x[:, 1]), 400, 850), (x[:, 2], .28, .42),
                             (np.exp(x[:, 3]), .35, .85)]:
        assert values.min() >= low-1e-12 and values.max() <= high+1e-12
        assert abs((values.mean()-low)/(high-low)-.5) < .012
        assert np.allclose(np.quantile((values-low)/(high-low), [.1, .5, .9]),
                           [.1, .5, .9], atol=.012)
    assert set(state.boundary_type) == set(range(4))
    assert set(state.force_type) == set(range(4))
    assert np.isfinite(prior.log_prob(state)).all()


def test_intrinsic_density_includes_material_jacobians(prior):
    s = prior.sample(1, np.random.default_rng(2))
    x = (prior.lower+prior.upper)[None]/2
    s = s.with_coordinates(x)
    other = x.copy(); other[0, 1] = math.log(800); other[0, 3] = math.log(.8)
    difference = prior.log_prob(s.with_coordinates(other))-prior.log_prob(s)
    assert difference[0] == pytest.approx(other[0, 1]+other[0, 3]-x[0, 1]-x[0, 3])
    # Continuous density integrates to one: only rho and drag are nonconstant
    # on this chart. Their exp Jacobians integrate exactly to physical widths.
    for dim, physical_width in [(1, 450), (3, .5)]:
        grid = np.linspace(prior.lower[dim], prior.upper[dim], 1001)
        assert np.trapezoid(np.exp(grid)/physical_width, grid) == pytest.approx(1, abs=1e-7)


def test_exact_discrete_normalization_including_inactive_cells(prior):
    s = prior.sample(1, np.random.default_rng(8))
    x = ((prior.lower+prior.upper)/2)[None]
    continuous_log = -np.log(prior.upper-prior.lower).sum()
    continuous_log += math.log(prior.upper[1]-prior.lower[1])+x[0, 1]-math.log(450)
    continuous_log += math.log(prior.upper[3]-prior.lower[3])+x[0, 3]-math.log(.5)
    mass = 0.
    for bc in range(4):
        for family in range(4):
            for face in prior.feasible_faces[bc]:
                c0 = prior.feasible_cells[bc, face]
                c1 = prior.feasible_cells[bc, (face+3) % 6] if bc == 2 else np.arange(prior.grid**2)
                for load0 in prior.incident_faces[0]:
                    for load1 in prior.incident_faces[1]:
                        row = PriorState(x, np.array([bc]), np.array([family]), np.array([face]),
                            np.array([[c0[0], c1[0]]]), np.array([[load0, load1]]))
                        mass += np.exp(prior.log_prob(row)[0]-continuous_log)*len(c0)*len(c1)
    assert mass == pytest.approx(1, abs=1e-12)


def test_supports_on_faces_and_actual_selected_nodes_are_clear(prior):
    s = prior.sample(700, np.random.default_rng(32))
    d = prior.decode(s); p = d['physical']
    for i in range(len(s)):
        bc = s.boundary_type[i]
        for j in range(2 if bc == 2 else 1):
            face = (s.support_face[i]+3*j) % 6
            center = p[i, 4+3*j:7+3*j]
            assert np.isclose(center[face % 3],
                (prior.minimum if face < 3 else prior.maximum)[face % 3])
            face_points = prior.reference_points[np.isclose(prior.reference_points[:, face % 3], center[face % 3])]
            count = [1, 16, 9, 16][bc]
            chosen = face_points[np.argsort(np.linalg.norm(face_points-center, axis=1))[:count]]
            assert np.linalg.norm(chosen[:, None]-prior.handle_centers_m[None], axis=-1).min() > .027
            assert np.linalg.norm(chosen-center, axis=-1).max() <= .09
    assert np.allclose(p[s.boundary_type != 2, 4:7], p[s.boundary_type != 2, 7:10])
    assert np.allclose(p[:, 10:16], prior.handle_centers_m.reshape(1, 6))


def test_force_geometry_and_moving_boundary_all_uncertain(prior):
    s = prior.sample(800, np.random.default_rng(52));p = prior.decode(s)['physical']
    f0, f1 = p[:, 16:19], p[:, 19:22]
    normal = FACE_NORMALS[s.load_faces[:, 0]]
    amplitude = s.coordinates[:, 8]*np.where(s.boundary_type == 0, .25, 1.)
    assert np.allclose(np.linalg.norm(f0, axis=-1), amplitude)
    for family, sign in [(0, -1), (1, 1)]:
        ids = s.force_type == family
        assert np.allclose((f0[ids]*normal[ids]).sum(-1), sign*amplitude[ids])
    shear = s.force_type == 2
    assert np.allclose((f0[shear]*normal[shear]).sum(-1), 0.)
    twist = s.force_type == 3
    assert np.allclose(f0[twist]+f1[twist], 0.)
    assert np.allclose(f0[twist]@(prior.handle_centers_m[1]-prior.handle_centers_m[0]), 0.)
    assert np.allclose(f1[~twist], 0.)
    moving = s.boundary_type == 3;motion = p[:, 22:25]
    assert np.allclose(motion[~moving], 0.)
    assert np.allclose(np.linalg.norm(motion[moving], axis=-1), s.coordinates[moving, 11])
    assert np.allclose((motion[moving]*FACE_NORMALS[s.support_face[moving]]).sum(-1), 0.)
    assert np.std(s.coordinates[moving, 10]) > 1 and np.std(s.coordinates[moving, 11]) > .0005


def test_edge_normals_are_categories_not_arbitrary_projection():
    prior = ConditionalPhysicalPrior(block_mesh()[0], [[.12, .06, .25], [-.12, -.06, .21]])
    assert set(prior.incident_faces[0]) == {3, 4}
    assert set(prior.incident_faces[1]) == {0, 1, 2}
    s = prior.sample(300, np.random.default_rng(3))
    assert set(s.load_faces[:, 0]) == {3, 4}
    assert set(s.load_faces[:, 1]) == {0, 1, 2}
    bad = s.load_faces.copy(); bad[0, 0] = 2
    assert np.isneginf(prior.log_prob(replace(s, load_faces=bad))[0])


def test_proper_inactive_nuisances_and_known_load_replacement(prior):
    s = prior.sample(1, np.random.default_rng(6))
    # Choose an allowed fixed-patch state with non-twist loads.
    s = prior.sample(200, np.random.default_rng(7))
    s = s.take(np.flatnonzero((s.boundary_type == 1)&(s.force_type != 3))[0])
    x = s.coordinates.copy(); x[:, 6:8] = 1-x[:, 6:8]
    x[:, 9:11] = .7; x[:, 11] = .009
    cells = s.support_cells.copy();cells[:, 1] = (cells[:, 1]+1) % prior.grid**2
    other = replace(s.with_coordinates(x), support_cells=cells)
    assert np.allclose(prior.decode(s)['physical'], prior.decode(other)['physical'])
    assert np.allclose(prior.log_prob(s), prior.log_prob(other))
    all_states = prior.sample(100, np.random.default_rng(19))
    altered = replace(all_states, force_type=(all_states.force_type+1) % 4)
    control = {'force_type_index':2, 'peak_force_N':[0., 0., .02], 'center_m':prior.handle_centers_m[0]}
    first = prior.decode(all_states, control);second = prior.decode(altered, control)
    assert np.array_equal(first['physical'], second['physical'])
    assert np.all(first['force_type'] == 2)
    assert np.allclose(first['physical'][:, 16:22], [0., 0., .02, 0., 0., 0.])
    with pytest.raises(ValueError, match='center'):
        prior.decode(all_states, dict(control, center_m=prior.handle_centers_m[1]))
    with pytest.raises(ValueError, match='waveform'):
        prior.decode(all_states, dict(control, start_s=.3))


def test_torch_numpy_decode_density_and_gradients_match(prior):
    s = prior.sample(20, np.random.default_rng(61))
    x = torch.tensor(s.coordinates, dtype=torch.float64, requires_grad=True)
    np_result = prior.decode(s)
    result = prior.decode_torch(s, x)
    assert np.allclose(result['physical'].detach().numpy(), np_result['physical'], atol=1e-14)
    lp = prior.log_prob_torch(s, x)
    assert np.allclose(lp.detach().numpy(), prior.log_prob(s), atol=1e-13)
    lp.sum().backward(retain_graph=True)
    target = np.zeros((len(s), 12));target[:, [1, 3]] = 1
    assert np.allclose(x.grad.numpy(), target)
    x.grad.zero_()
    result['physical'].square().sum().backward()
    assert torch.isfinite(x.grad).all()
    assert torch.all(x.grad[:, 0] != 0)
    # Changing original loads cannot affect a calibrated control response.
    control = {'force_type_index':0, 'peak_force_N':[.02, 0., 0.]}
    assert np.allclose(prior.decode_torch(s, known_control=control)['physical'].numpy(), prior.decode(s, control)['physical'])


def test_invalid_prior_mass_is_zero_and_empty_batches_work(prior):
    s = prior.sample(3, np.random.default_rng(12))
    x = s.coordinates.copy(); x[0, 0] = prior.lower[0]-.1;x[1, 9] = math.pi;x[2, 3] = np.nan
    assert np.isneginf(prior.log_prob(s.with_coordinates(x))).all()
    with pytest.raises(ValueError, match='outside'):
        prior.decode(s.with_coordinates(x))
    cells = s.support_cells.copy();cells[:, 0] = prior.grid**2
    assert np.isneginf(prior.log_prob(replace(s, support_cells=cells))).all()
    floats = replace(s, boundary_type=s.boundary_type.astype(float),
                     force_type=s.force_type.astype(float), support_face=s.support_face.astype(float),
                     support_cells=s.support_cells.astype(float), load_faces=s.load_faces.astype(float))
    assert np.array_equal(prior.decode(floats)['physical'], prior.decode(s)['physical'])
    assert np.isneginf(prior.log_prob(replace(floats, force_type=floats.force_type+.5))).all()
    assert prior.log_prob(prior.sample(0)).shape == (0,)
    assert prior.decode(prior.sample(0))['physical'].shape == (0, 25)
    with pytest.raises(ValueError, match='surface'):
        ConditionalPhysicalPrior(block_mesh()[0], [[0., 0., .25], [.12, 0., .25]])
    with pytest.raises(ValueError, match='Distinct'):
        ConditionalPhysicalPrior(block_mesh()[0], [[.12, 0., .25], [.12, 0., .25]])


def test_reproducible_prior_fingerprint_and_resampling(prior):
    a = prior.sample(8, np.random.default_rng(44));b = prior.sample(8, np.random.default_rng(44))
    assert np.array_equal(a.coordinates, b.coordinates)
    assert np.array_equal(a.support_cells, b.support_cells)
    repeated = a.take([3, 3, 0]);assert np.array_equal(repeated.coordinates[0], repeated.coordinates[1])
    assert prior.metadata()['prior_sha256'] == prior.metadata()['prior_sha256']
    assert 'Not exact' in prior.metadata()['qualification']
    other = ConditionalPhysicalPrior(block_mesh()[0], [[.12, 0., .25], [-.04, -.06, .25]], grid=6)
    assert prior.metadata()['prior_sha256'] != other.metadata()['prior_sha256']


def test_complete_public_attachment_descriptor_and_fixture_clearance():
    from isaac_dataset import geometry
    # A declaredTRAIN geometry seed, no observations/trajectory/recipe files.
    points, _, _, spec = geometry(dict(cells=[12, 6, 4], seed=531000,
        boundary_type='point_pin', force_type='push_pulse', load_patch_nodes=9, seconds=3.))
    handles = points[spec['load_nodes']]
    patches = np.array(spec['load_patches'])
    prior = ConditionalPhysicalPrior(points, handles, handle_patch_node_ids=patches)
    descriptor = prior.hardware_descriptor()
    assert descriptor['attachment_geometry_known']
    offsets = points[patches]-handles[:, None]
    expected = np.maximum(.001*np.mean((offsets**2).sum(-1)[..., None]-offsets**2, axis=1), .4*.001*.003**2)
    assert np.allclose(descriptor['local_offsets_m'], offsets)
    assert np.allclose(descriptor['diagonal_inertia_kg_m2'], expected)
    for j, face in enumerate(prior.incident_faces):
        normal = np.asarray(spec['reference_load_normals'][j])
        axis = abs(normal).argmax()
        assert int(axis+3*(normal[axis] > 0)) in face
    s = prior.sample(200, np.random.default_rng(94));p = prior.decode(s)['physical']
    for i in range(len(s)):
        for j in range(2 if s.boundary_type[i] == 2 else 1):
            f = (s.support_face[i]+3*j) % 6;center = p[i, 4+3*j:7+3*j]
            fp = points[np.isclose(points[:, f % 3], center[f % 3])]
            count = [1, 16, 9, 16][s.boundary_type[i]]
            fixed = fp[np.argsort(np.linalg.norm(fp-center, axis=1))[:count]]
            assert np.linalg.norm(fixed[:, None]-points[patches].reshape(1, -1, 3), axis=-1).min() > .027
    center_only = ConditionalPhysicalPrior(points, handles)
    assert prior.metadata()['prior_sha256'] != center_only.metadata()['prior_sha256']
    bad = patches.copy();bad[0, 0] = bad[0, 1]
    with pytest.raises(ValueError, match='distinct'):
        ConditionalPhysicalPrior(points, handles, handle_patch_node_ids=bad)
