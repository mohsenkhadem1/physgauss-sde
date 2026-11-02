"""Protocol invariants, before any reserved scenes are generated."""
import itertools
import numpy as np
import torch
from scripts.compositional_neural_mechanics.scenes import (
    split_cells, sample_scene, material_field, poses, TRAIN_GEOMETRIES,
)


def test_composition_split_contains_all_pairs_without_test_cells():
    split = split_cells()
    train, held = set(split['train']), set(split['held_composition'])
    assert len(train) == 18 and len(held) == 6 and train.isdisjoint(held)
    assert train | held == set(itertools.product(range(3), range(2), range(2), range(2)))
    for a, b in itertools.combinations(range(4), 2):
        assert {(c[a], c[b]) for c in held} <= {(c[a], c[b]) for c in train}


def test_scene_stiffness_ratios_and_material_coordinate_latents():
    scene = sample_scene(split_cells()['train'][0], np.random.default_rng(173), 'development-only')
    L, H, D = scene['geometry']['dimensions']
    E = 11250 * np.exp(scene['bulk_latent'][0])
    K = np.exp(np.array(scene['interface_latents'])[:, 0])
    np.testing.assert_allclose(K / (E * H * D / L), scene['support_stiffness_ratios'])
    q = torch.tensor([[0., .2, .3], [.5, .4, .8], [1., .8, .9]], dtype=torch.float64)
    params = torch.tensor([*scene['bulk_latent'], scene['gradient']], dtype=q.dtype, requires_grad=True)
    fixed = material_field(scene)(q)
    live = material_field(scene, params)(q)
    torch.testing.assert_close(fixed, live)
    live[:, 0].sum().backward()
    assert params.grad[0] == 3
    assert torch.isfinite(params.grad).all()


def test_commands_are_positions_and_orientations_not_force_labels():
    for geometry in TRAIN_GEOMETRIES:
        actions = poses(geometry)
        assert len(actions) == 6
        assert all(value.shape == (2, 6) and np.isfinite(value).all() for value in actions.values())
        assert np.isclose(actions['stretch'][1, 0] - actions['stretch'][0, 0], .24 * geometry.dimensions[0])
