"""Seeded geometry and Newton-valued loads for the Isaac boundary experiment.

This module has no Isaac imports so geometry and experiment generation can be
checked independently of the simulation runtime. All coordinates are metres.
"""
from itertools import permutations

import numpy as np


def block_mesh(size=(0.24, 0.12, 0.08), cells=(12, 6, 4), center=(0, 0, 0.25)):
    size, cells, center = np.asarray(size), np.asarray(cells, dtype=int), np.asarray(center)
    if size.shape != (3,) or cells.shape != (3,) or np.any(size <= 0) or np.any(cells < 1):
        raise ValueError('Three positive sizes and cell counts are required')
    points = np.stack(np.meshgrid(*[np.linspace(-s/2, s/2, n+1) for s, n in zip(size, cells)], indexing='ij'), -1).reshape(-1, 3) + center
    shape = tuple(cells+1)
    tetrahedra = []
    for origin in np.ndindex(*cells):
        for order in permutations(range(3)):
            v = np.asarray(origin).copy(); ids = [np.ravel_multi_index(tuple(v), shape)]
            for axis in order:
                v[axis] += 1; ids.append(np.ravel_multi_index(tuple(v), shape))
            tetrahedra.append(ids)
    tetrahedra = np.asarray(tetrahedra, dtype=np.int32)
    determinants = np.linalg.det(points[tetrahedra[:, 1:]]-points[tetrahedra[:, :1]])
    negative = determinants < 0
    tetrahedra[negative, :2] = tetrahedra[negative, 1::-1]
    faces = {}
    for a, b, c, d in tetrahedra:
        for face in [(b, c, d), (a, d, c), (a, b, d), (a, c, b)]:
            key = tuple(sorted(face))
            if key in faces: del faces[key]
            else: faces[key] = face
    return points, tetrahedra, np.asarray(list(faces.values()), dtype=np.int32)


def lumped_masses(points, tetrahedra, density):
    volumes = np.linalg.det(points[tetrahedra[:, 1:]]-points[tetrahedra[:, :1]])/6
    if np.any(volumes <= 0) or density <= 0:
        raise ValueError('Positive tetrahedra and density are required')
    mass = np.zeros(len(points))
    np.add.at(mass, tetrahedra.reshape(-1), np.repeat(volumes*density/4, 4))
    return mass


def episode(points, faces, seed=31001, fixture_nodes=1, loads=3, seconds=3.0, force_scale=1.0, load_patch_nodes=1):
    rng = np.random.default_rng(seed)
    surface = np.unique(faces)
    if not 1 <= fixture_nodes <= len(surface) or not 1 <= loads < len(surface):
        raise ValueError('Invalid fixture or load count')
    if seconds <= 0 or not np.isfinite(force_scale) or force_scale < 0:
        raise ValueError('Positive duration and nonnegative finite force scale required')
    anchor = int(rng.choice(surface))
    # A single point is a translational pin, not a rotational clamp. More nodes
    # give a finite patch variant; all selected nodes are explicitly recorded.
    fixed = surface[np.argsort(np.linalg.norm(points[surface]-points[anchor], axis=1), kind='stable')[:fixture_nodes]]
    available = surface[np.min(np.linalg.norm(points[surface, None]-points[fixed][None], axis=-1), axis=1) > .055]
    if len(available) < loads: raise ValueError('Not enough surface nodes separated from fixture')
    selected = rng.choice(available, loads, replace=False)
    if not 1 <= load_patch_nodes <= len(available): raise ValueError('Invalid load patch size')
    patches=[];used=set()
    for node in selected:
        candidates=np.array([j for j in available if int(j) not in used and (j==node or j not in selected)])
        order=np.argsort(np.linalg.norm(points[candidates]-points[node],axis=1),kind='stable')
        patch=candidates[order[:load_patch_nodes]]
        if len(patch)!=load_patch_nodes or np.linalg.norm(points[patch]-points[node],axis=1).max()>.065:
            raise ValueError('Cannot form compact disjoint load patches')
        patches.append(patch.tolist());used.update(map(int,patch))
    pulses = []
    for i, node in enumerate(selected):
        direction = rng.normal(size=3); direction /= np.linalg.norm(direction)
        pulses.append({'node': int(node), 'direction_world': direction.tolist(),
                       'peak_newtons': float(force_scale*rng.uniform(.5, 2.0)),
                       'start_s': float(seconds*(.08+.65*i/loads)),
                       'duration_s': float(seconds*.25)})
    return {'seed': seed, 'fixture_nodes': fixed.tolist(), 'fixture_rest_m': points[fixed].tolist(),
            'fixture_type': 'translational point pin' if fixture_nodes == 1 else 'fixed node patch',
            'load_nodes': selected.tolist(), 'load_patches':patches, 'pulses': pulses,
            'force_frame': 'world', 'force_profile': 'sin(pi*(t-start)/duration)^2 inside pulse, zero outside',
            'seconds': seconds}


def forces_at(spec, time):
    force = np.zeros((len(spec['pulses']), 3), dtype=np.float32)
    for i, pulse in enumerate(spec['pulses']):
        phase = (time-pulse['start_s'])/pulse['duration_s']
        if 0 < phase < 1:
            force[i] = pulse['peak_newtons']*np.sin(np.pi*phase)**2*np.asarray(pulse['direction_world'])
    return force


def reference_gaussians(points, faces, count=4096, seed=991):
    rng = np.random.default_rng(seed)
    tri = points[faces]
    area = np.linalg.norm(np.cross(tri[:, 1]-tri[:, 0], tri[:, 2]-tri[:, 0]), axis=1)/2
    selected = rng.choice(len(faces), count, p=area/area.sum())
    weights = rng.dirichlet([1, 1, 1], count)
    position = np.einsum('ni,nij->nj', weights, tri[selected])
    color = .2+.6*(position-points.min(0))/(points.max(0)-points.min(0))
    return {'means_m': position.astype(np.float32), 'colors': color.astype(np.float32),
            'scales_m': np.full((count, 3), .002, dtype=np.float32),
            'opacities': np.full(count, .9, dtype=np.float32),
            'face_vertex_ids': faces[selected], 'barycentric_weights': weights.astype(np.float32)}
