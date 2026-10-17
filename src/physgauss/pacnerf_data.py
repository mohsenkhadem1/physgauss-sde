"""Strict official PAC-NeRF adapter. Future particle files are scoring-only.

This first transfer benchmark conditions on frame-zero geometry and uses all
frame-zero camera images to calibrate appearance. It is NOT geometry recovery
from a single video. Dynamic fitting uses just one camera at each selected time.
"""
import hashlib
import json
from pathlib import Path
import numpy as np
from PIL import Image
import torch


def read_particles(path):
    with Path(path).open('rb') as stream:
        lines = []
        while True:
            line = stream.readline()
            if not line: raise ValueError('Incomplete PLY header')
            lines.append(line.decode('ascii').strip())
            if line == b'end_header\n': break
        if lines[1] != 'format binary_little_endian 1.0': raise ValueError('Unexpected PLY format')
        props = [x for x in lines if x.startswith('property ')]
        if props != ['property double x', 'property double y', 'property double z']:
            raise ValueError('Expected the official xyz-only double particle format')
        n = int(next(x.split()[-1] for x in lines if x.startswith('element vertex ')))
        data = np.frombuffer(stream.read(), dtype='<f8').reshape(-1, 3).copy()
    if len(data) != n or not np.isfinite(data).all(): raise ValueError('Invalid particle payload')
    return torch.from_numpy(data)


def farthest(points, count):
    selected = [int(points.square().sum(-1).argmin())]
    distance = torch.full((len(points),), torch.inf, dtype=points.dtype)
    for _ in range(count-1):
        distance = torch.minimum(distance, (points-points[selected[-1]]).square().sum(-1))
        selected.append(int(distance.argmax()))
    return torch.tensor(selected)


def load_scene(root, scene, size=32):
    root = Path(root); directory = root/'data'/scene
    entries = json.loads((directory/'all_data.json').read_text())
    cameras, times, images, backgrounds = {}, {}, {}, {}
    for e in entries:
        name = Path(e['file_path']).stem.split('_')
        camera, frame = int(name[1]), int(name[2])
        if frame >= 0:
            c2w = torch.tensor(e['c2w'], dtype=torch.float64)
            # Upstream get_rays uses OpenGL (+x right,+y up,-z forward).
            rotation = torch.diag(torch.tensor([1., -1., -1.], dtype=torch.float64))@c2w[:, :3].T
            k = torch.tensor(e['intrinsic'], dtype=torch.float64)*(size/800.)
            k[2, 2] = 1.
            cameras[camera] = (c2w[:, 3], rotation, k)
            times[frame] = e['time']
        path = directory/e['file_path']
        with Image.open(path) as image:
            if image.size != (800, 800): raise ValueError('Revisit intrinsic scaling for non-800px release')
            rgb = torch.from_numpy(np.array(image.convert('RGB').resize((size, size), Image.Resampling.LANCZOS)).copy()).double()/255
        if frame == -1: backgrounds[camera] = rgb
        elif frame >= 0: images[camera, frame] = rgb
    if sorted(times) != list(range(14)) or sorted(cameras) != list(range(11)):
        raise ValueError('Unexpected named elastic-scene timeline/cameras')
    rest_all = read_particles(root/'simulation_data'/scene/'0.ply')
    # Fixed stride then deterministic FPS: selection never sees later geometry.
    candidates = torch.arange(0, len(rest_all), max(1, len(rest_all)//6000))
    gauss_indices = candidates[farthest(rest_all[candidates], 128)]
    handle_indices = gauss_indices[farthest(rest_all[gauss_indices], 32)]
    rest, centers = rest_all[handle_indices], rest_all[gauss_indices]
    d2 = torch.cdist(centers, rest).square()
    weights = torch.softmax(-d2/.10**2, -1)
    return {'scene': scene, 'root': root, 'size': size, 'cameras': cameras,
            'times': torch.tensor([times[i] for i in range(14)], dtype=torch.float64),
            'images': images, 'backgrounds': backgrounds, 'rest': rest,
            'gaussians': centers, 'weights': weights, 'handle_indices': handle_indices,
            'gaussian_indices': gauss_indices, 'particle_count': len(rest_all),
            'geometry_sha256': hashlib.sha256(rest_all.numpy().tobytes()).hexdigest()}


def evaluation_particles(data):
    """Call only after inference/checkpointing. Nothing else reads future PLYs."""
    rows = []
    for frame in range(14):
        points = read_particles(data['root']/'simulation_data'/data['scene']/f'{frame}.ply')
        if len(points) != data['particle_count']: raise ValueError('Particle correspondence count changed')
        rows.append(points[data['gaussian_indices']])
    return torch.stack(rows)
