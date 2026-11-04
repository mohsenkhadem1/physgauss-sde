"""Presentation-only held clips; no fitting or benchmark scoring changes.

Run from repository root with PYTHONPATH=. and the repository Python environment.
Selection is world00, both queries, every family, regardless of outcome.
"""
import os
os.environ['MPLBACKEND'] = 'Agg'
import argparse
import copy
import hashlib
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import imageio_ffmpeg
from scripts.complex_scenes_v18.held_appearance import closed_census
from scripts.complex_scenes_v18.pair_contact import material_surface
from scripts.complex_scenes_v18.world_queries import Query, build_boundary_geometry
from scripts.complex_scenes_v18.render_development_cases import surroundings
from scripts.complex_scenes_v18.geometry import RigidPlane


def read(folder):
    path = folder / 'trajectory.npz'
    seal = json.loads((folder / 'complete.json').read_text())
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    assert seal['completed'] and seal['trajectory_sha256'] == digest
    with np.load(path) as data:
        result = {k: data[k].copy() for k in ('times_s', 'surface_positions_m', 'rigid_centers_m') if k in data}
    x = result['surface_positions_m']
    if np.iscomplexobj(x):
        assert not np.any(x.imag)
        result['surface_positions_m'] = x.real
    assert np.isfinite(result['surface_positions_m']).all()
    return result, dict(path=str(path), sha256=digest)


def render(campaign, world, index, rows, output):
    selected = [r for r in rows if r['world_id'] == world and r['query_index'] == index]
    assert len(selected) == 5
    query = Query(**selected[0]['public_query'])
    refs = list((campaign / 'references' / world / str(index)).glob('*/complete.json'))
    assert len(refs) == 1
    ref, source = read(refs[0].parent)
    panels = [('Simulation reference', ref, source)]
    names = {'ours': 'Ours', 'analytic-control': 'Analytic control', 'spring-gaus': 'Spring-Gaus', 'phystwin': 'PhysTwin', 'uniphy': 'UniPhy'}
    for method in names:
        row = next(r for r in selected if r['method'] == method)
        if row['state'] == 'sealed':
            data, source = read(Path(row['physical_output']))
            assert data['surface_positions_m'].shape == ref['surface_positions_m'].shape
            assert np.allclose(data['times_s'], ref['times_s'], atol=1e-9, rtol=0)
            panels.append((names[method], data, source))
        else:
            panels.append((names[method], None, dict(state=row['state'])))
    scene, _, _ = build_boundary_geometry(query)
    # The existing sphere/bowl helper expects a center; planes use a point.
    display_scene = copy.copy(scene)
    display_scene.models = list(scene.models)
    display_scene.models[0] = copy.copy(scene.models[0])
    display_scene.models[0].colliders = tuple(c for c in scene.models[0].colliders if not isinstance(c, RigidPlane))
    planes = [c for c in scene.models[0].colliders if isinstance(c, RigidPlane)]
    q, faces = material_surface(9)
    mid = q[faces].mean(1)
    checker = (np.floor(mid[:, 0]*8)+np.floor(mid[:, 1]*4)+np.floor(mid[:, 2]*4)) % 2
    all_x = np.concatenate([data['surface_positions_m'].reshape(-1, 3) for _, data, _ in panels if data is not None])
    lo = np.minimum(all_x.min(0)-.025, [-.05, -.11, -.13])
    hi = np.maximum(all_x.max(0)+.025, [.17, .11, .13])
    center = (lo+hi)/2
    half = max(hi-lo)/2
    lo, hi = center-half, center+half
    output.mkdir(parents=True, exist_ok=True)
    stem = f'held-{world}-query{index}'
    movie = output / (stem+'.mp4')
    assert not movie.exists(), 'Preserve existing media'
    fig = plt.figure(figsize=(16, 9), dpi=100)
    axes = [fig.add_subplot(2, 3, i+1, projection='3d') for i in range(6)]
    fig.subplots_adjust(left=.01, right=.99, bottom=.06, top=.89, wspace=.01, hspace=.05)
    fig.suptitle(f'{query.family.replace("_", " ")} | world 00 | held interaction {index}', fontsize=18)
    clock = fig.text(.5, .925, '', ha='center', fontsize=12)
    fig.text(.5, .025, 'Simulation reference | true deformation scale | shared axes | orbiting camera | geometry visualization, not image-quality evaluation', ha='center', fontsize=10)
    times = ref['times_s']
    writer = imageio_ffmpeg.write_frames(str(movie), (1600, 900), fps=24, codec='libx264', quality=8, macro_block_size=1)
    writer.send(None)
    try:
        for frame, t in enumerate(times):
            clock.set_text(f'{t:.3f} s simulated time | new interaction, no material refit | {1/(24*np.median(np.diff(times))):.1f}x slow motion')
            for ax, (name, data, _) in zip(axes, panels):
                ax.clear()
                ax.set_title(name, fontsize=14, pad=0)
                ax.set(xlim=(lo[0], hi[0]), ylim=(lo[1], hi[1]), zlim=(lo[2], hi[2]))
                ax.set_box_aspect((1, 1, 1))
                ax.view_init(elev=23, azim=-70+45*frame/max(len(times)-1, 1))
                ax.set_axis_off()
                if data is None:
                    state = next(p[2]['state'] for p in panels if p[0] == name)
                    ax.text2D(.5, .5, state.replace('_', ' '), transform=ax.transAxes, ha='center', fontsize=14)
                    continue
                for body, x in enumerate(data['surface_positions_m'][frame]):
                    color = np.array([.18, .49, .75] if body == 0 else [.91, .45, .17])
                    ax.add_collection3d(Poly3DCollection(x[faces], facecolors=(.75+.25*checker[:, None])*color, edgecolors=(0, 0, 0, .06), linewidths=.1))
                rigid = data.get('rigid_centers_m', np.empty((len(times), 0, 3)))[frame]
                surroundings(ax, display_scene, t, rigid)
                for plane in planes:
                    xx, yy = np.meshgrid([lo[0], hi[0]], [lo[1], hi[1]])
                    assert np.allclose(plane.normal, [0, 0, 1]), 'Only horizontal plane display supported'
                    ax.plot_surface(xx, yy, np.full_like(xx, plane.point[2]), color='#94a3b8', alpha=.12)
            fig.canvas.draw()
            writer.send(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy())
            if frame == round(.75*(len(times)-1)):
                fig.savefig(output / (stem+'-preview.png'), dpi=100)
    finally:
        writer.close()
        plt.close(fig)
    record = dict(world=world, query=index, selection='First predefined world, both queries, all families; no outcome selection', sources={name: source for name, _, source in panels}, frames=len(times), fps=24, deformation_scale=1, movie_sha256=hashlib.sha256(movie.read_bytes()).hexdigest())
    (output / (stem+'.json')).write_text(json.dumps(record, indent=2)+'\n')
    print(stem, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--family', required=True)
    parser.add_argument('--query', required=True, type=int, choices=(0, 1))
    args = parser.parse_args()
    campaign = Path('runs/complex-scenes-v18/held-campaign-01')
    render(campaign, args.family+'-world00', args.query, closed_census(campaign), Path('output/complex-scenes-v18/held-videos'))
