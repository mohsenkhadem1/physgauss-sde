"""Render a truth episode into a PAC-NeRF-format scene. Usage: render_scene.py EPISODE WORLD"""
import hashlib, json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import fem, render

R = Path(__file__).resolve().parents[2]
P = json.loads((Path(__file__).parent/'protocol.json').read_text())


def main(episode, world):
    z = np.load(R/f'runs/stem-trial-v01/truth/{episode}-{world}.npz')
    mesh = fem.Mesh(tuple(int(c) for c in z['cells']))
    assert np.allclose(mesh.X, z['X'])
    tris = mesh.surface_triangles()
    frames = z['X'][None]+z['U']
    out = R/f'runs/stem-trial-v01/scenes/{episode}-{world}'
    render.write_scene(out, frames, z['X'], tris, P['episodes'][episode]['fps'])
    h = hashlib.sha256()
    for f in sorted((out/'data').glob('m_*.png')):
        h.update(f.read_bytes())
    meta = dict(episode=episode, world=world, frames=len(frames), cameras=11, triangles=int(len(tris)), gic_camera_deviation=render.gic_camera_check(),
                images_sha256=h.hexdigest(), truth_sha256=hashlib.sha256((R/f'runs/stem-trial-v01/truth/{episode}-{world}.npz').read_bytes()).hexdigest())
    (out/'render-meta.json').write_text(json.dumps(meta, indent=1)); print(meta)


if __name__ == '__main__':
    main(*sys.argv[1:3])
