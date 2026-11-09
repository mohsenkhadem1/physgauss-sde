"""Render our predicted displacement fields on camera 0 with the known rest mesh and texture.
Usage: ours_render.py EPISODE   (renders every ours-fields/*-EPISODE.npz, both stages)"""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import fem, render

R = Path(__file__).resolve().parents[2]


def main(episode):
    mesh = fem.Mesh((33, 6, 6)); tris = mesh.surface_triangles(); cams, K = render.cameras()
    out = R/'runs/stem-trial-v01/renders'; out.mkdir(exist_ok=True)
    for f in sorted((R/'runs/stem-trial-v01/ours-fields').glob(f'*-{episode}.npz')):
        z = np.load(f); assert np.allclose(z['X'], mesh.X)
        for stage in ('D0', 'D0pD1'):
            imgs = [np.round(render.render(cams[0], K, mesh.X+U, mesh.X, tris)[0]*255).astype(np.uint8) for U in z[stage]]
            np.savez_compressed(out/f'ours-{f.stem}-{stage}.npz', images=np.stack(imgs)); print(f.stem, stage, flush=True)


if __name__ == '__main__':
    main(sys.argv[1])
