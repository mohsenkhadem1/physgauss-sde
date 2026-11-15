"""Export Figure-1 scenes for the Blender renders (blender_render.py). Same sealed inputs and error definition as
figure1.py / videos.py. usage: python blender_export.py S3 W1000  -> runs/fig1-v01/blender/<scen>-<world>.npz"""
import sys
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import figure1 as F
import block as B, fem

OUT = F.R/'runs/fig1-v01/blender'; OUT.mkdir(parents=True, exist_ok=True)
NPTS = 12000


def export(scen, w):
    X, U, tg, sphere = F.truth(scen, w)
    z = np.load(F.BK/f'truth/D2-{w}.npz') if scen == 'S2' else np.load(F.ST/f"truth/{F.SCEN[scen]['ep']}-{w}.npz")
    m = B.mesh(tuple(z['cells'])) if scen == 'S2' else fem.Mesh(tuple(z['cells']))
    tris = m.surface_triangles()
    n = len(U); rng = np.random.default_rng(0); d = dict(X=X, U=U.astype(np.float32), tris=tris.astype(np.int32), n=n)
    if sphere is not None:
        d['sphere'] = sphere
    allerr = []
    for i, meth in enumerate(F.METHODS):
        r, p, sc = F.shape(scen, meth, w); sel = rng.permutation(len(r))[:NPTS]
        e = np.stack([F.motion_error(r[sel], p[:, sel], k, X, U) for k in range(n)]).astype(np.float32)
        d[f'm{i}_pos'] = p[:, sel].astype(np.float32); d[f'm{i}_err'] = e; allerr.append(e.ravel())
        d[f'm{i}_spacing'] = float(np.percentile(cKDTree(p[0, sel]).query(p[0, sel], k=2)[0][:, 1], 90))  # p90 nearest-neighbour
        if sc is not None:
            d[f'm{i}_sphere'] = sc
        d[f'm{i}_rms'] = np.sqrt(((F.trajectory(scen, meth, w, 'D1')-tg)**2).sum(-1).mean(-1))*1e3
    # load glyphs (known commands, shown in every panel): arrow tip follows the true loaded region
    t = np.arange(n)/F.FPS
    if scen == 'S1':  # 20 N body force on the slab x in [0.085, 0.115), direction (0,-0.8,-0.6), 0.25 s ramp then held
        dirv = np.array([0, -0.8, -0.6]); sel = (X[:, 0] >= 0.085) & (X[:, 0] < 0.115) & (X[:, 1] > 0.639)
        d['arrow_tip'] = np.array([0.1, 0.64, 0.0])+U[:, sel].mean(1); d['arrow_dir'] = dirv; d['arrow_scale'] = fem.smooth_ramp(t, 0.25)
    if scen == 'S2':  # 30 N down over the 4x4 cm patch at x = -0.07 on the top face, 0.25 s ramp
        bx = B.press_box('D2'); sel = (np.abs(X[:, 0]+0.07) <= 0.02) & (np.abs(X[:, 2]) <= 0.02) & (X[:, 1] > B.TOP-1e-6)
        d['arrow_tip'] = np.array([-0.07, B.TOP+0.004, 0.0])+U[:, sel].mean(1); d['arrow_dir'] = np.array([0, -1.0, 0])
        d['arrow_scale'] = fem.smooth_ramp(t, B.EPISODE['ramp_s']); d['pad_half'] = 0.02
    d['vmax'] = float(np.ceil(np.percentile(np.concatenate(allerr), 95)))
    d['methods'] = np.array(F.METHODS); d['labels'] = np.array([F.LABEL[x] for x in F.METHODS]); d['colors'] = np.array([F.COL[x] for x in F.METHODS])
    d['title'] = F.SCEN[scen]['title']
    np.savez_compressed(OUT/f'{scen}-{w}.npz', **d)
    print(OUT/f'{scen}-{w}.npz', 'tris', len(tris), 'vmax', d['vmax'])


if __name__ == '__main__':
    export(sys.argv[1], sys.argv[2])
