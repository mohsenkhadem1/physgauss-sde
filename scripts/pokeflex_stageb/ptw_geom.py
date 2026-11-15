"""PhysTwin adapter, geometry: PhysTwin node set for every stage-B poke from its rest scan (free input).
Native PhysTwin builds its spring-mass from observed surface points (+ shape-completion interior points). Here:
  surface nodes  = rest-scan vertices voxel-downsampled at spacing h (one vertex per h-cell, closest to the cell centre)
  interior nodes = cell centres of an h-lattice inside the scan (ray-parity vote of realfem._voxelise) farther than
                   h/2 from every surface node
h is fixed per object (from the calibration rest volume, ~N_TARGET interior nodes), so global spring parameters
transfer between pokes.  Rest-scan vertices get inverse-distance weights over their K nearest nodes (PhysTwin's own
Gaussian/vertex interpolation is a kNN blend of node motion); the predicted surface keeps the scan topology.
Reads only stage-B inputs (free + calibration).  usage: python -I ptw_geom.py [OBJ ...]"""
import json, sys
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent/'pokeflex_v01'))
from sb_common import INP, WORK, OBJECTS, obj_info, load_poke
import realfem as RF  # inside test only (_voxelise); no data access

N_TARGET = 12000
K_INTERP = 4


def build(V, F, h):
    ys = float(V[:, 1].min())
    ins, (x0, y0, z0), (nx, ny, nz) = RF._voxelise(V, F, h, ys)
    I = np.argwhere(ins)
    Xi = np.c_[x0+(I[:, 0]+0.5)*h, y0+(I[:, 1]+0.5)*h, z0+(I[:, 2]+0.5)*h]
    key = np.floor((V-V.min(0))/h).astype(np.int64)
    cen = V.min(0)+(key+0.5)*h; dist = np.linalg.norm(V-cen, axis=1)
    order = np.lexsort((dist, key[:, 2], key[:, 1], key[:, 0]))
    k_s = key[order]; first = np.r_[True, np.any(k_s[1:] != k_s[:-1], axis=1)]
    Xs = V[order[first]]
    d, _ = cKDTree(Xs).query(Xi)
    Xi = Xi[d > 0.5*h]
    X = np.concatenate([Xs, Xi])
    dd, idx = cKDTree(X).query(V, k=K_INTERP)
    w = 1.0/np.maximum(dd, 1e-6); w /= w.sum(1, keepdims=True)
    return dict(X=X, n_surface=len(Xs), interp_idx=idx.astype(np.int32), interp_w=w.astype(np.float32), h=h, y_s=ys)


def main(objs):
    for obj in objs:
        info = obj_info(obj); d = WORK/'phystwin'/obj; d.mkdir(parents=True, exist_ok=True)
        t, k = info['calibration_pokes'][0]
        c = load_poke(obj, 'cal', t, k)
        ins, _, _ = RF._voxelise(c['V'], c['F'], 0.004, float(c['V'][:, 1].min()))
        vol = ins.sum()*0.004**3
        h = float(np.clip((vol/N_TARGET)**(1/3), 0.004, 0.02))
        rows = []
        for kind, lst in (('cal', info['calibration_pokes']), ('ho', info['heldout'])):
            for t, k in lst:
                p = load_poke(obj, kind, t, k); g = build(p['V'], p['F'], h)
                np.savez_compressed(d/f'geom_{kind}_{t}-p{k}.npz', **g)
                rows.append((kind, t, k, len(g['X']), g['n_surface']))
        (d/'geom.json').write_text(json.dumps(dict(h=h, volume_m3=float(vol), mass_kg=info['mass_kg'], pokes=rows), indent=1))
        print(obj, f'h {h*1e3:.1f} mm vol {vol*1e3:.2f} L nodes', [r[3] for r in rows][:4], '...', flush=True)


def dev():
    """Development take (debugging only): FoamDice_T1 poke 1 with FoamDice's h."""
    h = json.loads((WORK/'phystwin/FoamDice/geom.json').read_text())['h']
    p = dict(np.load(INP/'_dev/cal_FoamDice_T1-p1.npz')); d = WORK/'phystwin/_dev'; d.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(d/'geom_cal_FoamDice_T1-p1.npz', **build(p['V'], p['F'], h)); print('dev geom ok')


if __name__ == '__main__':
    dev() if sys.argv[1:] == ['--dev'] else main(sys.argv[1:] or OBJECTS)
