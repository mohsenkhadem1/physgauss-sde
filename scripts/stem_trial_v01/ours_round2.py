"""Our method, round 2: posterior from supplied tracks or RGB-D measurements; D2 and D2b predictions.
Usage: ours_round2.py SENSOR WORLD   (SENSOR in tracks, rgbd, rgb; STEM_OFFGRID=1 for the off-grid prior)
Writes runs/stem-trial-v01/{ours|ours-offgrid}/round2-SENSOR-WORLD.{npz,json}."""
import json, os, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from ours import OUT, RHO_GRID, C_GRID, NU_GRID, MARKERS_RGBD, P, R, observe

LOADS = {'D2': np.array([0, 0, 20.0]), 'D2b': np.array([0, -15.0, 20.0]), 'D2c': np.array([0, -16.0, -12.0])}
DELTA = 3e-3


def main(sensor, world):
    b = dict(np.load(OUT/'bank.npz')); b2 = dict(np.load(OUT/'bank2.npz'))
    if sensor == 'tracks':
        y0, y1 = observe('D0', world), observe('D1', world)
        mg, mp = b['mark_g'], b['mark_p']; sd = P['ours']['noise_sd_m']
    else:
        d = 'rgbd' if sensor == 'rgbd' else 'rgb'  # rgb: two-view triangulation of tracked RGB, no depth
        o0 = np.load(R/f'runs/stem-trial-v01/{d}/obs-D0-{world}.npz'); o1 = np.load(R/f'runs/stem-trial-v01/{d}/obs-D1-{world}.npz')
        assert o0['status'].all() and o1['status'].all()
        y0, y1 = o0['y']-MARKERS_RGBD[None], o1['y']-MARKERS_RGBD[None]
        mg, mp = b2['mark2_g'], b2['mark2_p']; sd = None
    n0 = len(y0)
    r0 = ((mg[:, :, :n0]-y0[None, None])**2).sum((2, 3, 4))
    if sd is None:  # plug-in noise level from the best D0 grid fit (protocol_v02 B)
        sd = float(np.sqrt(r0.min()/y0.size))
    pred1 = mg[:, :, None]+(40.0/RHO_GRID)[None, None, :, None, None, None]*mp[:, :, None]
    r1 = ((pred1-y1[None, None, None])**2).sum((3, 4, 5))
    ll0 = np.broadcast_to((-r0/(2*sd**2))[:, :, None], r1.shape); ll1 = -r1/(2*sd**2)
    out, arrays = dict(sensor=sensor, world=world, noise_sd_m=sd, offgrid=OUT.name.endswith('offgrid')), {}
    for stage, ll in [('D0', ll0), ('D0+D1', ll0+ll1)]:
        w = np.exp(ll-ll.max()); w /= w.sum(); arrays[f'w_{stage}'] = w
        marg = lambda ax: w.sum(tuple(i for i in range(3) if i != ax))
        q = lambda grid, p: [float(grid[min(np.searchsorted(np.cumsum(p), x), len(grid)-1)]) for x in (.05, .5, .95)]
        st = dict(c_q=q(C_GRID, marg(0)), nu_q=q(NU_GRID, marg(1)), rho_q=q(RHO_GRID, marg(2)))
        for ep, F in LOADS.items():
            unit = F[2]*b['st_h']+F[1]*b2['st_hy']  # response to F for unit density (C,NU,T,S,3)
            traj = b['st_g'][:, :, None]+(1/RHO_GRID)[None, None, :, None, None, None]*unit[:, :, None]
            mean = np.einsum('cnr,cnrfsj->fsj', w, traj)
            rms = np.sqrt(((traj-mean)**2).sum(-1).mean((-1, -2)))
            arrays[f'pred_{ep}_{stage}'] = mean
            st[f'risk_{ep}'] = float((w*(rms > DELTA)).sum())
        out[stage] = st
    tag = f"{os.environ.get('STEM_ROUND', 'round3')}-{sensor}-{world}"  # never overwrite sealed round-2 files
    np.savez_compressed(OUT/f'{tag}.npz', **arrays)
    (OUT/f'{tag}.json').write_text(json.dumps(out, indent=1)); print(json.dumps(out))


if __name__ == '__main__':
    main(*sys.argv[1:3])
