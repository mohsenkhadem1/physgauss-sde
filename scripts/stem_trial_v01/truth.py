"""Generate reference trajectories. Usage: truth.py EPISODE WORLD [--cells nx ny nz] [--dt-div k] [--tag t]
D2 is refused until reports/stem-trial-v01/prediction-freeze.json exists.
"""
import argparse, hashlib, json, sys, time
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import fem

R = Path(__file__).resolve().parents[2]
P = json.loads((Path(__file__).parent/'protocol.json').read_text())
OUT = R/'runs/stem-trial-v01/truth'


def force_fn(mesh, episode, world):
    w = P['worlds'][world]; g = mesh.gravity(w['rho'])
    if episode == 'D0':
        return lambda t: g
    lo, hi, F = {'D1': (-0.015, 0.015, 40.0), 'D2': (0.085, 0.115, 20.0)}[episode]
    f = mesh.body_force_region(lo, hi, [0, 0, F])
    return lambda t: g+fem.smooth_ramp(t, 0.25)*f


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('episode'); ap.add_argument('world')
    ap.add_argument('--cells', nargs=3, type=int, default=P['truth_numerics']['mesh_cells'])
    ap.add_argument('--dt-div', type=int, default=1); ap.add_argument('--tag', default='')
    a = ap.parse_args()
    if a.episode == 'D2' and not (R/'reports/stem-trial-v01/prediction-freeze.json').exists():
        raise SystemExit('D2 truth is sealed until predictions are frozen')
    e = P['episodes'][a.episode]; w = P['worlds'][a.world]
    mesh = fem.Mesh(tuple(a.cells)); model = fem.Elastic(mesh, w['E'], w['nu'], w['rho'])
    dt = P['truth_numerics']['dt_s']/a.dt_div; save = round(1/(e['fps']*dt))
    OUT.mkdir(parents=True, exist_ok=True)
    name = f"{a.episode}-{a.world}{a.tag}"
    t0 = time.time()
    U, stats = fem.newmark(model, force_fn(mesh, a.episode, a.world), e['duration_s'], dt, save,
                           log=lambda s: print(name, s, flush=True))
    assert U.shape[0] == e['frames'], U.shape
    np.savez_compressed(OUT/f'{name}.npz', X=mesh.X, U=U, elem=mesh.elem, cells=np.array(a.cells), dt=dt,
                        newton_iters=np.array(stats))
    meta = dict(episode=a.episode, world=a.world, cells=a.cells, dt=dt, seconds=time.time()-t0,
                newton_mean=float(np.mean(stats)), newton_max=int(np.max(stats)),
                protocol_sha256=hashlib.sha256((Path(__file__).parent/'protocol.json').read_bytes()).hexdigest(),
                fem_sha256=hashlib.sha256((Path(__file__).parent/'fem.py').read_bytes()).hexdigest())
    (OUT/f'{name}.json').write_text(json.dumps(meta, indent=1)); print(meta)


if __name__ == '__main__':
    main()
