"""Figure-1 S3 reference: nonlinear stem (33,6,6) released under gravity and hit by the rigid striker.
Usage: striker_truth.py WORLD [--cells nx ny nz] [--dt-div k] [--tag t]. Gated on the S3 freeze."""
import argparse, json, sys, time
from pathlib import Path
import numpy as np
R = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(R/'scripts/stem_trial_v01'))
import fem, contact as C
P = json.loads((R/'scripts/stem_trial_v01/protocol.json').read_text())
STR = dict(r=0.03, m=0.3, kappa=1e11, c0=[0.17, 0.60, 0.0755], v0=[0, 0, -1.0])
ap = argparse.ArgumentParser(); ap.add_argument('world'); ap.add_argument('--cells', nargs=3, type=int, default=[33, 6, 6])
ap.add_argument('--dt-div', type=int, default=1); ap.add_argument('--tag', default=''); a = ap.parse_args()
if not (R/'reports/fig1-v01/prediction-freeze-S3.json').exists():
    raise SystemExit('striker truth sealed until S3 predictions are frozen')
w = P['worlds'][a.world]; mesh = fem.Mesh(tuple(a.cells)); model = fem.Elastic(mesh, w['E'], w['nu'], w['rho'])
g = mesh.gravity(w['rho']); dt = 1/2400/a.dt_div; t0 = time.time()
U, Cs, V, st = C.newmark_striker(model, lambda t: g, C.Striker(STR['r'], STR['m'], STR['kappa'], STR['c0'], STR['v0']),
                                 0.75, dt, round(1/(48*dt)), nonlinear=True, log=lambda s: print(a.world, s, flush=True))
out = R/'runs/stem-trial-v01/truth'; np.savez_compressed(out/f'D3-{a.world}{a.tag}.npz', X=mesh.X, U=U, sphere_c=Cs, sphere_v=V, cells=np.array(a.cells))
print(dict(world=a.world, seconds=time.time()-t0, final_v=V[-1].tolist(), impulse=STR['m']*(V[-1][2]-STR['v0'][2])))
