"""Posterior-mean displacement FIELDS of our method on the reference surface mesh (for rendering).
Usage: ours_fields.py SENSOR WORLD EPISODE  (EPISODE in D2, D2b). Uses the saved round-2 posterior
weights; recomputes full linear fields only for (c, nu) cells carrying posterior mass.
Writes runs/stem-trial-v01/ours-fields/SENSOR-WORLD-EPISODE.npz with U[stage] (F, n_truth_nodes, 3)."""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import fem
from ours import OUT, C_GRID, NU_GRID, RHO_GRID, P, DT, FPS, POKE, HELD, R
from ours_round2 import LOADS

MASS = 1e-4


def fields(c, nu):
    mesh = fem.Mesh(tuple(P['ours']['mesh_cells'])); model = fem.Elastic(mesh, c, nu, 1.0)
    save = round(1/(FPS*DT)); ramp = lambda f: (lambda t: fem.smooth_ramp(t, .25)*f)
    g = mesh.gravity(1.0)
    fz = mesh.body_force_region(HELD[0], HELD[1], [0, 0, 1.0]); fy = mesh.body_force_region(HELD[0], HELD[1], [0, 1.0, 0])
    Ug, _ = fem.newmark(model, lambda t: g, .75, DT, save, nonlinear=False)
    Uz, _ = fem.newmark(model, ramp(fz), .75, DT, save, nonlinear=False)
    Uy, _ = fem.newmark(model, ramp(fy), .75, DT, save, nonlinear=False)
    return mesh, Ug, Uz, Uy


def main(sensor, world, episode):
    z = np.load(OUT/f'round2-{sensor}-{world}.npz'); F = LOADS[episode]
    truth_mesh = fem.Mesh(tuple(P['truth_numerics']['mesh_cells']))
    out = {}; cache = {}
    for stage in ('D0', 'D0+D1'):
        w = z[f'w_{stage}']; W = w.sum(2)
        cells = [tuple(i) for i in np.argwhere(W > MASS)]
        kept = sum(W[i] for i in cells)
        U = 0
        for ci, ni in cells:
            if (ci, ni) not in cache:
                mesh, Ug, Uz, Uy = fields(C_GRID[ci], NU_GRID[ni])
                I = mesh.interpolation(truth_mesh.X)
                cache[ci, ni] = [np.einsum('mn,fnj->fmj', I.toarray(), A) for A in (Ug, Uz, Uy)]
            Ug, Uz, Uy = cache[ci, ni]
            inv_rho = (w[ci, ni]*(1/RHO_GRID)).sum()/W[ci, ni]
            U = U+W[ci, ni]/kept*(Ug+inv_rho*(F[2]*Uz+F[1]*Uy))
        out[stage] = U; out[f'mass_kept_{stage}'] = kept; out[f'cells_{stage}'] = len(cells)
        print(sensor, world, episode, stage, 'cells', len(cells), 'mass kept', round(float(kept), 6), flush=True)
    d = R/'runs/stem-trial-v01/ours-fields'; d.mkdir(exist_ok=True)
    np.savez_compressed(d/f'{sensor}-{world}-{episode}.npz', X=truth_mesh.X, **{k.replace('+', 'p'): v for k, v in out.items()})


if __name__ == '__main__':
    main(*sys.argv[1:4])
