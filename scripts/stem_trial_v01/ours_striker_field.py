"""Posterior-mean full displacement field (coarse mesh) and sphere path of our striker forecast, for shape figures.
Same hypotheses, weights and simulator as ours_striker.py. Usage: ours_striker_field.py SENSOR WORLD STAGE"""
import sys
from multiprocessing import Pool
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import fem, contact as C
from ours import OUT, C_GRID, NU_GRID, RHO_GRID, P
from ours_striker import STR, DT, T, MASS


def sim(args):
    c, nu, rho = args
    mesh = fem.Mesh(tuple(P['ours']['mesh_cells'])); model = fem.Elastic(mesh, c*rho, nu, rho)
    s = C.Striker(STR['r'], STR['m'], STR['kappa'], STR['c0'], STR['v0']); g = mesh.gravity(rho)
    U, Cs, V, _ = C.newmark_striker(model, lambda t: g, s, T, DT, round(1/(48*DT)), nonlinear=False)
    return U, Cs


def main(sensor, world, stage):
    w = np.load(OUT/f'round3-{sensor}-{world}.npz')[f'w_{stage}']
    idx = [tuple(i) for i in np.argwhere(w > MASS)]; kept = float(sum(w[i] for i in idx))
    with Pool(min(12, len(idx))) as pool:
        res = pool.map(sim, [(C_GRID[i], NU_GRID[j], RHO_GRID[k]) for i, j, k in idx], chunksize=1)
    ww = np.array([w[i] for i in idx])/kept
    U = np.einsum('h,hfnj->fnj', ww, np.stack([r[0] for r in res])); Cs = np.einsum('h,hfj->fj', ww, np.stack([r[1] for r in res]))
    np.savez_compressed(OUT/f"striker/field-{sensor}-{world}-{stage.replace('+', 'p')}.npz", X=fem.Mesh(tuple(P['ours']['mesh_cells'])).X, U=U, sphere_c=Cs, mass_kept=kept)
    print(world, stage, 'hyp', len(idx), 'kept', kept)


if __name__ == '__main__':
    main(*sys.argv[1:4])
