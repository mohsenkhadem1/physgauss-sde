"""Our striker (D3) forecast: posterior-weighted contact simulations (linear T10 FEM, coarse mesh, same
contact law and striker as every method). Hypotheses with posterior mass > 1e-3 are simulated; the
retained mass is renormalised and reported. Usage: ours_striker.py SENSOR WORLD STAGE  (STAGE D0|D0+D1)"""
import json, os, sys, time
from multiprocessing import Pool
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import fem, contact as C
from ours import OUT, C_GRID, NU_GRID, RHO_GRID, P, R, STATIONS, station_matrix
SP = json.loads((R/'scripts/fig1_v01/protocol_fig1.json').read_text())
STR = dict(r=0.03, m=0.3, kappa=1e11, c0=[0.17, 0.60, 0.0755], v0=[0, 0, -1.0])
DT = 1/2400; T = 0.75; MASS = 1e-3


def sim(args):
    c, nu, rho = args
    mesh = fem.Mesh(tuple(P['ours']['mesh_cells'])); model = fem.Elastic(mesh, c*rho, nu, rho)
    g = mesh.gravity(rho); s = C.Striker(STR['r'], STR['m'], STR['kappa'], STR['c0'], STR['v0'])
    U, Cs, V, st = C.newmark_striker(model, lambda t: g, s, T, DT, round(1/(48*DT)), nonlinear=False)
    S = station_matrix(mesh)
    return np.einsum('sn,fnj->fsj', S, U), Cs, V


def main(sensor, world, stage):
    w = np.load(OUT/f'round3-{sensor}-{world}.npz')[f'w_{stage}']
    idx = [tuple(i) for i in np.argwhere(w > MASS)]; kept = float(sum(w[i] for i in idx))
    jobs = [(C_GRID[i], NU_GRID[j], RHO_GRID[k]) for i, j, k in idx]
    t0 = time.time()
    with Pool(min(12, len(jobs))) as pool:
        res = pool.map(sim, jobs, chunksize=1)
    ww = np.array([w[i] for i in idx])/kept
    st = np.stack([r[0] for r in res]); cs = np.stack([r[1] for r in res]); vs = np.stack([r[2] for r in res])
    imp = STR['m']*(vs[:, -1, 2]-STR['v0'][2])
    mean_st = np.einsum('h,hfsj->fsj', ww, st); mean_c = np.einsum('h,hfj->fj', ww, cs); mean_v = np.einsum('h,hfj->fj', ww, vs)
    rms = np.sqrt(((st-mean_st)**2).sum(-1).mean((-1, -2)))
    imp_mean = float(ww@imp)
    out = dict(sensor=sensor, world=world, stage=stage, hypotheses=len(jobs), mass_kept=kept, seconds=time.time()-t0,
               impulse_mean_Ns=imp_mean, impulse_sd_Ns=float(np.sqrt(ww@(imp-imp_mean)**2)),
               risk_station_rms_gt_3mm=float(ww@(rms > 3e-3)), risk_impulse_err_gt_0p03=float(ww@(np.abs(imp-imp_mean) > 0.03)))
    d = OUT/'striker'; d.mkdir(exist_ok=True); tag = f"{sensor}-{world}-{stage.replace('+', 'p')}"
    np.savez_compressed(d/f'{tag}.npz', station_disp=mean_st, sphere_c=mean_c, sphere_v=mean_v, weights=ww, impulses=imp)
    (d/f'{tag}.json').write_text(json.dumps(out, indent=1)); print(json.dumps(out))


if __name__ == '__main__':
    main(*sys.argv[1:4])
