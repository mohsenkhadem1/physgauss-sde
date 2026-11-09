"""Our method on the stem trial: finite-grid posterior over (c=E/rho, nu, rho).

Forward model: linear T10 elastodynamics on the coarse (22,4,4) mesh, unit
density. With rho divided out, M1 u'' + K(E=c, nu) u = g_1 + (F/rho) s(t), so
u = u_g(c,nu) + (F/rho) G(c,nu): one solve per (c,nu) and an exact scale in
rho. Evidence: supplied 3D marker tracks with Gaussian noise (sd 1 mm).
Usage: ours.py bank        -> unit-response bank (no truth access)
       ours.py infer WORLD -> posterior + sealed D2 predictions from D0, D0+D1
"""
import hashlib, json, sys, time
from multiprocessing import Pool
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import fem

R = Path(__file__).resolve().parents[2]
HERE = Path(__file__).parent
P = json.loads((HERE/'protocol.json').read_text())
import os
OFFGRID = os.environ.get('STEM_OFFGRID') == '1'
# Off-grid sensitivity variant (added before sealing): grids shifted by half a log step so the true
# c=4000 and rho in {500,1000,2000} are not hypotheses; the frozen primary grid contains them.
_h = 2**(1/24) if OFFGRID else 1.0  # grid log step is 2**(1/12)
OUT = R/('runs/stem-trial-v01/ours-offgrid' if OFFGRID else 'runs/stem-trial-v01/ours')
C_GRID = np.geomspace(1000, 16000, 49)*_h
RHO_GRID = np.geomspace(250, 4000, 49)*_h
NU_GRID = np.array(P['ours']['prior']['nu'])
DT = 1/480
FPS = 48
MARKERS = np.array([[x, .64, 0.] for x in (-.1, 0., .1, .2)]+[[x, .60, .04] for x in (-.1, 0., .1, .2)])
STATIONS = [(-.11, -.09), (-.01, .01), (.09, .11), (.18, .2001)]
POKE = (-.015, .015, 40.); HELD = (.085, .115, 20.)


def station_matrix(mesh):
    return np.stack([mesh.region_weights(lo, hi)[0] for lo, hi in STATIONS])


def _solve(args):
    c, nu = args
    mesh = fem.Mesh(tuple(P['ours']['mesh_cells']))
    model = fem.Elastic(mesh, c, nu, 1.0)
    H = mesh.interpolation(MARKERS); S = station_matrix(mesh)
    g = mesh.gravity(1.0)
    fp = mesh.body_force_region(POKE[0], POKE[1], [0, 0, 1.0])
    fh = mesh.body_force_region(HELD[0], HELD[1], [0, 0, 1.0])
    save = round(1/(FPS*DT)); T = 0.75
    Ug, _ = fem.newmark(model, lambda t: g, T, DT, save, nonlinear=False)
    Up, _ = fem.newmark(model, lambda t: fem.smooth_ramp(t, .25)*fp, T, DT, save, nonlinear=False)
    Uh, _ = fem.newmark(model, lambda t: fem.smooth_ramp(t, .25)*fh, T, DT, save, nonlinear=False)
    obs = lambda U: np.einsum('mn,fnj->fmj', H.toarray(), U)
    st = lambda U: np.einsum('sn,fnj->fsj', S, U)
    return dict(mark_g=obs(Ug), mark_p=obs(Up), st_g=st(Ug), st_h=st(Uh), st_p=st(Up))


def bank():
    OUT.mkdir(parents=True, exist_ok=True)
    jobs = [(c, nu) for c in C_GRID for nu in NU_GRID]
    t0 = time.time()
    with Pool(12) as pool:
        res = pool.map(_solve, jobs, chunksize=2)
    arr = {k: np.stack([r[k] for r in res]).reshape(len(C_GRID), len(NU_GRID), *res[0][k].shape) for k in res[0]}
    mesh = fem.Mesh(tuple(P['ours']['mesh_cells']))
    np.savez_compressed(OUT/'bank.npz', c=C_GRID, nu=NU_GRID, markers=MARKERS, marker_ref=MARKERS,
                        station_ref=station_matrix(mesh)@mesh.X, **arr)
    meta = dict(seconds=time.time()-t0, shapes={k: list(v.shape) for k, v in arr.items()},
                fem_sha256=hashlib.sha256((HERE/'fem.py').read_bytes()).hexdigest(),
                ours_sha256=hashlib.sha256((HERE/'ours.py').read_bytes()).hexdigest())
    (OUT/'bank.json').write_text(json.dumps(meta, indent=1)); print(meta)


def observe(episode, world):
    """Noisy marker tracks from truth (the only truth access); seeded per episode/world."""
    src = 'W1000' if episode == 'D0' else world  # D0 is shared; identical across worlds (verified 4e-17 m)
    z = np.load(R/f'runs/stem-trial-v01/truth/{episode}-{src}.npz')
    mesh = fem.Mesh(tuple(int(c) for c in z['cells']))
    H = mesh.interpolation(MARKERS).toarray()
    clean = np.einsum('mn,fnj->fmj', H, z['U'])
    seed = P['ours']['noise_seed']+{'D0': 0, 'D1': 1}[episode]*100+int(world[1:])
    rng = np.random.default_rng(seed)
    return clean+P['ours']['noise_sd_m']*rng.standard_normal(clean.shape)


def infer(world):
    b = dict(np.load(OUT/'bank.npz'))
    sd = P['ours']['noise_sd_m']
    y0 = observe('D0', world); y1 = observe('D1', world)
    n0 = y0.shape[0]
    # log-likelihoods on the (c, nu, rho) grid; prior uniform over the log-spaced grid
    r0 = ((b['mark_g'][:, :, :n0]-y0[None, None])**2).sum((2, 3, 4))  # (C,NU)
    ll0 = -r0/(2*sd**2)
    a = POKE[2]/RHO_GRID  # (RHO,)
    pred1 = b['mark_g'][:, :, None]+a[None, None, :, None, None, None]*b['mark_p'][:, :, None]
    r1 = ((pred1-y1[None, None, None])**2).sum((3, 4, 5))
    ll1 = -r1/(2*sd**2)
    out = {}
    for stage, ll in [('D0', np.broadcast_to(ll0[:, :, None], (len(C_GRID), len(NU_GRID), len(RHO_GRID)))),
                      ('D0+D1', ll0[:, :, None]+ll1)]:
        w = np.exp(ll-ll.max()); w /= w.sum()
        traj = b['st_g'][:, :, None]+(HELD[2]/RHO_GRID)[None, None, :, None, None, None]*b['st_h'][:, :, None]  # (C,NU,RHO,F,S,3)
        mean = np.einsum('cnr,cnrfsj->fsj', w, traj)
        rms = np.sqrt(((traj-mean)**2).sum(-1).mean((-1, -2)))
        risk = float((w*(rms > 3e-3)).sum())
        mi = np.unravel_index(np.argmax(w), w.shape)
        marg = lambda ax: w.sum(tuple(i for i in range(3) if i != ax))
        q = lambda grid, p: [float(grid[np.searchsorted(np.cumsum(p), x)]) for x in (.05, .5, .95)]
        out[stage] = dict(prediction_station_disp=mean, map_station_disp=traj[mi], risk_rms_gt_3mm=risk,
                          expected_rms_m=float((w*rms).sum()),
                          posterior=dict(c_q05_50_95=q(C_GRID, marg(0)), nu_q05_50_95=q(NU_GRID, marg(1)),
                                         rho_q05_50_95=q(RHO_GRID, marg(2)),
                                         E_map=float(C_GRID[mi[0]]*RHO_GRID[mi[2]]), c_map=float(C_GRID[mi[0]]),
                                         nu_map=float(NU_GRID[mi[1]]), rho_map=float(RHO_GRID[mi[2]])),
                          min_marker_rms_mm=float(np.sqrt((r0.min() if stage == 'D0' else (r0[:, :, None]+r1).min())/(y0.size+(0 if stage == 'D0' else y1.size)))*1e3))
    np.savez_compressed(OUT/f'pred-{world}.npz', station_ref=b['station_ref'],
                        **{f'{k}_{s}': v[k] for s, v in out.items() for k in ('prediction_station_disp', 'map_station_disp')})
    summ = {s: {k: v for k, v in d.items() if not isinstance(v, np.ndarray)} for s, d in out.items()}
    (OUT/f'pred-{world}.json').write_text(json.dumps(summ, indent=1)); print(world, json.dumps(summ, indent=1))


if __name__ == '__main__':
    if sys.argv[1] == 'bank':
        bank()
    else:
        infer(sys.argv[2])
