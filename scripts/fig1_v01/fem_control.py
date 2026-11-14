"""S2 control: a fitted spatially varying FEM (no hypothesis search, no posterior).
Same linear T10 elastodynamics, coarse mesh, density, RGB-tracked markers and known press commands as ours,
but the modulus is a generic 6 (x) x 2 (y) piecewise-constant field plus a global Poisson ratio, fitted by
damped Gauss-Newton (Levenberg-Marquardt) to the marker tracks. The field can represent the inclusion
(approximately: bins are not aligned to it), so this separates "able to represent a stiff region" from
"our inference over where it is".
fit WORLD -> runs/fig1-v01/block/control/pred-fem-{WORLD}.{npz,json}: D2 probe predictions after D0 and after D0+D1."""
import json, sys, time
from multiprocessing import Pool
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent; R = HERE.parents[1]
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(R/'scripts/stem_trial_v01'))
import fem, block as B
from ours_block import CELLS, DT

OUT = R/'runs/fig1-v01/block/control'
XB = np.linspace(B.LO[0], B.HI[0], 7)        # 6 bins of 4 cm
YB = np.array([B.LO[1], 0.57, B.HI[1]+1e-9])  # lower / upper half (not aligned with the true inclusion edge 0.555)
NP = 12
THETA0 = np.r_[np.full(NP, np.log10(3e4)), 0.35]  # neutral start: same E_init as the PAC-NeRF block fit
LO_T = np.r_[np.full(NP, 3.0), 0.2]; HI_T = np.r_[np.full(NP, 6.5), 0.48]
STEP = np.r_[np.full(NP, 0.03), 0.01]
RIDGE = 1e-3  # weak pull of every bin towards the field mean (log10 units), fixed a priori


def field(m, theta):
    c = m.element_centroids(); ix = np.clip(np.searchsorted(XB, c[:, 0])-1, 0, 5); iy = np.clip(np.searchsorted(YB, c[:, 1])-1, 0, 1)
    return 10**theta[:NP][ix*2+iy]


def simulate(args):
    theta, eps = args
    m = B.mesh(CELLS); model = fem.Elastic(m, field(m, theta), float(theta[NP]), B.RHO)
    save = round(1/(B.EPISODE['fps']*DT)); T = B.EPISODE['duration_s']; ramp = B.EPISODE['ramp_s']
    Hm = m.interpolation(B.MARKERS).toarray(); Hp = m.interpolation(B.PROBES).toarray(); g = m.gravity(B.RHO); out = {}
    for ep in eps:
        if ep == 'g':
            f = g; U, _ = fem.newmark(model, lambda t: f, T, DT, save, nonlinear=False)
        else:
            b = B.press_box(ep); f = m.body_force_box(b[0], b[1], [0, -B.FORCE, 0])
            U, _ = fem.newmark(model, lambda t: fem.smooth_ramp(t, ramp)*f, T, DT, save, nonlinear=False)
        out[ep] = (np.einsum('mn,fnj->fmj', Hm, U), np.einsum('mn,fnj->fmj', Hp, U))
    return out


def residual_of(sim, ys):
    return np.concatenate([(sim['g'][0]+sim[ep][0]-y).ravel() for ep, y in ys.items()])


def fit(world, eps, pool, log):
    ys = {ep: np.load(R/f'runs/fig1-v01/block/rgb/obs-{ep}-{world}.npz')['y']-B.MARKERS[None] for ep in eps}
    sim_eps = ('g',)+tuple(eps)
    def obj(th, r):
        reg = np.sqrt(RIDGE)*(th[:NP]-th[:NP].mean())*1e-3  # 1e-3: metres scale of the data residual
        return float(r@r+reg@reg), reg
    th = THETA0.copy(); r = residual_of(simulate((th, sim_eps)), ys); f, _ = obj(th, r); lam = 1e-2
    for it in range(20):
        steps = [th+np.eye(len(th))[k]*STEP[k] for k in range(len(th))]
        sims = pool.map(simulate, [(s, sim_eps) for s in steps])
        J = np.stack([(residual_of(s, ys)-r)/STEP[k] for k, s in enumerate(sims)], 1)
        Jr = np.vstack([J, np.sqrt(RIDGE)*1e-3*np.c_[np.eye(NP)-1/NP, np.zeros((NP, 1))]])
        _, reg = obj(th, r); rr = np.r_[r, reg]
        A = Jr.T@Jr; gvec = Jr.T@rr; improved = False
        for _ in range(6):
            d = -np.linalg.solve(A+lam*np.diag(np.diag(A)+1e-30), gvec)
            tn = np.clip(th+d, LO_T, HI_T); rn = residual_of(simulate((tn, sim_eps)), ys); fn, _ = obj(tn, rn)
            if fn < f:
                th, r, lam, improved = tn, rn, max(lam/3, 1e-6), True
                rel = (f-fn)/f; f = fn; break
            lam *= 4
        log(dict(it=it, rms_mm=float(np.sqrt(r@r/r.size)*1e3), lam=lam, theta=np.round(th, 3).tolist()))
        if not improved or rel < 1e-4:
            break
    return th, float(np.sqrt(r@r/r.size))


def main(world):
    OUT.mkdir(parents=True, exist_ok=True); t0 = time.time(); res = dict(world=world, parametrisation='6x2 log10 E bins + global nu'); arrays = {}
    log = lambda d: print(world, json.dumps(d), flush=True)
    with Pool(13) as pool:
        for stage, eps in (('D0', ('D0',)), ('D0+D1', ('D0', 'D1'))):
            th, rms = fit(world, eps, pool, log)
            sim = simulate((th, ('g', 'D2')))
            arrays[f'pred_D2_{stage}'] = sim['g'][1]+sim['D2'][1]
            res[stage] = dict(theta=th.tolist(), E_bins_Pa=(10**th[:NP]).reshape(6, 2).tolist(), nu=float(th[NP]), fit_rms_m=rms)
    res['seconds'] = time.time()-t0
    np.savez_compressed(OUT/f'pred-fem-{world}.npz', **arrays); (OUT/f'pred-fem-{world}.json').write_text(json.dumps(res, indent=1))
    print(json.dumps(res))


if __name__ == '__main__':
    main(sys.argv[1])
