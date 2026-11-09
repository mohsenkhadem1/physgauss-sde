"""Our method on the block: finite posterior over a hidden stiff region (centre x_c or none, contrast k),
soft modulus E_s and Poisson ratio; linear T10 elastodynamics on a coarse mesh; density 1000 known.
Evidence: RGB-tracked, two-view-triangulated surface markers from D0 (central press) and D1 (calibration).
bank          -> unit responses (gravity, D0, D1, D2 presses) for every hypothesis
infer WORLD   -> posterior, D2 probe-displacement predictions (D0 only, D0+D1), risks"""
import json, sys, time
from multiprocessing import Pool
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent; R = HERE.parents[1]
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(R/'scripts/stem_trial_v01'))
import fem, block as B

OUT = R/'runs/fig1-v01/block/ours'
CELLS = (12, 5, 6)
XC = [None]+list(np.round(np.arange(-0.08, 0.0801, 0.02), 3))
CONTRAST = [2.0, 4.0, 8.0, 16.0]
ES = np.geomspace(2e4, 2e5, 13)  # true 6e4 lies between grid points
NU = [0.3, 0.4, 0.45]
DT = 1/480; DELTA = 3e-3


def hypotheses():
    H = []
    for xc in XC:
        for k in ([1.0] if xc is None else CONTRAST):
            for e in ES:
                for nu in NU:
                    H.append((xc, k, e, nu))
    return H


def _solve(h):
    xc, k, e, nu = h
    m = B.mesh(CELLS); E = B.modulus(m, xc, contrast=k, e_soft=e)
    model = fem.Elastic(m, E, nu, B.RHO)
    save = round(1/(B.EPISODE['fps']*DT)); T = B.EPISODE['duration_s']; ramp = B.EPISODE['ramp_s']
    g = m.gravity(B.RHO)
    out = {}
    Ug, _ = fem.newmark(model, lambda t: g, T, DT, save, nonlinear=False)
    Hm = m.interpolation(B.MARKERS).toarray(); Hp = m.interpolation(B.PROBES).toarray()
    out['mark_g'] = np.einsum('mn,fnj->fmj', Hm, Ug); out['probe_g'] = np.einsum('mn,fnj->fmj', Hp, Ug)
    for ep in ('D0', 'D1', 'D2'):
        b = B.press_box(ep); f = m.body_force_box(b[0], b[1], [0, -B.FORCE, 0])
        U, _ = fem.newmark(model, lambda t: fem.smooth_ramp(t, ramp)*f, T, DT, save, nonlinear=False)
        out[f'mark_{ep}'] = np.einsum('mn,fnj->fmj', Hm, U); out[f'probe_{ep}'] = np.einsum('mn,fnj->fmj', Hp, U)
    return out


def bank():
    H = hypotheses(); OUT.mkdir(parents=True, exist_ok=True); t0 = time.time()
    with Pool(12) as pool:
        res = pool.map(_solve, H, chunksize=4)
    arr = {k: np.stack([r[k] for r in res]) for k in res[0]}
    np.savez_compressed(OUT/'bank.npz', hyp=np.array([(np.nan if x is None else x, k, e, n) for x, k, e, n in H]), **arr)
    (OUT/'bank.json').write_text(json.dumps(dict(hypotheses=len(H), seconds=time.time()-t0, cells=CELLS)))
    print(len(H), time.time()-t0)


def infer(world, sensor='rgb'):
    b = dict(np.load(OUT/'bank.npz')); hyp = b['hyp']
    o0 = np.load(R/f'runs/fig1-v01/block/{sensor}/obs-D0-{world}.npz'); o1 = np.load(R/f'runs/fig1-v01/block/{sensor}/obs-D1-{world}.npz')
    y0 = o0['y']-B.MARKERS[None]; y1 = o1['y']-B.MARKERS[None]
    p0 = b['mark_g']+b['mark_D0']; p1 = b['mark_g']+b['mark_D1']
    r0 = ((p0-y0[None])**2).sum((1, 2, 3)); r1 = ((p1-y1[None])**2).sum((1, 2, 3))
    sd = float(np.sqrt(r0.min()/y0.size))  # plug-in noise level from the best D0 fit
    pred_all = b['probe_g']+b['probe_D2']
    res = dict(world=world, sensor=sensor, noise_sd_m=sd, hypotheses=len(hyp)); arrays = {}
    for stage, ll in (('D0', -r0/(2*sd**2)), ('D0+D1', -(r0+r1)/(2*sd**2))):
        w = np.exp(ll-ll.max()); w /= w.sum(); arrays[f'w_{stage}'] = w
        mean = np.einsum('h,hfpj->fpj', w, pred_all)
        rms = np.sqrt(((pred_all-mean)**2).sum(-1).mean((-1, -2)))
        i = int(np.argmax(w)); xs = hyp[:, 0]
        p_none = float(w[np.isnan(xs)].sum())
        post_x = {str(x): float(w[xs == x].sum()) for x in np.unique(xs[~np.isnan(xs)])}
        res[stage] = dict(risk_rms_gt_3mm=float(w@(rms > DELTA)), map=dict(xc=None if np.isnan(hyp[i, 0]) else float(hyp[i, 0]),
                          contrast=float(hyp[i, 1]), E_soft=float(hyp[i, 2]), nu=float(hyp[i, 3])), p_no_inclusion=p_none,
                          posterior_xc=post_x)
        arrays[f'pred_D2_{stage}'] = mean
    np.savez_compressed(OUT/f'pred-{sensor}-{world}.npz', **arrays)
    (OUT/f'pred-{sensor}-{world}.json').write_text(json.dumps(res, indent=1)); print(json.dumps(res))


if __name__ == '__main__':
    bank() if sys.argv[1] == 'bank' else infer(*sys.argv[2:])
