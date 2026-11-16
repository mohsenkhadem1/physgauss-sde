"""Step 2: hidden-command variants of Figure-1 S2 with the patch-role model.

Variants (episodes given to the model):
  oracle  : D0 (central press, known) + D1 (press at x=+.05, KNOWN command)       -> configs 0F0S only
  hidden  : D0 known + D1 with the command HIDDEN (role/location/history inferred)  -> all (nF,nS), nF+nS<=K
  d1only  : D1 hidden only (material must come from gravity + the hidden episode)
Amplitudes: full = Fig-1 truths/RGB tracks (30 N); low = step-2 truths at 10 N (truth.py, obs.py).
Noise: disc = white sigma_w (inferred) + Matern discrepancy sigma_d = 0.15 mm, l_d = 0.05 s (pre-declared,
COMSOL-check level); white = sigma_w only.
Targets: D2 (press at x=-.07, known command, never observed) at the 27 Fig-1 probes (Fig-1 score), and the
hidden D1 response at the 27 probes (hidden target).  Optional executed calibrated probe (--probe-stage).
"""
from __future__ import annotations
import argparse, hashlib, json, math, sys, time
from pathlib import Path
import numpy as np
import torch
from scipy.special import logsumexp
from scripts.joint_field_identification.continuous import SMCConfig, run_smc
from scripts.joint_field_identification.hierarchy_surface import gaussian_mixture_quantiles, discrete_quantiles
from scripts.patch_roles_v02 import model as M, bank as BK
sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'scripts/fig1_v01')); sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'scripts/stem_trial_v01'))
import block as B

ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT/'runs/patch-roles-v02'
ROLE_PRIOR = dict(off=.5, force=.25, support=.25)
AMPS = dict(full=dict(scale=1.0, tag=''), low=dict(scale=1/3, tag='-a0.333333'))
SIGMA_D = dict(disc=1.5e-4, white=0.0)
LEN_D = 0.05
DELTA_FRACTION = 0.2
NEAR = 0.02           # force patch "at the press" if its centre is within 20 mm of the window centre
GRID_X = [-0.09, -0.06, -0.03, 0.0, 0.03, 0.06, 0.09]       # fixed candidate calibrated presses
PROBE_X = [x for x in BK.PRESS_X if x != -0.07]           # all bank presses except the held-out D2 site
C_PROBE = 0.05


def configs(K, variant):
    if variant == 'oracle':
        return [(0, 0)]
    if variant == 'oracleK':          # commanded loads known, extra support patches still allowed (spurious-role control)
        return [(0, nS) for nS in range(K+1)]
    return [(nF, nS) for nF in range(K+1) for nS in range(K+1-nF)]


def config_logprior(K, cfgs):
    out = {}
    for nF, nS in cfgs:
        nO = K-nF-nS
        out[(nF, nS)] = (math.lgamma(K+1)-math.lgamma(nF+1)-math.lgamma(nS+1)-math.lgamma(nO+1)
                         + nF*math.log(ROLE_PRIOR['force'])+nS*math.log(ROLE_PRIOR['support'])+nO*math.log(ROLE_PRIOR['off']))
    z = logsumexp(list(out.values()))
    return {k: v-z for k, v in out.items()}


def load_case(amp, world, probes=()):
    a = AMPS[amp]; tg = a['tag']; F = B.FORCE*a['scale']
    if amp == 'full':
        obs = lambda ep: ROOT/f'runs/fig1-v01/block/rgb/obs-{ep}-{world}.npz'; tru = lambda ep: ROOT/f'runs/fig1-v01/block/truth/{ep}-{world}.npz'
    else:
        obs = lambda ep: RUNS/f'rgb/obs-{ep}-{world}{tg}.npz'; tru = lambda ep: RUNS/f'truth/{ep}-{world}{tg}.npz'
    case = dict(amp=amp, world=world, force=F, y={}, truth={})
    for ep in ['D0', 'D1']+[f'P{x:+.2f}' for x in probes]:
        o = obs(ep) if not ep.startswith('P') else RUNS/f'rgb/obs-{ep}-{world}-a{a["scale"]:g}.npz'
        y = np.load(o)['y']-B.MARKERS[None]
        case['y'][ep] = y[1:].reshape(24, 60)
    m = B.mesh(B.TRUTH_CELLS); Hp = m.interpolation(B.PROBES).toarray()
    for ep in ('D1', 'D2'):
        z = np.load(tru(ep)); case['truth'][ep] = np.einsum('pn,fnj->fpj', Hp, z['U'])      # (25,27,3)
    return case


def episodes(variant, case, probes=()):
    F = case['force']; P = lambda x: M.Bank.__dict__  # noqa
    pidx = {x: 1+i for i, x in enumerate(BK.PRESS_X)}
    grav = ('step', 0, 1.0)
    eps = []
    if variant in ('oracle', 'oracleK', 'hidden'):
        eps.append(dict(name='D0', y=case['y']['D0'], known=[grav, ('ramp', pidx[0.0], F)]))
    if variant.startswith('oracle'):
        eps.append(dict(name='D1', y=case['y']['D1'], known=[grav, ('ramp', pidx[0.05], F)]))
    else:
        eps.append(dict(name='D1', y=case['y']['D1'], known=[grav], hidden=True))
    for x in probes:
        eps.append(dict(name=f'P{x:+.2f}', y=case['y'][f'P{x:+.2f}'], known=[grav, ('ramp', pidx[x], F)]))
    eps.append(dict(name='D2', y=None, known=[grav, ('ramp', pidx[-0.07], F)]))
    for x in PROBE_X:
        if x not in probes:
            eps.append(dict(name=f'C{x:+.2f}', y=None, known=[grav, ('ramp', pidx[x], F)]))
    return eps


def data_hash(eps, extra=''):
    h = hashlib.sha256(extra.encode())
    for e in eps:
        if e.get('y') is not None:
            h.update(e['name'].encode()); h.update(np.ascontiguousarray(e['y']).tobytes()); h.update(str(e['known']).encode()); h.update(str(e.get('hidden')).encode())
    return h.hexdigest()[:16]


def run_config(pm, nF, nS, seed, path, particles, moves, tid):
    names, bounds = pm.layout(nF, nS)
    tid = f'{tid}-{nF}-{nS}-{particles}-{moves}-{seed}'
    if path.exists():
        z = np.load(path, allow_pickle=False)
        if str(z['tid']) == tid:
            return dict(theta=z['theta'], ll=z['ll'], logZ=float(z['logZ']), seconds=float(z['seconds']), stages=int(z['stages']),
                        evals=int(z['evals']), names=names)
    t0 = time.time()
    cfg = SMCConfig(particles=particles, moves=moves, batch_size=particles, maximum_stages=300)
    res = run_smc(lambda th: pm.loglik(th, nF, nS), bounds=bounds, config=cfg, seed=seed, target_id=tid)
    logZ = res.log_evidence+math.lgamma(nF+1)+math.lgamma(nS+1)
    sec = time.time()-t0; stages = sum(s.get('kind') == 'tempering' for s in res.stages)
    np.savez(path, theta=res.theta, ll=res.log_likelihood, logZ=logZ, seconds=sec, stages=stages, evals=res.evaluation_count, tid=tid)
    print(f'  config {nF}F{nS}S logZ={logZ:.2f} stages={stages} evals={res.evaluation_count} {sec:.0f}s', flush=True)
    return dict(theta=res.theta, ll=res.log_likelihood, logZ=logZ, seconds=sec, stages=stages, evals=res.evaluation_count, names=names)


# ------------------------------------------------------------------ per-particle quantities

def extras(pm, theta, nF, nS, batch=64):
    rows = list(range(141)); out = {}
    hid = [e for e in pm.episodes if e.get('hidden')]
    for i in range(0, len(theta), batch):
        th = theta[i:i+batch]; Bn = len(th)
        with torch.no_grad():
            op = pm.operators(th, nF, nS, rows)
            opm = dict(op); opm['G'] = op['G'][:, :, pm.rows] if op['G'] is not None else None
            for e in pm.episodes:
                opm['d_'+e['name']] = op['d_'+e['name']][:, :, pm.rows]
            ll, cond = pm.marginal(th, nF, nS, opm, conditionals=True)
            r = dict(ll=ll, d2=op['d_D2'][:, :, 60:], v=pm.noise_var(th))
            r['cand'] = torch.stack([op[f'd_C{x:+.2f}'][:, :, pm.rows] for x in PROBE_X if f'd_C{x:+.2f}' in op], 1)
            if hid and nF:
                mu, S = cond; G = op['G'][:, :, 60:]                                      # (B,24,81,nA)
                r['h1_mean'] = op['d_D1'][:, :, 60:]+torch.einsum('bgrq,bq->bgr', G, mu)
                r['h1_var'] = torch.einsum('bgrq,bqs,bgrs->bgr', G, S, G)
                mu4 = mu.view(Bn, nF, 3, M.NH); S6 = S.view(Bn, nF, 3, M.NH, nF, 3, M.NH)
                r['res_mean'] = mu4.sum(1)                                                # (B,3,24)
                r['res_var'] = torch.stack([torch.diagonal(S6[:, :, e, :, :, e, :].sum((1, 3)), dim1=1, dim2=2) for e in range(3)], 1)
            else:
                r['h1_mean'] = op['d_D1'][:, :, 60:]; r['h1_var'] = torch.zeros_like(r['h1_mean'])
                r['res_mean'] = torch.zeros(Bn, 3, M.NH, dtype=torch.float64, device=pm.dev); r['res_var'] = torch.zeros_like(r['res_mean'])
        for k, v in r.items():
            out.setdefault(k, []).append(v.cpu().numpy())
    return {k: np.concatenate(v) for k, v in out.items()}


def disc_draws(n, sigma_d, rng, shape=(81,)):
    """Matern discrepancy draws over frames 1..24 for `shape` independent coordinates -> (n,24,*shape)."""
    if sigma_d == 0:
        return np.zeros((n, 24)+shape)
    L = np.linalg.cholesky(M.matern(M.TF[1:], LEN_D)+1e-10*np.eye(24))
    return sigma_d*np.einsum('gh,nh...->ng...', L, rng.standard_normal((n, 24)+shape))


def target_metrics(w, q, var, truth, sigma_d, rng, name, n_mc=4000):
    """q (N,24,81) per-particle means, var (N,24,81) per-particle Gaussian variance (0 if deterministic), truth (24,81).
    Predictive of the TRUE response = particle mixture + discrepancy (if sigma_d>0)."""
    N = len(w); qf = q.reshape(N, -1); vf = var.reshape(N, -1)+sigma_d**2; tf = truth.reshape(-1)
    mean = w@qf
    lo, hi = gaussian_mixture_quantiles(w, qf, vf+1e-30) if vf.max() > 0 else discrete_quantiles(w, qf)
    tot_var = w@(vf+(qf-mean)**2)
    # Fig-1 score: sqrt(mean over 25 frames x 27 probes of |3D error|^2); frame 0 is exactly zero for all
    err = (mean-tf).reshape(24, 27, 3); rms = float(np.sqrt((err**2).sum(-1).sum()/(25*27))*1e3)
    sig = float(np.sqrt((mean.reshape(24, 27, 3)**2).sum(-1).sum()/(25*27))*1e3); delta = DELTA_FRACTION*sig
    idx = rng.choice(N, size=n_mc, p=w)
    draw = qf[idx]+np.sqrt(var.reshape(N, -1)[idx])*rng.standard_normal((n_mc, qf.shape[1]))+disc_draws(n_mc, sigma_d, rng, (q.shape[2],)).reshape(n_mc, -1)
    d_rms = np.sqrt(((draw-mean).reshape(n_mc, 24, 27, 3)**2).sum(-1).sum((1, 2))/(25*27))*1e3
    return dict(target=name, rmse_mm=rms, truth_rms_mm=float(np.sqrt((truth.reshape(24, 27, 3)**2).sum(-1).sum()/(25*27))*1e3),
                pred_sd_mm=float(np.sqrt(tot_var.mean())*1e3), cover90=float(np.mean((tf >= lo) & (tf <= hi))),
                z_rms=float(np.sqrt(np.mean((mean-tf)**2/tot_var))), delta_mm=delta, risk=float(np.mean(d_rms > delta)),
                exceeded=bool(rms > delta), pred_rms_err_median_mm=float(np.median(d_rms)))


def analyze(pm, case, runs, K, variant, sigma_d, label, seed):
    rng = np.random.default_rng(1000+seed)
    cfgs = list(runs); lp = config_logprior(K, cfgs) if variant != 'oracle' else {(0, 0): 0.}
    orc = variant.startswith('oracle')
    logpost = {c: lp[c]+runs[c]['logZ'] for c in cfgs}; zz = logsumexp(list(logpost.values()))
    post = {c: float(np.exp(v-zz)) for c, v in logpost.items()}
    keep = [c for c in cfgs if post[c] > 1e-4]
    parts = {c: extras(pm, runs[c]['theta'], *c) for c in keep}
    w = np.concatenate([np.full(len(runs[c]['theta']), post[c]/len(runs[c]['theta'])) for c in keep]); w /= w.sum()
    cat = lambda k: np.concatenate([parts[c][k] for c in keep])
    th_all = [(c, runs[c]['theta']) for c in keep]
    res = dict(label=label, config_posterior={f'{a}F{b}S': p for (a, b), p in post.items()},
               config_log_evidence={f'{a}F{b}S': runs[c]['logZ'] for c in cfgs for (a, b) in [c]},
               true_config='0F0S' if orc else '1F0S', map_config='%dF%dS' % max(post, key=post.get))
    res['true_config_mass'] = post.get((0, 0) if orc else (1, 0), 0.)
    # ---- force role / location
    true_c = np.array([0.05, B.TOP, 0.0])
    if not orc:
        p_near = 0.; e_nf = 0.; spur = 0.; locs = []
        for c in keep:
            nF, nS = c; F, S, _ = pm.split(runs[c]['theta'], nF, nS)
            e_nf += post[c]*nF; spur += post[c]*nS
            if nF:
                cent = M.surface_point(F[..., 0], F[..., 1])[0]                       # (N,nF,3)
                d = np.linalg.norm(cent-true_c, axis=-1)
                p_near += post[c]*np.mean(d.min(1) < NEAR)
                res.setdefault('force_loc_err_mm_by_config', {})[f'{nF}F{nS}S'] = float(np.mean(d.min(1))*1e3)
                if c == (1, 0):
                    locs = cent[:, 0]
        res['p_force_patch_at_press'] = p_near; res['expected_n_force'] = e_nf; res['expected_n_support'] = spur
        if len(locs):
            res['force_loc_1F0S'] = dict(mean_mm=(locs.mean(0)*1e3).tolist(), err_of_mean_mm=float(np.linalg.norm(locs.mean(0)-true_c)*1e3),
                                         mean_err_mm=float(np.mean(np.linalg.norm(locs-true_c, axis=1))*1e3),
                                         width_mm=float(np.exp(runs[(1, 0)]['theta'][:, 8]).mean()*1e3))
        # MAP-config location (whatever the config)
        cm = max(post, key=post.get)
        if cm[0]:
            F, _, _ = pm.split(runs[cm]['theta'], *cm); cent = M.surface_point(F[..., 0], F[..., 1])[0]
            d = np.linalg.norm(cent-true_c, axis=-1).min(1)
            res['force_loc_map_config_mean_err_mm'] = float(d.mean()*1e3)
            res['map_force_centres_mm'] = (np.median(cent, 0)*1e3).tolist()
        # resultant force history (frames 1..24, 3 comps) vs truth [0, -F ramp, 0]
        rm = cat('res_mean').reshape(len(w), -1); rv = cat('res_var').reshape(len(w), -1)
        ramp = .5*(1-np.cos(np.pi*np.clip(M.TF[1:]/M.RAMP, 0, 1)))
        tr = np.zeros((3, 24)); tr[1] = -case['force']*ramp
        lo, hi = gaussian_mixture_quantiles(w, rm, rv+1e-30)
        res['force_history'] = dict(cover90=float(np.mean((tr.ravel() >= lo) & (tr.ravel() <= hi))),
                                    cover90_y=float(np.mean((tr[1] >= lo.reshape(3, 24)[1]) & (tr[1] <= hi.reshape(3, 24)[1]))),
                                    rmse_N=float(np.sqrt(np.mean((w@rm-tr.ravel())**2))),
                                    final_Fy_mean=float((w@rm).reshape(3, 24)[1, -1]), final_Fy_90=[float(lo.reshape(3, 24)[1, -1]), float(hi.reshape(3, 24)[1, -1])],
                                    final_Fy_true=float(tr[1, -1]))
    # ---- material
    th = np.concatenate([t[:, :5] for _, t in th_all]); mi = pm.material_index(th)
    mats = np.array([(np.nan if x is None else x, k, nu) for x, k, nu in pm.b.mats])[mi]
    xt = B.WORLDS[case['world']]
    lE = th[:, 0]; lo, hi = discrete_quantiles(w, lE)[:, 0]
    res['material'] = dict(logE_mean=float(w@lE), logE_true=math.log(B.E_SOFT), logE_err=float(abs(w@lE-math.log(B.E_SOFT))),
                           logE_cover90=bool(lo <= math.log(B.E_SOFT) <= hi), logE_90=[float(lo), float(hi)],
                           p_xc_true=float(w@np.isclose(mats[:, 0], xt)), p_contrast8=float(w@(mats[:, 1] == 8.)),
                           p_contrast_4_or_8=float(w@np.isin(mats[:, 1], [4., 8.])), p_nu_true=float(w@(mats[:, 2] == B.NU)),
                           p_no_inclusion=float(w@np.isnan(mats[:, 0])),
                           sigma_w_mm_mean=float(w@np.exp(th[:, 4])*1e3))
    i = int(np.argmax(w)); res['material']['map'] = dict(xc=None if np.isnan(mats[i, 0]) else float(mats[i, 0]), contrast=float(mats[i, 1]),
                                                       nu=float(mats[i, 2]), E=float(np.exp(lE[i])))
    # ---- predictions
    d2t = case['truth']['D2'][1:].reshape(24, 81); d1t = case['truth']['D1'][1:].reshape(24, 81)
    res['D2'] = target_metrics(w, cat('d2'), np.zeros_like(cat('d2')), d2t, sigma_d, rng, 'D2 press x=-0.07 (known command)')
    res['D1_hidden'] = target_metrics(w, cat('h1_mean'), cat('h1_var'), d1t, sigma_d, rng, 'D1 probes (hidden-command response)')
    near = None
    cm = max(post, key=post.get)
    if not orc and cm[0]:
        F, _, _ = pm.split(runs[cm]['theta'], *cm); cent = np.median(M.surface_point(F[..., 0], F[..., 1])[0], 0)
        near = [min(PROBE_X, key=lambda x: abs(x-c[0])) for c in cent if abs(c[1]-B.TOP) < 1e-9]
    if not orc and cm[1]:            # also a press near each MAP support patch on the top face (structure check)
        _, S, _ = pm.split(runs[cm]['theta'], *cm); cs = np.median(M.surface_point(S[..., 0], S[..., 1])[0], 0)
        near = (near or [])+[min(PROBE_X, key=lambda x: abs(x-c[0])) for c in cs if abs(c[1]-B.TOP) < 1e-9]
    res['design'] = design(pm, w, cat, sigma_d, rng, sorted(set(GRID_X) | set(near or [])))
    res['design']['near_map_force_candidates'] = near
    if cm[1]:
        _, S, _ = pm.split(runs[cm]['theta'], *cm); cs = M.surface_point(S[..., 0], S[..., 1])
        res['map_support'] = dict(centres_mm=(np.median(cs[0], 0)*1e3).tolist(), faces=[str(f) for f in cs[1][0]],
                                  width_mm=(np.exp(np.median(S[..., 2], 0))*1e3).tolist(), k=np.exp(np.median(S[..., 3], 0)).tolist())
    return res


def design(pm, w, cat, sigma_d, rng, cands, n_y=128):
    """Preposterior D2 threshold risk for each candidate calibrated press (observed by the markers, known command).
    Observation noise per particle: white sigma_w + discrepancy (time-whitened exactly)."""
    dev = pm.dev; tt = lambda a: torch.as_tensor(a, dtype=torch.float64, device=dev)
    N = len(w); W = tt(w); q = tt(cat('d2').reshape(N, -1)); v = tt(cat('v'))          # (N,24)
    allx = [x for x in PROBE_X if any(e['name'] == f'C{x:+.2f}' for e in pm.episodes)]
    cand = cat('cand'); names = [x for x in cands if x in allx]

    def risk(Wp):
        qh = Wp@q; sig = torch.sqrt((qh.view(-1, 24, 27, 3)**2).sum(-1).sum((1, 2))/(25*27))
        dist = torch.sqrt(((q[None]-qh[:, None]).view(Wp.shape[0], N, 24, 27, 3)**2).sum(-1).sum((2, 3))/(25*27))
        return (Wp*(dist > DELTA_FRACTION*sig[:, None])).sum(1)

    rows = [dict(action='proceed', J=float(risk(W[None])[0]), expected_risk=float(risk(W[None])[0]))]
    idx = rng.choice(N, size=n_y, p=w)
    for x in names:
        P = tt(cand[:, allx.index(x)]); Pt = torch.einsum('gi,ngr->nir', pm.Ut, P)                   # whitened (N,24,R)
        z = tt(rng.standard_normal((n_y, 24, pm.R)))
        yt = Pt[idx]+v[idx].sqrt()[..., None]*z
        rs = []
        for y0 in range(0, n_y, 8):
            yy = yt[y0:y0+8]
            ll = -.5*(((yy[:, None]-Pt[None])**2/v[None, :, :, None]).sum((2, 3))+pm.R*torch.log(v).sum(1)[None])+torch.log(W)[None]
            rs.append(risk(torch.softmax(ll, 1)))
        r = float(torch.cat(rs).mean())
        rows.append(dict(action=f'press x={x:+.2f}', x=x, cost=C_PROBE, expected_risk=r, J=C_PROBE+r))
    best = min(rows, key=lambda r: r['J']); probes = [r for r in rows if 'x' in r]
    return dict(rows=rows, decision=best['action'], best_probe=min(probes, key=lambda r: r['expected_risk']) if probes else None)


def outlier_rows(bank, case, a, out):
    """D0-only fit with the commanded D0 load (0F0S, same noise model); markers whose posterior-mean residual RMS
    exceeds 3x the median marker residual are treated as tracking outliers and dropped from every episode."""
    pidx = {x: 1+i for i, x in enumerate(BK.PRESS_X)}
    ep0 = [dict(name='D0', y=case['y']['D0'], known=[('step', 0, 1.0), ('ramp', pidx[0.0], case['force'])])]
    pm0 = M.PatchModel(bank, ep0, sigma_d=SIGMA_D[a.noise], len_d=LEN_D)
    r = run_config(pm0, 0, 0, seed=a.seed*1000+999, path=out/f'robust-D0fit-s{a.seed}.npz', particles=a.particles, moves=a.moves,
                   tid=data_hash(ep0, 'robust'+a.noise))
    with torch.no_grad():
        d = pm0.operators(r['theta'][:256], 0, 0, list(range(60)))['d_D0'].mean(0).cpu().numpy()
    res = (case['y']['D0']-d).reshape(24, 20, 3)
    rms = np.sqrt((res**2).sum(-1).mean(0)); flag = np.flatnonzero(rms > 3*np.median(rms))
    rows = [3*m+k for m in range(20) if m not in flag for k in range(3)]
    print('robust: marker residual rms mm', np.round(rms*1e3, 3), 'dropped', flag.tolist(), flush=True)
    return rows, dict(markers=flag.tolist(), residual_rms_mm=(rms*1e3).tolist())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--amp', default='low'); ap.add_argument('--world', default='Bright'); ap.add_argument('--variant', default='hidden')
    ap.add_argument('--noise', default='disc'); ap.add_argument('--seed', type=int, default=0); ap.add_argument('--K', type=int, default=2)
    ap.add_argument('--particles', type=int, default=1024); ap.add_argument('--moves', type=int, default=10)
    ap.add_argument('--probes', type=float, nargs='*', default=[]); ap.add_argument('--tag', default='')
    ap.add_argument('--robust', action='store_true', help='drop markers flagged as tracking outliers by a D0-only known-command fit')
    a = ap.parse_args()
    torch.set_num_threads(4)
    name = f'{a.amp}-{a.world}-{a.variant}-{a.noise}' + (''.join(f'-P{x:+.2f}' for x in a.probes)) + ('-robust' if a.robust else '') + a.tag
    out = RUNS/'cases'/name; out.mkdir(parents=True, exist_ok=True)
    case = load_case(a.amp, a.world, a.probes)
    eps = episodes(a.variant, case, a.probes)
    need_sup = a.variant != 'oracle'
    bank = M.Bank('cuda', supports=need_sup)
    rows = None; flag = None
    if a.robust:
        rows, flag = outlier_rows(bank, case, a, out)
    pm = M.PatchModel(bank, eps, sigma_d=SIGMA_D[a.noise], len_d=LEN_D, rows_obs=rows)
    tid = data_hash(eps, f'{a.noise}-{SIGMA_D[a.noise]}-{LEN_D}-{pm.SIGMA_F}-{pm.LEN_F}-{rows}')
    t0 = time.time(); runs = {}
    for c in configs(a.K, a.variant):
        runs[c] = run_config(pm, *c, seed=a.seed*1000+17*c[0]+c[1], path=out/f'{c[0]}F{c[1]}S-s{a.seed}.npz',
                             particles=a.particles, moves=a.moves, tid=tid)
    smc = time.time()-t0; t1 = time.time()
    res = analyze(pm, case, runs, a.K, a.variant, SIGMA_D[a.noise], name, a.seed)
    res['robust_dropped_markers'] = flag
    res.update(case=name, seed=a.seed, particles=a.particles, moves=a.moves, K=a.K, sigma_d=SIGMA_D[a.noise], len_d=LEN_D,
               compute=dict(smc_seconds=smc, analysis_seconds=time.time()-t1,
                            per_config={f'{c[0]}F{c[1]}S': dict(seconds=r['seconds'], stages=r['stages'], evals=r['evals']) for c, r in runs.items()}))
    (out/f'result-s{a.seed}.json').write_text(json.dumps(res, indent=1, default=float))
    print(json.dumps({k: v for k, v in res.items() if k != 'design'}, indent=1, default=float)[:4000])
    print('design', json.dumps(res['design'], default=float))


if __name__ == '__main__':
    main()
