"""Patch-role recovery for one synthetic scene.

Discrete roles: patches are exchangeable a priori (iid role prior p_off, p_F, p_S and
iid continuous priors), so the posterior over labelled role vectors depends only on
the counts (nF, nS). We enumerate admissible count configurations (nS >= 1, since
without a support the stiffness is singular), run the published adaptive tempered SMC
(continuous.run_smc, unchanged) within each, with x-ordering inside each role to
remove label switching (evidence corrected by log nF! + log nS!), and combine:
    p(nF,nS | y) ∝ Multinomial(nF,nS,nOff; K, p) * Z_(nF,nS).
"""
from __future__ import annotations
import argparse, hashlib, json, math, time
from pathlib import Path
import numpy as np
import torch
from scipy.special import logsumexp, ndtr
from scripts.joint_field_identification.continuous import SMCConfig, run_smc, MarginalGaussianLikelihood
from scripts.joint_field_identification.hierarchy_surface import gaussian_mixture_quantiles, discrete_quantiles
from scripts.joint_field_identification import surface as pub
from scripts.patch_roles_v01 import model as M, scenes
from scripts.patch_roles_v01.truth import FineTruth

ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT / 'runs/patch-roles-v01'
ROLE_PRIOR = dict(off=.5, force=.25, support=.25)
C_LOOK, C_PROBE = .02, .05
DECISION_TARGET = 1          # held-out command C2 (top -z, x=+30 mm)
DELTA_FRACTION = .20         # threshold = 20% of predicted response RMS
LOC_RADIUS = .015


def configs(K):
    return [(nF, nS) for nS in range(1, K + 1) for nF in range(0, K - nS + 1)]


def config_logprior(K):
    out = {}
    for nF, nS in configs(K):
        nO = K - nF - nS
        out[(nF, nS)] = (math.lgamma(K + 1) - math.lgamma(nF + 1) - math.lgamma(nS + 1) - math.lgamma(nO + 1)
                         + nF * math.log(ROLE_PRIOR['force']) + nS * math.log(ROLE_PRIOR['support']) + nO * math.log(ROLE_PRIOR['off']))
    z = logsumexp(list(out.values()))
    return {k: v - z for k, v in out.items()}


def data_hash(data):
    h = hashlib.sha256()
    for k in ('y', 'mask', 'cmd'): h.update(np.ascontiguousarray(data[k]).tobytes())
    for i, y in data.get('probes', []): h.update(str(i).encode()); h.update(np.ascontiguousarray(y).tobytes())
    return h.hexdigest()[:16]


def run_config(pm, data, nF, nS, seed, path, particles, moves):
    names, bounds = pm.layout(nF, nS)
    tid = f'{data_hash(data)}-{pm.family}-{pm.hier}-{nF}-{nS}-{particles}-{moves}-{seed}'
    if path.exists():
        z = np.load(path, allow_pickle=False)
        if str(z['tid']) == tid:
            return dict(theta=z['theta'], ll=z['ll'], logZ=float(z['logZ']), seconds=float(z['seconds']),
                        stages=int(z['stages']), evals=int(z['evals']), names=names)
    t0 = time.time()
    cfg = SMCConfig(particles=particles, moves=moves, batch_size=particles, maximum_stages=300)
    res = run_smc(lambda th: pm.loglik(th, nF, nS, data), bounds=bounds, config=cfg, seed=seed, target_id=tid)
    logZ = res.log_evidence + math.lgamma(nF + 1) + math.lgamma(nS + 1)
    sec = time.time() - t0
    stages = sum(s.get('kind') == 'tempering' for s in res.stages)
    np.savez(path, theta=res.theta, ll=res.log_likelihood, logZ=logZ, seconds=sec, stages=stages,
             evals=res.evaluation_count, tid=tid)
    print(f'  config {nF}F{nS}S logZ={logZ:.2f} stages={stages} evals={res.evaluation_count} {sec:.0f}s', flush=True)
    return dict(theta=res.theta, ll=res.log_likelihood, logZ=logZ, seconds=sec, stages=stages,
                evals=res.evaluation_count, names=names)


# ------------------------------------------------------------------ posterior quantities

def extras(pm, theta, nF, nS, data, batch=128):
    """Per-particle predictive quantities (numpy). Hidden target = back points, 3D, all times."""
    T = pm.T; s2 = M.NOISE ** 2
    keys = {}
    cmd = torch.as_tensor(np.asarray(data['cmd']).reshape(-1, T), dtype=pm.dtype, device=pm.device)
    for i in range(0, len(theta), batch):
        th = theta[i:i + batch]
        with torch.no_grad():
            fw = pm.forward(th, nF, nS, extras=True)
            ll, mu, S, _ = pm.marginal(fw, data, conditionals=True)
            B = len(th); nA = fw['nA']
            G3h = fw['G3'][:, 20:30]; D3h = fw['D3'][:, 20:30]
            hmean = torch.einsum('bpkc,ct->btpk', D3h, cmd) if cmd.shape[0] else torch.zeros(B, T, 10, 3, dtype=pm.dtype, device=pm.device)
            hvar = torch.zeros_like(hmean)
            Gb = fw['G'][:, 40:60]; Db = fw['D'][:, 40:60]
            looks_m, looks_c, looks_x = [], [], []
            if nA:
                Sb = S.view(B, T, nA, T, nA)
                hmean = hmean + torch.einsum('bpki,bti->btpk', G3h, mu)
                Stt = torch.stack([Sb[:, t, :, t, :] for t in range(T)], 1)
                hvar = torch.einsum('bpki,btij,bpkj->btpk', G3h, Stt, G3h)
                fvar = torch.diagonal(S, dim1=1, dim2=2).reshape(B, T, nA)
                m = pm.modes; Sd = Sb.view(B, T, nF, m, T, nF, m)
                rvar = torch.stack([torch.stack([Sd[:, t, :, k, t, :, k].sum((1, 2)) for k in range(m)], -1) for t in range(T)], 1)
                rmean = mu.view(B, T, nF, m).sum(2)
            else:
                mu = torch.zeros(B, T, 0, dtype=pm.dtype, device=pm.device); fvar = mu.clone()
                rmean = rvar = torch.zeros(B, T, pm.modes, dtype=pm.dtype, device=pm.device)
            for t in M.LOOK_TIMES:
                ym = (Db @ cmd[:, t]) if cmd.shape[0] else torch.zeros(B, 20, dtype=pm.dtype, device=pm.device)
                yc = s2 * torch.eye(20, dtype=pm.dtype, device=pm.device).expand(B, 20, 20).clone()
                X = torch.zeros(B, T, 10, 3, 20, dtype=pm.dtype, device=pm.device)
                if nA:
                    ym = ym + torch.einsum('bmi,bi->bm', Gb, mu[:, t])
                    yc = yc + torch.einsum('bmi,bij,bnj->bmn', Gb, Stt[:, t], Gb)
                    X = torch.einsum('bpki,btij,bmj->btpkm', G3h, Sb[:, :, :, t, :], Gb)
                looks_m.append(ym); looks_c.append(yc); looks_x.append(X)
            th5 = torch.zeros(B, 5, dtype=pm.dtype, device=pm.device)
            th5[:, :2] = torch.as_tensor(th[:, :2], dtype=pm.dtype, device=pm.device)
            logE = torch.log(pm.body.modulus_at_points(th5, CHECK_POINTS))
            out = dict(ll=ll, hmean=hmean, hvar=hvar, tgt=fw['Tgt3'].permute(0, 3, 1, 2), probe=fw['Pr'],
                       fmean=mu, fvar=fvar, rmean=rmean, rvar=rvar, logE=logE,
                       look_m=torch.stack(looks_m, 1), look_c=torch.stack(looks_c, 1), look_x=torch.stack(looks_x, 1))
        for k, v in out.items(): keys.setdefault(k, []).append(v.cpu().numpy())
    return {k: np.concatenate(v) for k, v in keys.items()}


CHECK_POINTS = np.stack(np.meshgrid(np.linspace(-.05, .05, 6), np.linspace(-.02, .02, 3), np.linspace(.01, .05, 3),
                                    indexing='ij'), -1).reshape(-1, 3)


def patch_centres(pm, theta, nF, nS):
    """[N,nF,3], [N,nS,3] centres; published family: (x,0,0) proxies."""
    _, F, S, _ = pm.split(theta, nF, nS)
    F = F.cpu().numpy(); S = S.cpu().numpy()
    if pm.family == 'published':
        z = lambda A: np.stack([A[..., 0], 0 * A[..., 0], 0 * A[..., 0]], -1)
        return z(F), z(S), F, S
    return M.surface_point(F[..., 0], F[..., 1])[0], M.surface_point(S[..., 0], S[..., 1])[0], F, S


def true_centres(spec):
    out = {'force': [], 'support': []}
    for p in sorted(spec['patches'], key=lambda p: p['x']):
        c = np.array([p['x'], 0, 0]) if p.get('kind') == 'pub' else M.surface_point(p['x'], p['phi'])[0]
        out[p['role']].append((c, p))
    return out


def analyze(pm, spec, gen, data, runs, K, label):
    lp = config_logprior(K)
    logpost = {c: lp[c] + r['logZ'] for c, r in runs.items()}
    zz = logsumexp(list(logpost.values()))
    post = {c: float(np.exp(v - zz)) for c, v in logpost.items()}
    tc = scenes.true_config(spec)
    keep = [c for c in runs if post[c] > 1e-6]
    parts = {c: extras(pm, runs[c]['theta'], c[0], c[1], data) for c in keep}
    w = np.concatenate([np.full(len(runs[c]['theta']), post[c] / len(runs[c]['theta'])) for c in keep]); w /= w.sum()
    cat = lambda k: np.concatenate([parts[c][k] for c in keep])
    res = dict(label=label, config_posterior={f'{a}F{b}S': p for (a, b), p in post.items()},
               config_log_evidence={f'{a}F{b}S': runs[c]['logZ'] for c in runs for (a, b) in [c]},
               config_log_prior={f'{a}F{b}S': lp[(a, b)] for (a, b) in lp},
               true_config=f'{tc[0]}F{tc[1]}S', true_config_mass=post.get(tc, 0.),
               map_config='%dF%dS' % max(post, key=post.get))
    # ---- roles at locations (all configs)
    tcent = true_centres(spec)
    for role, idx in (('force', 0), ('support', 1)):
        probs = []
        for c0, p in tcent[role]:
            pr = 0.
            for c in keep:
                Fc, Sc, _, _ = patch_centres(pm, runs[c]['theta'], *c)
                A = Fc if role == 'force' else Sc
                if A.shape[1] == 0: continue
                d = np.linalg.norm(A - c0, axis=-1).min(1)
                pr += post[c] * np.mean(d < LOC_RADIUS)
            probs.append(pr)
        res[f'p_{role}_within_15mm_of_truth'] = probs
    # spurious: expected number of active patches not within 15mm of any true patch of same role
    spur = 0.
    for c in keep:
        Fc, Sc, _, _ = patch_centres(pm, runs[c]['theta'], *c)
        for A, role in ((Fc, 'force'), (Sc, 'support')):
            tr = [c0 for c0, _ in tcent[role]]
            if A.shape[1] == 0: continue
            if not tr: spur += post[c] * A.shape[1]; continue
            d = np.min(np.stack([np.linalg.norm(A - c0, axis=-1) for c0 in tr], -1), -1)
            spur += post[c] * np.mean((d >= LOC_RADIUS).sum(1))
    res['expected_spurious_patches'] = spur
    # ---- matched metrics under the true configuration
    if tc in keep:
        th = runs[tc]['theta']; names = runs[tc]['names']
        Fc, Sc, Fp, Sp = patch_centres(pm, th, *tc)
        mt = {}
        mt['force_location_error_mm'] = [float(np.mean(np.linalg.norm(Fc[:, j] - c0, axis=-1)) * 1000) for j, (c0, _) in enumerate(tcent['force'])]
        mt['support_location_error_mm'] = [float(np.mean(np.linalg.norm(Sc[:, j] - c0, axis=-1)) * 1000) for j, (c0, _) in enumerate(tcent['support'])]
        mt['support_logk_error'] = []; mt['support_logk_cover90'] = []; mt['support_logk_post_sd'] = []
        for j, (_, p) in enumerate(tcent['support']):
            lk = Sp[:, j, -1]; mt['support_logk_error'].append(float(abs(lk.mean() - math.log(p['k']))))
            lo, hi = np.quantile(lk, [.05, .95]); mt['support_logk_cover90'].append(bool(lo <= math.log(p['k']) <= hi))
            mt['support_logk_post_sd'].append(float(lk.std()))
        if pm.family == 'general':
            mt['width_true_mm'] = [p.get('width', p.get('radius')) * 1000 for _, p in tcent['force'] + tcent['support']]
        # force histories, matched by x order
        f = parts[tc]; nA = f['fmean'].shape[-1]; m = pm.modes
        if nA:
            truthF = np.stack([np.asarray(p['history']) for _, p in tcent['force']], 1)       # [T,nF,m]
            mu = f['fmean'].reshape(len(th), -1); var = f['fvar'].reshape(len(th), -1)
            ww = np.full(len(th), 1 / len(th))
            lo, hi = gaussian_mixture_quantiles(ww, mu, var)
            tv = truthF.reshape(-1)
            mt['force_cover90'] = float(np.mean((tv >= lo) & (tv <= hi)))
            mt['force_rmse_N'] = float(np.sqrt(np.mean((ww @ mu - tv) ** 2)))
            mt['force_band_width_N'] = float(np.mean(hi - lo))
        res['matched_true_config'] = mt
    # ---- material
    gl = res.setdefault('global', {})
    allth = [(c, runs[c]['theta']) for c in keep]
    m0 = np.concatenate([t[:, 0] for _, t in allth]); m1 = np.concatenate([t[:, 1] for _, t in allth])
    gl['logE0_mean'] = float(w @ m0); gl['logE0_true'] = spec['logE']; gl['logE0_error'] = float(abs(w @ m0 - spec['logE']))
    lo, hi = discrete_quantiles(w, m0)[:, 0]; gl['logE0_cover90'] = bool(lo <= spec['logE'] <= hi); gl['logE0_interval90'] = [float(lo), float(hi)]
    gl['m1_mean'] = float(w @ m1); gl['m1_true'] = spec['m1']
    tth = np.zeros((1, 5)); tth[0, 0] = spec['logE']; tth[0, 1] = spec['m1']
    tlogE = np.log(pm.body.modulus_at_points(torch.as_tensor(tth, device=pm.device), CHECK_POINTS).cpu().numpy()[0])
    lE = cat('logE'); gl['logE_field_rmse'] = float(np.sqrt(np.mean((w @ lE - tlogE) ** 2)))
    lo, hi = discrete_quantiles(w, lE); gl['logE_field_cover90'] = float(np.mean((tlogE >= lo) & (tlogE <= hi)))
    if pm.hier:
        tau = np.concatenate([np.exp(t[:, 2]) for _, t in allth]); gl['tau_mean'] = float(w @ tau); gl['tau_true'] = spec.get('tau', 1.)
        lo, hi = discrete_quantiles(w, tau)[:, 0]; gl['tau_interval90'] = [float(lo), float(hi)]
        cv = np.cov(np.stack([m0, np.log(tau)]), aweights=w); gl['corr_logE0_logtau'] = float(cv[0, 1] / np.sqrt(cv[0, 0] * cv[1, 1]))
    # resultant force over all configs (zero-force configs contribute exact zeros)
    rm = cat('rmean').reshape(len(w), -1); rv = cat('rvar').reshape(len(w), -1)
    tr = np.zeros((8, pm.modes))
    for _, p in tcent['force']: tr += np.asarray(p['history'])
    lo, hi = gaussian_mixture_quantiles(w, rm, rv + 1e-30)
    gl['resultant_force_cover90'] = float(np.mean((tr.reshape(-1) >= lo) & (tr.reshape(-1) <= hi)))
    gl['resultant_force_rmse_N'] = float(np.sqrt(np.mean((w @ rm - tr.reshape(-1)) ** 2)))
    # ---- prediction (a) hidden displacement under the original loads
    pr = res.setdefault('prediction', {})
    hm = cat('hmean').reshape(len(w), -1); hv = cat('hvar').reshape(len(w), -1)
    htrue = gen['hidden'].reshape(-1)
    hmean = w @ hm; hvar_tot = w @ (hv + (hm - hmean) ** 2)
    lo, hi = gaussian_mixture_quantiles(w, hm, hv + 1e-30)
    pr['hidden_rmse_mm'] = float(np.sqrt(np.mean((hmean - htrue) ** 2)) * 1000)
    pr['hidden_pred_sd_mm'] = float(np.sqrt(hvar_tot.mean()) * 1000)
    pr['hidden_cover90'] = float(np.mean((htrue >= lo) & (htrue <= hi)))
    pr['hidden_z_rms'] = float(np.sqrt(np.mean((hmean - htrue) ** 2 / hvar_tot)))
    # ---- prediction (b) held-out known commands
    tg = cat('tgt')                                                    # [N,4,30,3]
    pr['commands'] = []
    for c in range(tg.shape[1]):
        q = tg[:, c].reshape(len(w), -1); qt = gen['target3d'][c].reshape(-1)
        qh = w @ q; delta = DELTA_FRACTION * np.sqrt(np.mean(qh ** 2))
        dist = np.sqrt(np.mean((q - qh) ** 2, 1)); risk = float(w @ (dist > delta))
        err = float(np.sqrt(np.mean((qh - qt) ** 2)))
        lo, hi = discrete_quantiles(w, q)
        pr['commands'].append(dict(name=M.TARGET_COMMANDS[c]['name'], rmse_mm=err * 1000, truth_rms_mm=float(np.sqrt(np.mean(qt ** 2)) * 1000),
                                   pred_sd_mm=float(np.sqrt((w @ (q - qh) ** 2).mean()) * 1000), cover90=float(np.mean((qt >= lo) & (qt <= hi))),
                                   delta_mm=delta * 1000, risk=risk, exceeded=bool(err > delta)))
    res['design'] = design(w, cat, gen, data)
    return res


def design(w, cat, gen, data, n_y=128, seed=11):
    """Proceed / look (back camera frame of the original episode) / calibrated probe.
    Target (b): threshold risk of C2; target (a): hidden-displacement predictive variance."""
    rng = np.random.default_rng(seed)
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    tt = lambda a: torch.as_tensor(a, dtype=torch.float64, device=dev)
    W = tt(w); N = len(w); s2 = M.NOISE ** 2
    q = tt(cat('tgt')[:, DECISION_TARGET].reshape(N, -1))
    hm = tt(cat('hmean').reshape(N, -1)); hv = tt(cat('hvar').reshape(N, -1))

    def risk_b(Wp):                       # Wp [Y,N]
        qh = Wp @ q; delta = DELTA_FRACTION * torch.sqrt((qh ** 2).mean(1, keepdim=True))
        dist = torch.sqrt(((q[None] - qh[:, None]) ** 2).mean(-1))       # [Y,N]
        return (Wp * (dist > delta)).sum(1)

    def var_b(Wp):                        # squared-error risk of the decision target
        qh = Wp @ q
        return (Wp @ (q ** 2)).mean(1) - (qh ** 2).mean(1)

    def hvar_mix(Wp, means=None, var=None):
        means = hm[None] if means is None else means; var = hv[None] if var is None else var
        mm = torch.einsum('yn,ynd->yd', Wp, means.expand(Wp.shape[0], -1, -1))
        second = torch.einsum('yn,ynd->yd', Wp, (var + means ** 2).expand(Wp.shape[0], -1, -1))
        return (second - mm ** 2).mean(1)

    r0 = float(risk_b(W[None])[0]); h0 = float(hvar_mix(W[None])[0])
    v0 = float(var_b(W[None])[0])
    rows = [dict(action='proceed', cost=0., expected_risk_b=r0, J=r0, expected_hidden_sd_mm=math.sqrt(h0) * 1000,
                 expected_target_sd_b_mm=math.sqrt(max(v0, 0)) * 1000)]
    idx = rng.choice(N, size=n_y, p=w)
    z = rng.standard_normal((n_y, 40))
    probe = tt(cat('probe'))                                        # [N,40,6]
    done = {i for i, _ in data.get('probes', [])}
    for s in range(probe.shape[-1]):
        if s in done: continue
        g = probe[:, :, s]
        y = g[idx] + M.NOISE * tt(z)
        ll = -.5 * ((y[:, None] - g[None]) ** 2).sum(-1) / s2 + torch.log(W)[None]
        Wp = torch.softmax(ll, 1)
        r = float(risk_b(Wp).mean()); h = float(hvar_mix(Wp).mean()); v = float(var_b(Wp).clamp(min=0).mean())
        rows.append(dict(action=M.PROBES[s]['name'], kind='probe', index=s, cost=C_PROBE, expected_risk_b=r, J=C_PROBE + r,
                         expected_hidden_sd_mm=math.sqrt(h) * 1000, expected_target_sd_b_mm=math.sqrt(v) * 1000))
    lm, lc, lx = cat('look_m'), cat('look_c'), cat('look_x')
    for li, t in enumerate(M.LOOK_TIMES):
        m_ = tt(lm[:, li]); C_ = tt(lc[:, li]); X = tt(lx[:, li].reshape(N, -1, 20))
        L = torch.linalg.cholesky(C_)
        z2 = tt(rng.standard_normal((n_y, 20)))
        y = m_[idx] + torch.einsum('yij,yj->yi', L[idx], z2)
        A = torch.cholesky_solve(X.transpose(1, 2), L).transpose(1, 2)    # X C^-1  [N,D,20]
        vpost = hv - (A * X).sum(-1)
        logdet = 2 * torch.log(torch.diagonal(L, dim1=1, dim2=2)).sum(1)
        rs, hs, vs = [], [], []
        for y0 in range(0, n_y, 8):
            yy = y[y0:y0 + 8]
            res = yy[:, None] - m_[None]                                    # [Y,N,20]
            sol = torch.cholesky_solve(res.permute(1, 2, 0), L)             # [N,20,Y]
            ll = -.5 * ((res.permute(1, 2, 0) * sol).sum(1).T + logdet[None]) + torch.log(W)[None]
            Wp = torch.softmax(ll, 1)
            means = hm[None] + torch.einsum('ndm,nmy->ynd', A, res.permute(1, 2, 0))
            rs.append(risk_b(Wp)); hs.append(hvar_mix(Wp, means, vpost[None])); vs.append(var_b(Wp).clamp(min=0))
        r = float(torch.cat(rs).mean()); h = float(torch.cat(hs).mean()); v = float(torch.cat(vs).mean())
        rows.append(dict(action=f'look back camera t{t}', kind='look', index=t, cost=C_LOOK, expected_risk_b=r, J=C_LOOK + r,
                         expected_hidden_sd_mm=math.sqrt(h) * 1000, expected_target_sd_b_mm=math.sqrt(v) * 1000))
    best = min(rows, key=lambda r: r['J'])
    probes = [r for r in rows if r.get('kind') == 'probe']
    # Ties in threshold risk (common when every hypothesis is inside tolerance) are broken by squared-error risk.
    best_probe = min(probes, key=lambda r: (round(r['expected_risk_b'], 3), r['expected_target_sd_b_mm'])) if probes else None
    return dict(rows=rows, decision=best['action'], best_probe=best_probe, n_y=n_y,
                costs=dict(look=C_LOOK, probe=C_PROBE), target=M.TARGET_COMMANDS[DECISION_TARGET]['name'])


# ------------------------------------------------------------------ published regression

def published_reference(pm, data, seed, particles, moves):
    """Independent run: published dense MarginalGaussianLikelihood + published body.solve,
    PARAMETERS order (logE, m1, support_x, log k, load_x), published BOUNDS."""
    body = pm.body; T = pm.T; mask = np.asarray(data['mask'], bool)
    rows = [(t, m) for t in range(T) for m in range(60) if mask[t, m]]
    yy = np.array([data['y'][t, m] for t, m in rows])

    def mapping(theta):
        P = []
        for i in range(0, len(theta), 128):
            with torch.no_grad():
                U = body.solve(torch.as_tensor(theta[i:i + 128], device=pm.device))['unit_response']   # [B,n,3,2]
                P.append(torch.einsum('pn,bnkc->bpkc', pm.interp, U)[:, pm._row_point, pm._row_comp].cpu().numpy())
        P = np.concatenate(P)
        H = np.zeros((len(theta), len(rows), 2 * T))
        for i, (t, m) in enumerate(rows): H[:, i, 2 * t:2 * t + 2] = P[:, m]
        return H
    lik = MarginalGaussianLikelihood(mapping, yy, pub.load_cov(), M.NOISE ** 2)
    t0 = time.time()
    res = run_smc(lik, bounds=pub.BOUNDS, config=SMCConfig(particles=particles, moves=moves, batch_size=particles, maximum_stages=300), seed=seed)
    return res, time.time() - t0


# ------------------------------------------------------------------ main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--scene', required=True); ap.add_argument('--particles', type=int, default=512)
    ap.add_argument('--moves', type=int, default=6); ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--no-probe', action='store_true')
    ap.add_argument('--truth-cells', type=int, nargs=3, default=[16, 8, 8])
    a = ap.parse_args()
    torch.set_num_threads(2)
    coarse_truth = tuple(a.truth_cells) != (16, 8, 8)
    out = RUNS / (a.scene + ('-truth%dx%dx%d' % tuple(a.truth_cells) if coarse_truth else '')); out.mkdir(parents=True, exist_ok=True)
    spec, st = scenes.get(a.scene)
    t_start = time.time()
    tp = out / 'truth.npz'
    if tp.exists():
        gen = dict(np.load(tp))
    else:
        gen = FineTruth('cuda', cells=tuple(a.truth_cells)).generate(spec, seed=100); np.savez(tp, **gen)
    pm = M.PatchModel(st['family'], device='cuda', commands=spec['commands'], hierarchical=st['hier'])
    mask = np.zeros((8, 60), bool); mask[:, :40] = True
    data = dict(y=gen['y'], mask=mask, cmd=np.asarray(spec['cmd']).reshape(-1, 8), probes=[])
    report = dict(scene=a.scene, settings=st, particles=a.particles, moves=a.moves, seed=a.seed,
                  truth=dict(logE=spec['logE'], m1=spec['m1'], tau=spec.get('tau', 1.),
                             patches=[{k: (np.asarray(v).tolist() if k == 'history' else v) for k, v in p.items()} for p in spec['patches']],
                             commands=spec['commands'], cmd=np.asarray(spec['cmd']).tolist()),
                  inference_mesh=list(M.COARSE_CELLS), truth_mesh=list(a.truth_cells), stages={})
    rp = out / f'result-s{a.seed}.json'

    def stage(name, data):
        print(f'[{a.scene}] stage {name}', flush=True)
        runs = {}; t0 = time.time()
        for c in configs(st['K']):
            runs[c] = run_config(pm, data, *c, seed=a.seed * 1000 + 17 * c[0] + c[1], path=out / f'{name}-{c[0]}F{c[1]}S-s{a.seed}.npz',
                                 particles=a.particles, moves=a.moves)
        smc_sec = time.time() - t0
        t1 = time.time(); res = analyze(pm, spec, gen, data, runs, st['K'], name)
        res['compute'] = dict(smc_seconds=smc_sec, analysis_seconds=time.time() - t1,
                              per_config={f'{c[0]}F{c[1]}S': dict(seconds=r['seconds'], stages=r['stages'], evals=r['evals']) for c, r in runs.items()})
        report['stages'][name] = res
        rp.write_text(json.dumps(report, indent=2, default=float))
        print(json.dumps({k: res[k] for k in ('config_posterior', 'true_config_mass', 'global', 'prediction')}, indent=1, default=float)[:3000], flush=True)
        return res, runs

    res, runs = stage('initial', data)
    if st.get('regression'):
        ref, sec = published_reference(pm, data, a.seed + 99, a.particles, a.moves)
        mine = runs[(1, 1)]['theta']          # (logE, m1, F_x, S_x, S_logk)
        mine5 = np.stack([mine[:, 0], mine[:, 1], mine[:, 3], mine[:, 4], mine[:, 2]], 1)
        report['regression_published_1F1S'] = dict(
            parameters=list(pub.PARAMETERS), published_mean=ref.theta.mean(0).tolist(), published_sd=ref.theta.std(0).tolist(),
            patch_mean=mine5.mean(0).tolist(), patch_sd=mine5.std(0).tolist(),
            published_logZ=ref.log_evidence, patch_logZ=runs[(1, 1)]['logZ'], published_seconds=sec,
            truth=[spec['logE'], spec['m1'], .0125, math.log(8000.), -.02])
        rp.write_text(json.dumps(report, indent=2, default=float))
        print('regression', json.dumps(report['regression_published_1F1S'], indent=1), flush=True)
    if st.get('probe') and not a.no_probe:
        bp = res['design']['best_probe']; s = bp['index']
        yprobe = gen['probe_clean'][s] + gen['probe_noise'][s]
        data2 = dict(data, probes=[(s, yprobe)])
        report['executed_probe'] = dict(index=s, name=M.PROBES[s]['name'], expected_risk_b=bp['expected_risk_b'])
        stage('after_probe', data2)
    if st.get('fixed_probe') is not None and not a.no_probe:
        s = st['fixed_probe']
        data3 = dict(data, probes=[(s, gen['probe_clean'][s] + gen['probe_noise'][s])])
        report['fixed_probe'] = dict(index=s, name=M.PROBES[s]['name'])
        stage('after_fixed_probe', data3)
    report['total_seconds'] = time.time() - t_start
    rp.write_text(json.dumps(report, indent=2, default=float))
    print('done', a.scene, report['total_seconds'], flush=True)


if __name__ == '__main__':
    main()
