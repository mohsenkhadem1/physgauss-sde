"""PokeFlex v02 ladder: calibration surface tables, grid posteriors/evidence, selection, held-out predictions.
Protocol: scripts/pokeflex_v02/protocol_v02.json.  Calibration-window outcomes only (read through the v01 gate);
predictions use free inputs only.
usage: python -I v02_fit.py surftab OBJ | fit OBJ | select | predict OBJ
  surftab  chamfer_excl_tool of every 2nd calibration frame on the (Delta, delta_s) grid, per usable nu
  fit      M0 (recomputed, check vs v01), M1, M1b, M2, M3 grid posteriors + evidence -> work/fit/<obj>.npz, pred/<rung>/<obj>/fit.json
  select   selection rule on calibration evidence (all objects) -> runs/pokeflex-v02/selection.json
  predict  held-out predictions for M1, M1b, M2, M3 -> pred/<rung>/<obj>/<take>-p<k>.{npz,json}"""
import json, sys, time
from pathlib import Path
import numpy as np
from scipy.special import logsumexp

sys.path.insert(0, str(Path(__file__).resolve().parent))
from v02_model import (P, RF, OBJECTS, NUS, E_GRID, TAU, G, K, RFR, DELTA_GRID, S_GRID, E_WIN, SURF_EVERY, OUT2, PRED2, WORK2,
                       V01PRED, PokeSim, forward, tau_pairs, g_pairs, save_pred, heldout_pokes, poke, contact_depth)
from pfx_common import calibration_force

RUNGS = ['M1', 'M1b', 'M2', 'M3']
N_RISK = 2000
TOP_MASS, TOP_MAX = 0.999, 600


def log(*a):
    print(time.strftime('%H:%M:%S'), *a, flush=True)


def cal_data(obj):
    cal, off, n_air = calibration_force(obj)
    return [c for c in cal if c[4][0] <= 0], off


def usable_nus(obj, cal):
    """v01 rule: nu usable iff every calibration poke has a non-snapped simulation."""
    out = {}
    for nu in NUS:
        ps = [PokeSim(obj, t, k, nu) for t, k, *_ in cal]
        if all(p.ok and not p.snapped for p in ps):
            out[nu] = ps
    return out


# ---------------------------------------------------------------- surface tables
def run_surftab(obj):
    cal, _ = cal_data(obj); sims = usable_nus(obj, cal)
    out = {}; n = 0; t0 = time.time()
    for ci, (t, k, fr, Fm, dep) in enumerate(cal):
        T = P.robot_trajectory(obj, t)['T_WT']
        meshes = {j: P.mesh(obj, t, int(fr[j])) for j in range(0, len(fr), SURF_EVERY)}
        for nu, pss in sims.items():
            ps = pss[ci]; a = ps.a
            tab = np.zeros((len(meshes), len(DELTA_GRID), len(S_GRID)))
            for jj, j in enumerate(meshes):
                Vt, Ft, conf = meshes[j]; Tk = T[fr[j]-1]
                V0 = ps.z['surf_V'][j].astype(np.float64); Sd = ps.S(ps.dep[j])
                prev = None
                for di, D in enumerate(DELTA_GRID):
                    dS = ps.S(ps.dep[j]-D)-Sd if D > 0 else 0.0
                    if di > 0 and np.max(np.abs(dS)) < 1e-9:
                        tab[jj, di] = tab[jj, 0]; continue
                    for si, s in enumerate(S_GRID):
                        e = P.surface_error_mm(V0+dS+s*a, ps.surfF, Vt, Ft, conf_t=conf, tool_tip=Tk[:3, 3], tool_axis=Tk[:3, 2])
                        tab[jj, di, si] = e['chamfer_excl_tool']; n += 1
            out[f'c{ci}_nu{nu:.1f}'] = tab
            log(obj, t, k, nu, f'{n} chamfers {time.time()-t0:.0f}s')
    (WORK2/'surftab').mkdir(parents=True, exist_ok=True)
    np.savez(WORK2/'surftab'/f'{obj}.npz', **out)


def bilinear(tab, D, s):
    """tab (nD, nS); D, s arrays (same shape) -> interpolated values and clamp flag."""
    clamp = (D > DELTA_GRID[-1]) | (s > S_GRID[-1])
    D = np.clip(D, 0, DELTA_GRID[-1]); s = np.clip(s, 0, S_GRID[-1])
    i = np.clip(np.searchsorted(DELTA_GRID, D, side='right')-1, 0, len(DELTA_GRID)-2)
    j = np.clip(np.searchsorted(S_GRID, s, side='right')-1, 0, len(S_GRID)-2)
    u = (D-DELTA_GRID[i])/(DELTA_GRID[i+1]-DELTA_GRID[i]); v = (s-S_GRID[j])/(S_GRID[j+1]-S_GRID[j])
    val = (1-u)*(1-v)*tab[i, j]+u*(1-v)*tab[i+1, j]+(1-u)*v*tab[i, j+1]+u*v*tab[i+1, j+1]
    return val, clamp


# ---------------------------------------------------------------- fit
def lag1(series_list):
    num = sum(float(r[:-1] @ r[1:]) for r in series_list if len(r) > 1)
    den = sum(float(r @ r) for r in series_list)
    return float(np.clip(num/den if den > 0 else 0.0, 0, 0.95))


def quad_sse(Fhat, cal):
    """Fhat: list over cal pokes of (N,K,3) at c=1 -> SSE over E_GRID (N, 401)."""
    ff = sum((f*f).sum((1, 2)) for f in Fhat); fF = sum((f*c[3][None]).sum((1, 2)) for f, c in zip(Fhat, cal))
    FF = sum(float((c[3]**2).sum()) for c in cal)
    cc = E_GRID/RF.E_REF
    return cc[None]**2*ff[:, None]-2*cc[None]*fF[:, None]+FF


def run_fit(obj):
    cal, off = cal_data(obj); sims = usable_nus(obj, cal)
    st = np.load(WORK2/'surftab'/f'{obj}.npz')
    fit01 = json.loads((V01PRED/'ours'/obj/'fit.json').read_text())
    s2F0, s2S0 = fit01['noise_force_N']**2, fit01['noise_surface_mm']**2
    nF = sum(c[3].size for c in cal)
    sframes = [list(range(0, len(c[2]), SURF_EVERY)) for c in cal]
    nS = sum(len(s) for s in sframes)
    inu = {nu: NUS.index(nu) for nu in sims}
    R = {}  # rung -> dict(sseF=array over grid incl nu axis, sseS=..., axes=...)

    # ---- M0 and surface base term (Delta = s = 0) per nu
    sse0S = np.full(len(NUS), np.nan)
    for nu in sims:
        sse0S[inu[nu]] = sum(float((st[f'c{ci}_nu{nu:.1f}'][:, 0, 0]**2).sum()) for ci in range(len(cal)))
    sseF = np.full((len(NUS), len(E_GRID)), np.inf)
    for nu, pss in sims.items():
        sseF[inu[nu]] = quad_sse([p.F[None] for p in pss], cal)[0]
    R['M0'] = dict(sseF=sseF, sseS=np.broadcast_to(sse0S[:, None], sseF.shape).copy(), shape=('nu', 'E'))

    # ---- M1 (1 Prony term) and M1b (2 terms): closed form in E
    TT, GG = np.meshgrid(np.arange(len(TAU)), np.arange(len(G)), indexing='ij')
    taus1, gs1 = TAU[TT.ravel()][:, None], G[GG.ravel()][:, None]
    tp, gp = tau_pairs(), g_pairs()
    taus2 = np.array([[TAU[i], TAU[j]] for i, j in tp for _ in gp]); gs2 = np.array([[G[a], G[b]] for _ in tp for a, b in gp])
    for name, taus, gs, shp in (('M1', taus1, gs1, (len(TAU), len(G))), ('M1b', taus2, gs2, (len(tp), len(gp)))):
        sseF = np.full((len(NUS),)+shp+(len(E_GRID),), np.inf)
        for nu, pss in sims.items():
            Fh = [forward(p, np.ones(len(taus)), taus, gs)[0] for p in pss]
            sseF[inu[nu]] = quad_sse(Fh, cal).reshape(shp+(len(E_GRID),))
        R[name] = dict(sseF=sseF, sseS=np.broadcast_to(sse0S.reshape((-1,)+(1,)*(len(shp)+1)), sseF.shape).copy(),
                       shape=('nu', 'tau', 'g', 'E') if name == 'M1' else ('nu', 'taupair', 'gpair', 'E'))
        log(obj, name, 'done')

    # ---- M2 / M3: E window, bisection compliance
    lE0 = np.log10(fit01['posterior_mean_E']); lE = np.log10(E_GRID)
    lo, hi = lE0+E_WIN[0], lE0+E_WIN[1]
    extensions = []
    while True:
        ew = np.flatnonzero((lE >= lo-1e-9) & (lE <= hi+1e-9)); Ew = E_GRID[ew]
        shp = (len(Ew), len(TAU), len(G), len(K))
        sseF2 = np.full((len(NUS),)+shp, np.inf); sseS2 = np.full((len(NUS),)+shp, np.inf)
        sseS3 = np.full((len(NUS),)+shp+(len(RFR),), np.inf); clamp3 = np.zeros((len(NUS),)+shp+(len(RFR),), np.int16)
        clamp2 = np.zeros((len(NUS),)+shp, np.int16)
        _, TTk, GGk, KKk = np.meshgrid(np.arange(1), np.arange(len(TAU)), np.arange(len(G)), np.arange(len(K)), indexing='ij')
        tk, gk, kk = TAU[TTk.ravel()], G[GGk.ravel()], K[KKk.ravel()]
        for nu, pss in sims.items():
            for ie, E in enumerate(Ew):
                c = np.full(len(tk), E/RF.E_REF)
                sF = np.zeros(len(tk)); sS2 = np.zeros(len(tk)); sS3 = np.zeros((len(tk), len(RFR))); cl2 = np.zeros(len(tk), int); cl3 = np.zeros((len(tk), len(RFR)), int)
                for ci, (p, cd) in enumerate(zip(pss, cal)):
                    Fh, Dl, fl = forward(p, c, tk[:, None], gk[:, None], kk, want=('state',))
                    sF += ((Fh-cd[3][None])**2).sum((1, 2))
                    tab = st[f'c{ci}_nu{nu:.1f}']
                    for jj, j in enumerate(sframes[ci]):
                        v, cflag = bilinear(tab[jj], Dl[:, j], np.zeros(len(tk))); sS2 += v**2; cl2 += cflag
                        for ir, r in enumerate(RFR):
                            v, cflag = bilinear(tab[jj], Dl[:, j], r*fl[:, j]/kk); sS3[:, ir] += v**2; cl3[:, ir] += cflag
                sseF2[inu[nu], ie] = sF.reshape(shp[1:]); sseS2[inu[nu], ie] = sS2.reshape(shp[1:]); clamp2[inu[nu], ie] = cl2.reshape(shp[1:])
                sseS3[inu[nu], ie] = sS3.reshape(shp[1:]+(len(RFR),)); clamp3[inu[nu], ie] = cl3.reshape(shp[1:]+(len(RFR),))
            log(obj, 'M2/M3 nu', nu, 'done')
        # edge check (posterior for prediction, M2 and M3)
        need = []
        for sF, sS in ((sseF2, sseS2), (sseF2[..., None], sseS3)):
            lp = -sF/(2*np.min(sF)/nF)-sS/(2*np.min(sS)/nS); pp = np.exp(lp-lp.max()); pp /= pp.sum()
            mE = pp.sum(axis=tuple(i for i in range(pp.ndim) if i != 1))
            if mE[:3].sum() > 1e-3 and lo > lE[0]:
                need.append('lo')
            if mE[-3:].sum() > 1e-3 and hi < lE[-1]:
                need.append('hi')
        if not need:
            break
        if 'lo' in need:
            lo -= 0.5
        if 'hi' in need:
            hi += 0.5
        extensions.append(dict(lo=lo, hi=hi)); log(obj, 'E window extended', lo, hi)
    R['M2'] = dict(sseF=sseF2, sseS=sseS2, clamp=clamp2, shape=('nu', 'E', 'tau', 'g', 'k'))
    R['M3'] = dict(sseF=np.broadcast_to(sseF2[..., None], sseS3.shape), sseS=sseS3, clamp=clamp3, shape=('nu', 'E', 'tau', 'g', 'k', 'r'))

    # ---- M0 check against the v01 posterior
    lp0 = -R['M0']['sseF']/(2*np.min(R['M0']['sseF'])/nF)-R['M0']['sseS']/(2*np.nanmin(sse0S)/nS)
    lp0 = np.where(np.isfinite(lp0), lp0, -np.inf); p0 = np.exp(lp0-lp0.max()); p0 /= p0.sum()
    pv1 = np.load(ROOT_V01_POST/f'{obj}.npz')['post']
    m0_check = float(np.abs(p0-pv1).max())

    # ---- tempering from M0 residuals
    resid = []
    for ci, cd in enumerate(cal):
        Fp = sum(float((p0[inu[nu]]*E_GRID).sum())/RF.E_REF*pss[ci].F for nu, pss in sims.items())
        resid.append((Fp-cd[3]) @ -pss[ci].a)
    rhoF = lag1(resid)
    numap = NUS[int(np.argmax(p0.sum(1)))]
    sres = [st[f'c{ci}_nu{numap:.1f}'][:, 0, 0] for ci in range(len(cal))]; mu = np.mean(np.concatenate(sres))
    rhoS = lag1([s-mu for s in sres])
    wF, wS = (1-rhoF)/(1+rhoF), (1-rhoS)/(1+rhoS)

    # ---- evidence + posterior per rung
    nE_full = len(E_GRID)
    cells = {'nu': len(NUS), 'E': nE_full, 'tau': len(TAU), 'g': len(G), 'k': len(K), 'r': len(RFR), 'taupair': len(tp), 'gpair': len(gp)}
    nparam = {'M0': 2, 'M1': 4, 'M1b': 6, 'M2': 5, 'M3': 6}
    summary = dict(object=obj, nF=nF, nS=nS, rho_force=rhoF, rho_surface=rhoS, w_force=wF, w_surface=wS, n_eff_force=nF*wF, n_eff_surface=nS*wS,
                   sigma_F_M0=float(np.sqrt(s2F0)), sigma_S_M0=float(np.sqrt(s2S0)), m0_vs_v01_posterior_maxabs=m0_check,
                   E_window_log10=[lo, hi], E_window_extensions=extensions, usable_nu=list(sims), rungs={})
    post_store = {}
    for name in ['M0', 'M1', 'M1b', 'M2', 'M3']:
        r = R[name]; sF, sS = np.asarray(r['sseF']), np.asarray(r['sseS'])
        ok = np.isfinite(sF) & np.isfinite(sS)
        lprior = -sum(np.log(cells[a]) for a in r['shape'])
        ll_sel = np.where(ok, -wF*sF/(2*s2F0)-wS*sS/(2*s2S0), -np.inf)
        ll_unt = np.where(ok, -sF/(2*s2F0)-sS/(2*s2S0), -np.inf)
        logZ, logZu = float(logsumexp(ll_sel)+lprior), float(logsumexp(ll_unt)+lprior)
        s2F, s2S = float(sF[ok].min()/nF), float(sS[ok].min()/nS)
        lp = np.where(ok, -sF/(2*s2F)-sS/(2*s2S), -np.inf); pp = np.exp(lp-lp.max()); pp /= pp.sum()
        imap = np.unravel_index(int(np.argmax(np.where(ok, sF/s2F0*wF+sS/s2S0*wS, np.inf)*-1)), sF.shape)
        rec = dict(log_evidence_selection=logZ, log_evidence_untempered=logZu, max_loglik_selection=float(ll_sel.max()),
                   BIC_neff=float(-2*ll_sel.max()+nparam[name]*np.log(nF*wF)), n_params=nparam[name],
                   noise_force_N=float(np.sqrt(s2F)), noise_surface_mm=float(np.sqrt(s2S)),
                   cal_force_rmse_best_N=float(np.sqrt(sF[ok].min()/(nF/3))), cal_surface_rms_best_mm=float(np.sqrt(sS[ok].min()/nS)))
        if 'clamp' in r:
            rec['posterior_mass_surface_clamped'] = float(pp[np.asarray(r['clamp']) > 0].sum())
        # parameter values of every grid point
        axes_vals = axis_values(name, r['shape'], ew, tp, gp)
        rec['posterior'] = summarize(name, pp, axes_vals, r['shape'])
        rec['selection_map'] = {a: np.atleast_1d(axes_vals[a][i]).tolist() for a, i in zip(r['shape'], imap)}
        summary['rungs'][name] = rec
        if name != 'M0':
            post_store[name] = top_points(name, pp, axes_vals, r['shape'])
            post_store[name]['noise_force_N'] = rec['noise_force_N']
    (WORK2/'fit').mkdir(parents=True, exist_ok=True)
    with open(WORK2/'fit'/f'{obj}.json', 'w') as fh:
        json.dump(summary, fh, indent=1)
    for name, pts in post_store.items():
        (PRED2/name/obj).mkdir(parents=True, exist_ok=True)
        np.savez(WORK2/'fit'/f'{obj}-{name}-points.npz', **{a: np.asarray(v) for a, v in pts.items()})
        fj = dict(rung=name, calibration_pokes=[[c[0], c[1]] for c in cal], offset_N=off.tolist(), **{k: v for k, v in summary.items() if k != 'rungs'},
                  **summary['rungs'][name], retained_mass=pts['retained_mass'].item(), n_points=int(len(pts['w'])))
        (PRED2/name/obj/'fit.json').write_text(json.dumps(fj, indent=1))
    log(obj, 'fit done', {n: round(v['log_evidence_selection'], 2) for n, v in summary['rungs'].items()}, 'M0 check', m0_check)


ROOT_V01_POST = V01PRED.parent/'posterior'


def axis_values(name, shape, ew, tp, gp):
    v = {'nu': np.array(NUS), 'tau': TAU, 'g': G, 'k': K, 'r': RFR}
    v['E'] = E_GRID if name in ('M0', 'M1', 'M1b') else E_GRID[ew]
    v['taupair'] = np.array([[TAU[i], TAU[j]] for i, j in tp]); v['gpair'] = np.array([[G[a], G[b]] for a, b in gp])
    return v


def param_arrays(name, idx, av, shape):
    """Per-point parameter arrays from grid indices idx (tuple of arrays)."""
    d = {a: av[a][i] for a, i in zip(shape, idx)}
    out = dict(nu=d['nu'], E=d['E'])
    if 'tau' in d:
        out['taus'] = d['tau'][:, None]; out['gs'] = d['g'][:, None]
    elif 'taupair' in d:
        out['taus'] = d['taupair']; out['gs'] = d['gpair']
    else:
        out['taus'] = np.zeros((len(d['E']), 0)); out['gs'] = np.zeros((len(d['E']), 0))
    out['k'] = d.get('k', np.full(len(d['E']), np.inf)); out['r'] = d.get('r', np.zeros(len(d['E'])))
    return out


def summarize(name, pp, av, shape):
    idx = np.nonzero(pp > pp.max()*1e-12); w = pp[idx]; w = w/w.sum()
    pa = param_arrays(name, idx, av, shape)
    def ms(x):
        m = float(w @ x); return dict(mean=m, sd=float(np.sqrt(max(w @ (x-m)**2, 0))))
    s = dict(log10_E=ms(np.log10(pa['E'])), nu=ms(pa['nu']))
    if pa['gs'].shape[1]:
        ginf = 1-pa['gs'].sum(1)
        s['g_sum'] = ms(pa['gs'].sum(1)); s['log10_E_relaxed'] = ms(np.log10(pa['E']*ginf))
        for i in range(pa['gs'].shape[1]):
            s[f'g{i+1}'] = ms(pa['gs'][:, i]); s[f'log10_tau{i+1}'] = ms(np.log10(pa['taus'][:, i]))
    if np.isfinite(pa['k']).all():
        s['log10_k_eff'] = ms(np.log10(pa['k']))
        if name == 'M3':
            kc, ks = np.log10(pa['k']/(1-pa['r'])), np.log10(pa['k']/pa['r'])
            s['r'] = ms(pa['r']); s['r_sd_over_prior_sd'] = s['r']['sd']/np.sqrt(1/12)
            s['log10_k_c'] = ms(kc); s['log10_k_s'] = ms(ks)
            c = w @ ((kc-s['log10_k_c']['mean'])*(ks-s['log10_k_s']['mean']))
            s['corr_log_kc_log_ks'] = float(c/max(s['log10_k_c']['sd']*s['log10_k_s']['sd'], 1e-12))
            rm = pp.sum(axis=tuple(range(pp.ndim-1))); s['r_marginal'] = (rm/rm.sum()).tolist()
    nm = pp.sum(axis=tuple(range(1, pp.ndim))); s['nu_marginal'] = dict(zip([f'{n:.1f}' for n in NUS], map(float, nm/nm.sum())))
    return s


def top_points(name, pp, av, shape):
    flat = pp.ravel(); order = np.argsort(flat)[::-1]
    cum = np.cumsum(flat[order]); n = int(min(np.searchsorted(cum, TOP_MASS)+1, TOP_MAX))
    sel = order[:n]; idx = np.unravel_index(sel, pp.shape)
    pa = param_arrays(name, idx, av, shape)
    return dict(w=flat[sel]/flat[sel].sum(), retained_mass=np.array(float(cum[n-1])), nu=pa['nu'], E=pa['E'], taus=pa['taus'], gs=pa['gs'], k=pa['k'], r=pa['r'])


# ---------------------------------------------------------------- selection (calibration evidence only)
def run_select():
    fits = {o: json.loads((WORK2/'fit'/f'{o}.json').read_text()) for o in OBJECTS}
    ladder = ['M0', 'M1', 'M2', 'M3']
    tot = {r: sum(fits[o]['rungs'][r]['log_evidence_selection'] for o in OBJECTS) for r in ladder+['M1b']}
    steps, selected, chain = [], 'M0', True
    for a, b in zip(ladder[:-1], ladder[1:]):
        d = tot[b]-tot[a]; acc = d > 3.0
        per = {o: fits[o]['rungs'][b]['log_evidence_selection']-fits[o]['rungs'][a]['log_evidence_selection'] for o in OBJECTS}
        steps.append(dict(step=f'{b} vs {a}', delta_log_evidence_sum=d, accepted=acc, per_object=per,
                          objects_favouring=sum(v > 0 for v in per.values())))
        if acc and chain:
            selected = b
        else:
            chain = False
    out = dict(rule=json.loads((Path(__file__).resolve().parent/'protocol_v02.json').read_text())['selection_rule'],
               created_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), total_log_evidence=tot, steps=steps, selected=selected,
               M1b_vs_M1_sum=tot['M1b']-tot['M1'],
               bic_sum={r: sum(fits[o]['rungs'][r]['BIC_neff'] for o in OBJECTS) for r in ladder+['M1b']},
               untempered_sum={r: sum(fits[o]['rungs'][r]['log_evidence_untempered'] for o in OBJECTS) for r in ladder+['M1b']})
    (OUT2/'selection.json').write_text(json.dumps(out, indent=1))
    print(json.dumps({k: out[k] for k in ('total_log_evidence', 'selected', 'M1b_vs_M1_sum', 'bic_sum')}, indent=1))
    for s in steps:
        print(s['step'], round(s['delta_log_evidence_sum'], 2), s['accepted'], {o: round(v, 1) for o, v in s['per_object'].items()})


# ---------------------------------------------------------------- held-out predictions (free inputs only)
def run_predict(obj, rungs=RUNGS):
    rng = np.random.default_rng(0)
    for rung in rungs:
        pts = dict(np.load(WORK2/'fit'/f'{obj}-{rung}-points.npz'))
        s2F = float(pts['noise_force_N'])**2
        missing = []
        for t, k in heldout_pokes(obj):
            pss = {nu: PokeSim(obj, t, k, nu) for nu in NUS}
            pss = {nu: p for nu, p in pss.items() if p.ok}
            mass = {nu: float(pts['w'][pts['nu'] == nu].sum()) for nu in pss}
            valid = [nu for nu, p in pss.items() if not p.snapped and mass[nu] > 0]; flag = None
            if not valid:
                valid = [nu for nu in pss if mass[nu] > 0]; flag = 'every nu snapped; snapped runs used' if valid else None
            if valid:
                m = np.isin(pts['nu'], valid); P_ = {a: v[m] for a, v in pts.items() if np.ndim(v) and len(v) == len(pts['w'])}
                P_['w'] = P_['w']/P_['w'].sum()
            else:
                valid = [nu for nu, p in pss.items() if not p.snapped] or list(pss)
                if not valid:
                    missing.append(f'{t}-p{k}'); continue
                flag = 'no simulated nu in posterior support: nu-marginal posterior applied uniformly over the available nu'
                base = {a: v for a, v in pts.items() if np.ndim(v) and len(v) == len(pts['w'])}
                P_ = {a: np.concatenate([base[a]]*len(valid)) for a in base}
                P_['nu'] = np.repeat(np.array(valid, float), len(base['w'])); P_['w'] = P_['w']/P_['w'].sum()
            q = poke(obj, t, k); p0 = pss[valid[0]]; sel = p0.frames > q['start_frame']
            Kf = len(p0.frames); force = np.zeros((Kf, 3)); surf = 0.0; Fpts = np.zeros((len(P_['w']), Kf, 3))
            for nu in valid:
                m = P_['nu'] == nu
                if not m.any():
                    continue
                ps = pss[nu]; w = P_['w'][m]; Wn = w.sum()
                kk = None if not np.isfinite(P_['k'][m]).all() else P_['k'][m]
                Fh, Dl, fl = forward(ps, P_['E'][m]/RF.E_REF, P_['taus'][m], P_['gs'][m], kk, want=('state',))
                Fpts[m] = Fh; force += np.einsum('p,pkj->kj', w, Fh)
                V = ps.z['surf_V'].astype(np.float64)
                if kk is None:
                    surf = surf+Wn*V
                else:
                    Sn = V[ps.t_idx].astype(np.float64); s_tr = P_['r'][m][:, None]*fl/kk[:, None]   # (p,K)
                    S_all = np.empty_like(V)
                    for n in range(Kf):
                        nwu = ps.node_weights(ps.dep[n]-Dl[:, n], w); nwd = ps.node_weights(np.array([ps.dep[n]]), Wn)
                        S_all[n] = Wn*V[n]+np.tensordot(nwu-nwd, Sn, 1)+float(w @ s_tr[:, n])*ps.a
                    surf = surf+S_all
            idx = rng.choice(len(P_['w']), N_RISK, p=P_['w'])
            mu = Fpts[idx][:, sel]; y = mu+rng.normal(0, np.sqrt(s2F), mu.shape)
            risk = float((np.sqrt(((y-force[sel])**2).sum(-1).mean(-1)) > 2.0).mean())
            meta = dict(method=rung, nu_used=valid, nu_weights=[float(P_['w'][P_['nu'] == nu].sum()) for nu in valid], n_points=int(len(P_['w'])),
                        retained_mass=float(pts['retained_mass']), flag=flag, snapped_nu=[nu for nu, p in pss.items() if p.snapped],
                        surface='elastic sims (no contact-loss lag)' if rung in ('M1', 'M1b') else 'sim surface shifted to body depth u (+ support translation for M3)')
            save_pred(rung, obj, t, k, p0.frames[sel], force[sel], meta, surf_V=surf[sel], surf_F=p0.surfF, risk=risk)
        (PRED2/rung/obj/'missing.json').write_text(json.dumps(dict(no_prediction=missing)))
        log(obj, rung, f'{len(heldout_pokes(obj))-len(missing)}/{len(heldout_pokes(obj))} predicted')


if __name__ == '__main__':
    cmd = sys.argv[1]
    if cmd == 'surftab':
        run_surftab(sys.argv[2])
    elif cmd == 'fit':
        run_fit(sys.argv[2])
    elif cmd == 'select':
        run_select()
    elif cmd == 'predict':
        run_predict(sys.argv[2], sys.argv[3].split(',') if len(sys.argv) > 3 else RUNGS)
