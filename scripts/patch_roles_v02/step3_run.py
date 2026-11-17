"""Step 3 runner (pre-specified settings, fixed before any step-3 result):
  * discrepancy hyperparameters (sigma_d, l_t, l_s) = posterior medians of a commanded-D0-only fit
    (0F0S, material + sigma_w + the three hyperparameters, 1024 x 10, seed 0), per (amplitude, world);
    never from D1 truth, D2 or any truth file;
  * force patches = box footprint (step3_model), supports = Gaussian (step 2);
  * sampler: configurations with a support patch 2048 particles x 12 moves, others 1024 x 10;
  * ONE risk definition for reporting and planning: risk = P(RMS_3D(true response - posterior mean) > DELTA),
    DELTA = 0.25 mm absolute, the true response drawn from the posterior particle mixture (+ per-particle
    Gaussian force-conditional spread for the hidden target) + the spatio-temporal discrepancy GP.
    Evaluated on a 512-particle systematic resample with 32 common discrepancy draws, both for the reported
    risk and inside the planner's pre-posterior (which reweights the same 512 particles).
Variants: oracle (D0+D1 commanded, 0F0S), oracleK (commanded, supports allowed: the GATE), hidden (D1 hidden).
"""
from __future__ import annotations
import argparse, json, math, time
from pathlib import Path
import numpy as np
import torch
from scipy.special import logsumexp
from scripts.joint_field_identification.continuous import SMCConfig, run_smc, systematic_resample
from scripts.joint_field_identification.hierarchy_surface import gaussian_mixture_quantiles, discrete_quantiles
from scripts.patch_roles_v02 import model as M, run as R2, step3_model as S3

RUNS = R2.RUNS/'step3'
DELTA = 2.5e-4
N_THIN, N_DISC, N_Y = 512, 32, 64
BUDGET = dict(sup=(2048, 12), nosup=(1024, 10))
CTX = {}


# ------------------------------------------------------------------ discrepancy hyperparameters from D0 only
def fit_disc(bank, case, out):
    p = out.parent/f'disc-{case["amp"]}-{case["world"]}.json'
    if p.exists():
        return json.loads(p.read_text())
    pidx = {x: 1+i for i, x in enumerate(M.BK.PRESS_X)}
    ep0 = [dict(name='D0', y=case['y']['D0'], known=[('step', 0, 1.0), ('ramp', pidx[0.0], case['force'])])]
    pm0 = M.PatchModel(bank, ep0); dev = pm0.dev; dt = torch.float64
    Y = torch.as_tensor(case['y']['D0'], dtype=dt, device=dev).reshape(24, 20, 3)
    tk = torch.as_tensor(M.TF[1:], dtype=dt, device=dev); X = torch.as_tensor(S3.B.MARKERS, dtype=dt, device=dev)
    _, b0 = pm0.layout(0, 0)
    bounds = np.vstack([b0, [[math.log(1e-5), math.log(1e-3)], [math.log(.01), math.log(.5)], [math.log(.01), math.log(.3)]]])

    def ll(th, batch=128):
        out_ = []
        for i in range(0, len(th), batch):
            t = th[i:i+batch]; Bn = len(t)
            with torch.no_grad():
                d = pm0.operators(t[:, :5], 0, 0, list(range(60)))['d_D0'].reshape(Bn, 24, 20, 3)
                sw2, sd2 = (torch.as_tensor(np.exp(2*t[:, k]), dtype=dt, device=dev) for k in (4, 5))
                lt_, ls_ = (torch.as_tensor(np.exp(t[:, k]), dtype=dt, device=dev) for k in (6, 7))
                r = np.sqrt(3)*(tk[None, :, None]-tk[None, None, :]).abs()/lt_[:, None, None]
                Ct = (1+r)*torch.exp(-r); Cs = torch.exp(-.5*((X[:, None]-X[None])**2).sum(-1)[None]/ls_[:, None, None]**2)
                et, Ut = torch.linalg.eigh(Ct); es, Us = torch.linalg.eigh(Cs+1e-9*torch.eye(20, dtype=dt, device=dev))
                v = sw2[:, None, None]+sd2[:, None, None]*et.clamp_min(0)[:, :, None]*es.clamp_min(0)[:, None, :]
                rt = torch.einsum('bgi,bmj,bgmk->bijk', Ut, Us, Y[None]-d)
                l = -.5*((rt**2/v[..., None]).sum((1, 2, 3))+3*torch.log(v).sum((1, 2))+1440*math.log(2*math.pi))
            out_.append(l.cpu().numpy())
        return np.concatenate(out_)
    t0 = time.time()
    res = run_smc(ll, bounds=bounds, config=SMCConfig(particles=1024, moves=10, batch_size=1024, maximum_stages=300), seed=0,
                  target_id=f'disc-{case["amp"]}-{case["world"]}')
    th = res.theta; q = lambda k: [float(np.exp(np.quantile(th[:, k], x))) for x in (.05, .5, .95)]
    hyp = dict(sigma_w_mm=[1e3*x for x in q(4)], sigma_d_mm=[1e3*x for x in q(5)], len_t_s=q(6), len_s_m=q(7),
               sigma_d=q(5)[1], len_t=q(6)[1], len_s=q(7)[1], log_evidence=res.log_evidence, seconds=time.time()-t0)
    out.parent.mkdir(parents=True, exist_ok=True); p.write_text(json.dumps(hyp, indent=1)); print('disc fit', hyp, flush=True)
    return hyp


def fit_disc2(bank, case, out, episodes_used=('D0', 'D1')):
    """Two-component discrepancy hyperparameters (a, b, l_t, l_s) + material + sigma_w from the COMMANDED
    episodes (dev mode: D0 and D1 with their commands), 0F0S, 1024 x 10, seed 0."""
    tag = '+'.join(episodes_used)
    p = out.parent/f'disc2-{case["amp"]}-{case["world"]}-{tag}.json'
    if p.exists():
        return json.loads(p.read_text())
    pidx = {x: 1+i for i, x in enumerate(M.BK.PRESS_X)}; xs = dict(D0=0.0, D1=0.05)
    eps_ = [dict(name=e, y=case['y'][e], known=[('step', 0, 1.0), ('ramp', pidx[xs[e]], case['force'])]) for e in episodes_used]
    pm0 = M.PatchModel(bank, eps_); dev = pm0.dev; dt = torch.float64
    Ys = {e['name']: torch.as_tensor(e['y'], dtype=dt, device=dev).reshape(24, 20, 3) for e in eps_}
    tk = torch.as_tensor(M.TF[1:], dtype=dt, device=dev); X = torch.as_tensor(S3.B.MARKERS, dtype=dt, device=dev)
    _, b0 = pm0.layout(0, 0)
    bounds = np.vstack([b0, [[math.log(1e-5), math.log(1e-3)], [math.log(1e-5), math.log(1e-3)], [math.log(.01), math.log(.5)], [math.log(.02), math.log(.3)]]])
    I20 = torch.eye(20, dtype=dt, device=dev)

    def ll(th, batch=128):
        res_ = []
        for i in range(0, len(th), batch):
            t = th[i:i+batch]
            with torch.no_grad():
                op = pm0.operators(t[:, :5], 0, 0, list(range(60)))
                sw2, a2, b2 = (torch.as_tensor(np.exp(2*t[:, k]), dtype=dt, device=dev) for k in (4, 5, 6))
                lt_, ls_ = (torch.as_tensor(np.exp(t[:, k]), dtype=dt, device=dev) for k in (7, 8))
                r = np.sqrt(3)*(tk[None, :, None]-tk[None, None, :]).abs()/lt_[:, None, None]
                et, Ut = torch.linalg.eigh((1+r)*torch.exp(-r))
                S = a2[:, None, None]*I20+b2[:, None, None]*torch.exp(-.5*((X[:, None]-X[None])**2).sum(-1)[None]/ls_[:, None, None]**2)
                es, Us = torch.linalg.eigh(S)
                v = sw2[:, None, None]+et.clamp_min(0)[:, :, None]*es.clamp_min(0)[:, None, :]
                l = 0.
                for e in eps_:
                    rt = torch.einsum('bgi,bmj,bgmk->bijk', Ut, Us, Ys[e['name']][None]-op['d_'+e['name']].reshape(-1, 24, 20, 3))
                    l = l-.5*((rt**2/v[..., None]).sum((1, 2, 3))+3*torch.log(v).sum((1, 2))+1440*math.log(2*math.pi))
            res_.append(l.cpu().numpy())
        return np.concatenate(res_)
    t0 = time.time()
    res = run_smc(ll, bounds=bounds, config=SMCConfig(particles=1024, moves=10, batch_size=1024, maximum_stages=300), seed=0,
                  target_id=f'disc2-{case["amp"]}-{case["world"]}-{tag}')
    th = res.theta; q = lambda k: [float(np.exp(np.quantile(th[:, k], x))) for x in (.05, .5, .95)]
    hyp = dict(episodes=list(episodes_used), sigma_w_mm=[1e3*x for x in q(4)], a_mm=[1e3*x for x in q(5)], b_mm=[1e3*x for x in q(6)],
               len_t_s=q(7), len_s_m=q(8), a=q(5)[1], b=q(6)[1], len_t=q(7)[1], len_s=q(8)[1],
               sigma_d=float(np.sqrt(q(5)[1]**2+q(6)[1]**2)), log_evidence=res.log_evidence, seconds=time.time()-t0)
    p.parent.mkdir(parents=True, exist_ok=True); p.write_text(json.dumps(hyp, indent=1)); print('disc2 fit', hyp, flush=True)
    return hyp


def fit_disc3(bank, case, out):
    """Hyperparameters of the episode-shared discrepancy (Step3ModelShared) from the COMMANDED D0+D1 episodes
    (0F0S, material + sigma_w + a,b,c,d,l_t,l_s), 1024 x 10, seed 0.  Dev-mode choice: uses D1's command."""
    p = out.parent/f'disc3-{case["amp"]}-{case["world"]}.json'
    if p.exists():
        return json.loads(p.read_text())
    pidx = {x: 1+i for i, x in enumerate(M.BK.PRESS_X)}; xs = dict(D0=0.0, D1=0.05)
    eps_ = [dict(name=e, y=case['y'][e], known=[('step', 0, 1.0), ('ramp', pidx[xs[e]], case['force'])]) for e in ('D0', 'D1')]
    pm0 = M.PatchModel(bank, eps_); dev = pm0.dev; dt = torch.float64
    Y = torch.stack([torch.as_tensor(e['y'], dtype=dt, device=dev).reshape(24, 20, 3) for e in eps_])
    tk = torch.as_tensor(M.TF[1:], dtype=dt, device=dev); X = torch.as_tensor(S3.B.MARKERS, dtype=dt, device=dev)
    _, b0 = pm0.layout(0, 0); lo, hi = math.log(3e-6), math.log(1e-3)
    bounds = np.vstack([b0, [[lo, hi]]*4, [[math.log(.01), math.log(.5)], [math.log(.02), math.log(.3)]]])
    Hm = torch.as_tensor(S3.helmert(2), dtype=dt, device=dev)

    def ll(th, batch=128):
        res_ = []
        for i in range(0, len(th), batch):
            t = th[i:i+batch]
            with torch.no_grad():
                op = pm0.operators(t[:, :5], 0, 0, list(range(60)))
                sw2, a2, b2, c2, d2 = (torch.as_tensor(np.exp(2*t[:, k]), dtype=dt, device=dev) for k in (4, 5, 6, 7, 8))
                lt_, ls_ = (torch.as_tensor(np.exp(t[:, k]), dtype=dt, device=dev) for k in (9, 10))
                r = np.sqrt(3)*(tk[None, :, None]-tk[None, None, :]).abs()/lt_[:, None, None]
                et, Ut = torch.linalg.eigh((1+r)*torch.exp(-r)); et = et.clamp_min(0)
                es, Us = torch.linalg.eigh(torch.exp(-.5*((X[:, None]-X[None])**2).sum(-1)[None]/ls_[:, None, None]**2)); es = es.clamp_min(0)
                R = torch.stack([torch.einsum('bgi,bmj,bgmk->bijk', Ut, Us, Y[k][None]-op['d_'+eps_[k]['name']].reshape(-1, 24, 20, 3)) for k in range(2)])
                Rk = torch.einsum('ke,ebijc->kbijc', Hm, R)
                l = 0.
                for k in range(2):
                    v = sw2[:, None, None]+et[:, :, None]*(a2[:, None, None]+b2[:, None, None]*es[:, None, :]+(2 if k == 0 else 0)*(c2[:, None, None]+d2[:, None, None]*es[:, None, :]))
                    l = l-.5*((Rk[k]**2/v[..., None]).sum((1, 2, 3))+3*torch.log(v).sum((1, 2))+1440*math.log(2*math.pi))
            res_.append(l.cpu().numpy())
        return np.concatenate(res_)
    t0 = time.time()
    res = run_smc(ll, bounds=bounds, config=SMCConfig(particles=1024, moves=10, batch_size=1024, maximum_stages=300), seed=0,
                  target_id=f'disc3-{case["amp"]}-{case["world"]}')
    th = res.theta; q = lambda k: [float(np.exp(np.quantile(th[:, k], x))) for x in (.05, .5, .95)]
    hyp = dict(sigma_w_mm=[1e3*x for x in q(4)], a_mm=[1e3*x for x in q(5)], b_mm=[1e3*x for x in q(6)], c_mm=[1e3*x for x in q(7)],
               d_mm=[1e3*x for x in q(8)], len_t_s=q(9), len_s_m=q(10), a=q(5)[1], b=q(6)[1], c=q(7)[1], d=q(8)[1], len_t=q(9)[1], len_s=q(10)[1],
               log_evidence=res.log_evidence, seconds=time.time()-t0)
    hyp['sigma_d'] = float(np.sqrt(hyp['b']**2+hyp['d']**2)); hyp['pred_sd'] = hyp['sigma_d']
    p.parent.mkdir(parents=True, exist_ok=True); p.write_text(json.dumps(hyp, indent=1)); print('disc3 fit', hyp, flush=True)
    return hyp


# ------------------------------------------------------------------ unified risk
def disc_draws_probes(rng, n, hyp):
    Lt = np.linalg.cholesky(M.matern(M.TF[1:], hyp['len_t'])+1e-10*np.eye(24))
    Ls = np.linalg.cholesky(S3.se_kernel(S3.B.PROBES, hyp['len_s'])+1e-8*np.eye(27))
    Z = rng.standard_normal((n, 24, 27, 3))
    return hyp.get('pred_sd', hyp['sigma_d'])*np.einsum('gh,pq,nhqk->ngpk', Lt, Ls, Z).reshape(n, -1)


def risk_fn(Q, Vsd, E, Z):
    """Q (N,D), Vsd (N,D), E (K,D), Z (K,D) torch -> f(Wp (Y,N)) = sum_n Wp P_k[RMS(Q_n+Vsd_n Z_k+E_k - Wp@Q) > DELTA]."""
    def f(Wp):
        out = []
        for y in range(Wp.shape[0]):
            mean = Wp[y]@Q
            x = Q[:, None]+Vsd[:, None]*Z[None]+E[None]-mean[None, None]          # (N,K,D)
            rms = torch.sqrt((x**2).sum(-1)/(25*27))
            out.append((Wp[y]*(rms > DELTA).float().mean(1)).sum())
        return torch.stack(out)
    return f


def thin(w, rng):
    return systematic_resample(w, rng)[np.linspace(0, len(w)-1, N_THIN).astype(int)] if len(w) > N_THIN else np.arange(len(w))


def target_metrics(w, q, var, truth, sigma_d, rng, name, n_mc=None):
    hyp = CTX['hyp']; dev = 'cuda'; tt = lambda a: torch.as_tensor(a, dtype=torch.float32, device=dev)
    N = len(w); qf = q.reshape(N, -1); vf = var.reshape(N, -1); tf = truth.reshape(-1)
    mean = w@qf; tot = vf+hyp.get('pred_sd', hyp['sigma_d'])**2
    lo, hi = gaussian_mixture_quantiles(w, qf, tot)
    tot_var = w@(tot+(qf-mean)**2)
    err = (mean-tf).reshape(24, 27, 3); rms = float(np.sqrt((err**2).sum(-1).sum()/(25*27))*1e3)
    idx = thin(w, rng); E = tt(disc_draws_probes(rng, N_DISC, hyp)); Z = tt(rng.standard_normal((N_DISC, qf.shape[1])))
    f = risk_fn(tt(qf[idx]), tt(np.sqrt(vf[idx])), E, Z)
    risk = float(f(torch.full((1, len(idx)), 1/len(idx), device=dev))[0])
    if name.startswith('D2'):
        CTX['d2'] = dict(idx=idx, E=E, Z=Z, f=f)
    return dict(target=name, rmse_mm=rms, truth_rms_mm=float(np.sqrt((truth.reshape(24, 27, 3)**2).sum(-1).sum()/(25*27))*1e3),
                pred_sd_mm=float(np.sqrt(tot_var.mean())*1e3), cover90=float(np.mean((tf >= lo) & (tf <= hi))),
                z_rms=float(np.sqrt(np.mean((mean-tf)**2/tot_var))), delta_mm=DELTA*1e3, risk=risk, exceeded=bool(rms > DELTA*1e3))


def design(pm, w, cat, sigma_d, rng, cands, n_y=N_Y):
    """Pre-posterior of the SAME risk (D2 target) for each candidate calibrated press, on the same 512 particles."""
    d = CTX['d2']; idx = d['idx']; f = d['f']; dev = pm.dev; n = len(idx)
    W0 = torch.full((1, n), 1/n, device=dev); r0 = float(f(W0)[0])
    v = torch.as_tensor(cat('v')[idx], dtype=torch.float64, device=dev)                  # (n,24,20)
    allx = [x for x in R2.PROBE_X if any(e['name'] == f'C{x:+.2f}' for e in pm.episodes)]
    cand = cat('cand')[idx]; names = [x for x in cands if x in allx]
    rows = [dict(action='proceed', J=r0, expected_risk=r0)]
    rot = lambda X: torch.einsum('gi,mj,bgmk->bijk', pm.Ut, pm.Us, X.reshape(-1, 24, 20, 3))
    for x in names:
        P = rot(torch.as_tensor(cand[:, allx.index(x)], dtype=torch.float64, device=dev))             # (n,24,20,3)
        j = rng.integers(0, n, n_y); z = torch.as_tensor(rng.standard_normal((n_y, 24, 20, 3)), dtype=torch.float64, device=dev)
        y = P[j]+v[j].sqrt()[..., None]*z
        rs = []
        for k in range(n_y):
            ll = -.5*(((y[k][None]-P)**2/v[..., None]).sum((1, 2, 3))+3*torch.log(v).sum((1, 2)))
            rs.append(f(torch.softmax(ll, 0)[None].float())[0])
        r = float(torch.stack(rs).mean())
        rows.append(dict(action=f'press x={x:+.2f}', x=x, cost=R2.C_PROBE, expected_risk=r, J=R2.C_PROBE+r))
    best = min(rows, key=lambda r: r['J']); probes = [r for r in rows if 'x' in r]
    return dict(rows=rows, decision=best['action'], best_probe=min(probes, key=lambda r: r['expected_risk']) if probes else None,
                risk_definition=f'P(RMS_3D > {DELTA*1e3:.2f} mm), 512-particle resample, 32 common discrepancy draws')


R2.target_metrics = target_metrics
R2.design = design


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--amp', default='low'); ap.add_argument('--world', default='Bright'); ap.add_argument('--variant', default='hidden')
    ap.add_argument('--seed', type=int, default=0); ap.add_argument('--K', type=int, default=2)
    ap.add_argument('--probes', type=float, nargs='*', default=[]); ap.add_argument('--tag', default='')
    ap.add_argument('--hyp', default='d0', choices=['d0', 'infer', 'freq', 'two', 'shared', 'vol'], help='d0: fixed from commanded-D0 fit; infer: joint (fallback)')
    ap.add_argument('--pred-sd', default='smooth', choices=['smooth', 'all'], help='discrepancy parts added to the predictive of the true response')
    ap.add_argument('--budget-scale', type=float, default=1.0, help='smoke tests only')
    a = ap.parse_args()
    torch.set_num_threads(4)
    name = f'{a.amp}-{a.world}-{a.variant}' + ''.join(f'-P{x:+.2f}' for x in a.probes) + ({'d0': '', 'infer': '-infer', 'freq': '-freq', 'two': '-two', 'shared': '-shared', 'vol': '-vol'}[a.hyp]) + a.tag
    out = RUNS/'cases'/name; out.mkdir(parents=True, exist_ok=True)
    case = R2.load_case(a.amp, a.world, a.probes)
    eps = R2.episodes(a.variant, case, a.probes)
    bank = M.Bank('cuda', supports=a.variant != 'oracle')
    hyp = fit_disc(bank, case, RUNS/'disc'/'x'); CTX['hyp'] = hyp
    if a.hyp in ('shared', 'vol'):
        hyp = dict(fit_disc3(bank, case, RUNS/'disc'/'x'))
        if a.pred_sd == 'all':
            hyp['pred_sd'] = float(np.sqrt(hyp['a']**2+hyp['b']**2+hyp['c']**2+hyp['d']**2))
        CTX['hyp'] = hyp
        pm = (S3.Step3ModelShared if a.hyp == 'shared' else S3.Step3ModelVol)(bank, eps, hyp); tid = R2.data_hash(eps, f'step3-{a.hyp}-'+'-'.join(f"{hyp[k]:.6g}" for k in 'abcd'))
    elif a.hyp == 'two':
        hyp = dict(fit_disc2(bank, case, RUNS/'disc'/'x')); hyp['pred_sd'] = hyp['b']; CTX['hyp'] = hyp   # truth predictive: smooth part only
        pm = S3.Step3Model2(bank, eps, hyp); tid = R2.data_hash(eps, f"step3-two-{hyp['a']:.6g}-{hyp['b']:.6g}-{hyp['len_t']:.6g}-{hyp['len_s']:.6g}")
    elif a.hyp in ('infer', 'freq'):
        cls = S3.Step3ModelInfer if a.hyp == 'infer' else S3.Step3ModelFreq
        pm = cls(bank, eps, hyp); tid = R2.data_hash(eps, f'step3-{a.hyp}-box')
    else:
        pm = S3.Step3Model(bank, eps, hyp['sigma_d'], hyp['len_t'], hyp['len_s'])
        tid = R2.data_hash(eps, f"step3-{hyp['sigma_d']:.6g}-{hyp['len_t']:.6g}-{hyp['len_s']:.6g}-box")
    t0 = time.time(); runs = {}
    for c in R2.configs(a.K, a.variant):
        npart, mv = BUDGET['sup' if c[1] else 'nosup']; npart = max(64, int(npart*a.budget_scale)); mv = max(1, int(mv*a.budget_scale))
        runs[c] = R2.run_config(pm, *c, seed=a.seed*1000+17*c[0]+c[1], path=out/f'{c[0]}F{c[1]}S-s{a.seed}.npz', particles=npart, moves=mv, tid=tid)
    smc = time.time()-t0; t1 = time.time()
    if a.hyp in ('infer', 'freq'):      # predictive/planner use the posterior-median hyperparameters of the evidence-weighted configs
        lp = R2.config_logprior(a.K, list(runs)) if a.variant != 'oracle' else {(0, 0): 0.}
        cm = max(runs, key=lambda c: lp[c]+runs[c]['logZ']); th = runs[cm]['theta']
        hyp = dict(hyp, sigma_d=float(np.exp(np.median(th[:, -3]))), len_t=float(np.exp(np.median(th[:, -2]))), len_s=float(np.exp(np.median(th[:, -1]))),
                   source=f'joint posterior median ({cm[0]}F{cm[1]}S)')
        CTX['hyp'] = hyp; pm.sigma_d = hyp['sigma_d']
        Ct = M.matern(M.TF[1:], hyp['len_t']); lt, Ut = np.linalg.eigh(Ct); pm.Ut = torch.as_tensor(Ut, dtype=torch.float64, device=pm.dev)
        ls, Us = np.linalg.eigh(S3.se_kernel(S3.B.MARKERS, hyp['len_s'])+1e-9*np.eye(20)); pm.Us = torch.as_tensor(Us, dtype=torch.float64, device=pm.dev)
    res = R2.analyze(pm, case, runs, a.K, a.variant, hyp['sigma_d'], name, a.seed)
    res.update(case=name, seed=a.seed, K=a.K, disc=hyp, budget=BUDGET, budget_scale=a.budget_scale,
               compute=dict(smc_seconds=smc, analysis_seconds=time.time()-t1,
                            per_config={f'{c[0]}F{c[1]}S': dict(seconds=r['seconds'], stages=r['stages'], evals=r['evals']) for c, r in runs.items()}))
    (out/(f'result-s{a.seed}.json' if a.pred_sd == 'smooth' else f'result-predall-s{a.seed}.json')).write_text(json.dumps(res, indent=1, default=float))
    print(json.dumps({k: v for k, v in res.items() if k not in ('design',)}, indent=1, default=float)[:3500])
    print('design', json.dumps(res['design'], default=float))


if __name__ == '__main__':
    main()
