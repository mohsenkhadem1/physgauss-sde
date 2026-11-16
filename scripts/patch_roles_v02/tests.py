"""Identity tests for the step-2 model against direct sparse linear Newmark on the same coarse mesh.
1. known loads (gravity + D1 press): mode-acceleration response == fem.newmark(linear) at all 47 points
2. one force patch with a random hat history, 3 modes
3. one support patch (rank-3 spring) + gravity + press + force patch (full system with the spring in K)
4. Woodbury marginal == dense Gaussian log density with the full (white + Matern discrepancy + force) covariance
Writes runs/patch-roles-v02/tests.json"""
import json, math, sys
import numpy as np
import torch
from pathlib import Path
R = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(R/'scripts/fig1_v01')); sys.path.insert(0, str(R/'scripts/stem_trial_v01'))
import fem, block as B
from scripts.patch_roles_v02 import model as M, bank as BK


def direct(G, mat, E, loads, spring=None):
    """loads: list of (vector ndof, history fn). spring: (k, W3 (ndof,3)). Returns rows (25,141)."""
    xc, k, nu = mat; m = G.m; fr = m.free
    model = fem.Elastic(m, B.modulus(m, xc, contrast=k, e_soft=E), nu, B.RHO)
    K = model.linear_K().tocsr()
    if spring is not None:
        import scipy.sparse as sp
        kk, W3 = spring; K = K+sp.csr_matrix(kk*W3@W3.T)
    K = K[fr][:, fr]; Mm = model.M.tocsr()[fr][:, fr]
    c0 = 4/M.DT**2; fac = fem._factor(K+c0*Mm); facM = fem._factor(Mm)
    f = lambda t: sum(v[fr]*h(t) for v, h in loads)
    n = len(fr); u = np.zeros(n); v = np.zeros(n); a = facM(f(0.)-K@u)
    out = [np.zeros(n)]
    for s in range(1, 241):
        t = s*M.DT; pc = c0*u+4/M.DT*v+a
        x = fac(f(t)+Mm@pc); an = c0*x-pc; v = v+M.DT/2*(a+an); a = an; u = x
        if s % 10 == 0: out.append(u.copy())
    U = np.zeros((25, 3*m.n)); U[:, fr] = np.stack(out)
    H = G.H
    return np.einsum('pn,fnj->fpj', H, U.reshape(25, -1, 3)).reshape(25, -1)


def main():
    torch.set_num_threads(4)
    bank = M.Bank('cuda'); G = bank.geo
    imat = 37+1+7*4+2      # nu=.4, xc=+.06, contrast 8
    mat = bank.mats[imat]; E = 6.3e4; res = {}
    th0 = np.array([math.log(E), 7.5, 1/9+2.5, 1.5, math.log(1e-4)])
    assert M.PatchModel.material_index(th0[None])[0] == imat
    ramp = lambda t: .5*(1-math.cos(math.pi*min(t/.25, 1)))
    ep = [dict(name='D1', y=np.zeros((24, 60)), known=[('step', 0, 1.0), ('ramp', bank.press_index[0.05], 30.0)], hidden=True)]
    pm = M.PatchModel(bank, ep, sigma_d=1.5e-4)
    rows = list(range(141))
    # 1 known loads
    with torch.no_grad():
        op = pm.operators(th0[None], 0, 0, rows)
    mine = op['d_D1'][0].cpu().numpy()
    ref = direct(G, mat, E, [(G.g, lambda t: 1.), (BK.press_load(G.m, 0.05)*30, ramp)])[1:]
    res['known_rel_err'] = float(np.abs(mine-ref).max()/np.abs(ref).max())
    # 2 force patch
    rng = np.random.default_rng(0)
    pf = np.array([0.3, 0.4, math.log(0.012)]); hist = rng.standard_normal((24, 3))
    th = np.concatenate([th0, [0.], pf])[None]
    with torch.no_grad():
        op = pm.operators(th, 1, 0, rows)
        w = pm.footprint(pf[:1], pf[1:2], pf[2:3])[0].cpu().numpy()
    Gm = op['G'][0].cpu().numpy()                                   # (24,141,72) cols (mode, hat)
    mine = np.einsum('grq,q->gr', Gm, hist.T.ravel())
    vec = np.zeros((3*G.m.n, 3));
    for e in range(3): vec[3*G.S+e, e] = w
    hf = lambda t: np.interp(t*48, np.arange(25), np.r_[np.zeros((1, 3)), hist], left=0, right=0) if False else None
    loads = [(vec[:, e], (lambda e: lambda t: float(np.interp(t*48, np.arange(25), np.r_[0., hist[:, e]])))(e)) for e in range(3)]
    ref = direct(G, mat, E, loads)[1:]
    res['force_rel_err'] = float(np.abs(mine-ref).max()/np.abs(ref).max())
    # 3 support + known + force
    ps = np.array([0.55, 0.3, math.log(0.015), math.log(3e3)])
    th = np.concatenate([th0, [0.], pf, ps])[None]
    with torch.no_grad():
        op = pm.operators(th, 1, 1, rows)
        ws = pm.footprint(ps[:1], ps[1:2], ps[2:3])[0].cpu().numpy()
    mine = op['d_D1'][0].cpu().numpy()+np.einsum('grq,q->gr', op['G'][0].cpu().numpy(), hist.T.ravel())
    W3 = np.zeros((3*G.m.n, 3))
    for e in range(3): W3[3*G.S+e, e] = ws
    ref = direct(G, mat, E, [(G.g, lambda t: 1.), (BK.press_load(G.m, 0.05)*30, ramp)]+loads, spring=(math.exp(ps[3]), W3))[1:]
    res['support_rel_err'] = float(np.abs(mine-ref).max()/np.abs(ref).max())
    res['support_effect_rel'] = float(np.abs(ref-direct(G, mat, E, [(G.g, lambda t: 1.), (BK.press_load(G.m, 0.05)*30, ramp)]+loads)[1:]).max()/np.abs(ref).max())
    # 4 marginal vs dense
    y = (ref[:, :60]+rng.standard_normal((24, 60))*1e-4)
    ep[0]['y'] = y; pm = M.PatchModel(bank, ep, sigma_d=1.5e-4)
    th[0, 5] = math.log(0.5)
    with torch.no_grad():
        op = pm.operators(th, 1, 1, list(range(60)))
        ll, _ = pm.marginal(th, 1, 1, op)
    Gd = op['G'][0].cpu().numpy().reshape(24*60, -1); d = op['d_D1'][0].cpu().numpy().ravel()
    Lam = (0.5*pm.SIGMA_F)**2*np.kron(np.eye(3), M.matern(M.TF[1:], pm.LEN_F))
    Ct = M.matern(M.TF[1:], 0.05)
    Rn = 1e-8*np.eye(24*60)+1.5e-4**2*np.kron(Ct, np.eye(60))
    C = Rn+Gd@Lam@Gd.T
    from scipy.stats import multivariate_normal
    ref_ll = multivariate_normal(d, C, allow_singular=False).logpdf(y.ravel())
    res['marginal_abs_err'] = float(abs(ll.item()-ref_ll)); res['marginal_value'] = float(ref_ll)
    res['note'] = 'rel errs vs full sparse Newmark on the same mesh; residual attributed to 80-mode truncation'
    res['passed'] = bool(res['known_rel_err'] < 2e-2 and res['force_rel_err'] < 2e-2 and res['support_rel_err'] < 2e-2 and res['marginal_abs_err'] < 1e-4)
    print(json.dumps(res, indent=1))
    (R/'runs/patch-roles-v02/tests.json').write_text(json.dumps(res, indent=1))


if __name__ == '__main__':
    main()
