"""Patch-role model on the Figure-1 block (step 2): hidden commands, dynamics, inclusion material.

Body: Fig-1 block, linear T10 on the Fig-1 ours mesh (12,5,6), bottom glued (known support), rho known,
gravity on from t=0 (known load in every episode).  Material = Fig-1 ours inclusion family
(xc, contrast or none, nu on the Fig-1 grid; soft modulus E continuous).  Exact linear elastodynamics
(undamped Newmark, dt=1/480, the truth's integrator) by mode acceleration:

    u(t) = Khat^-1 f a(t)/E + sum_{k<=80} phi_k phi_k^T f [eta_k(t) - a(t)/(E lamhat_k)],
    eta_k'' + E lamhat_k eta_k = a(t)  integrated with the same Newmark recursion,

so the response is exact for the 80 retained modes and quasi-static for the rest (bank.py).

Patches (K exchangeable; role in {force, support, off}) live on all five non-glued faces: centre (u,b) in
[0,1]^2 maps area-uniformly onto left/top/right/front/back, log width s in [6,30] mm, normalised Gaussian
surface footprint w (ambient distance).
  force patch:   traction w(x) a_j(t), a_j(t) in R^3 piecewise linear on the 24 frame knots (a(0)=0:
                 the episode starts at rest).  a ~ N(0, (sigma_F tau)^2 Matern32(l_F) x I), hierarchical tau.
  support patch: rank-3 spring k_j on the footprint-averaged displacement  (energy k/2 |w^T u|^2),
                 present in EVERY episode (a physical constraint), reaction piecewise linear on frames,
                 coupled exactly at the frame knots (collocation) through the static compliance and modes.
Observation: RGB-triangulated marker displacements (20 markers x 3) at frames 1..24.
Noise: white sigma_w (inferred, log-uniform) + optional model discrepancy sigma_d Matern32(l_d) in time,
independent across marker coordinates and episodes.  The force amplitudes are integrated out exactly
(Woodbury in amplitude space after exact time-whitening).
"""
from __future__ import annotations
import math
from pathlib import Path
import numpy as np
import torch
from scripts.patch_roles_v02 import bank as BK

DT, SAVE, NFR = 1/480, 10, 25
TF = np.arange(NFR)/48.
RAMP = 0.25
FACES = [  # name, area, fixed axis/value, (axis1, lo, hi), (axis2, lo, hi)
    ('left', 0.012, (0, -0.12), (2, -0.06, 0.06), (1, 0.53, 0.63)),
    ('top', 0.0288, (1, 0.63), (0, -0.12, 0.12), (2, -0.06, 0.06)),
    ('right', 0.012, (0, 0.12), (2, -0.06, 0.06), (1, 0.53, 0.63)),
    ('front', 0.024, (2, 0.06), (0, -0.12, 0.12), (1, 0.53, 0.63)),
    ('back', 0.024, (2, -0.06), (0, -0.12, 0.12), (1, 0.53, 0.63)),
]
AREA_CUM = np.cumsum([0.]+[f[1] for f in FACES])/sum(f[1] for f in FACES)


def matern(t, length):
    d = np.abs(t[:, None]-t[None]); r = np.sqrt(3)*d/length
    return (1+r)*np.exp(-r)


def surface_point(u, b):
    """(u,b) in [0,1]^2 -> centre on the 5 loadable faces (area-uniform). numpy arrays."""
    u = np.clip(np.asarray(u, float), 0, 1-1e-12); b = np.asarray(b, float)
    fi = np.searchsorted(AREA_CUM, u, side='right')-1
    out = np.zeros(u.shape+(3,)); face = np.empty(u.shape, object)
    for i, (nm, _, (ax0, v0), (a1, l1, h1), (a2, l2, h2)) in enumerate(FACES):
        s = fi == i
        if not s.any():
            continue
        loc = (u[s]-AREA_CUM[i])/(AREA_CUM[i+1]-AREA_CUM[i])
        out[s, ax0] = v0; out[s, a1] = l1+loc*(h1-l1); out[s, a2] = l2+b[s]*(h2-l2); face[s] = nm
    return out, face


def surface_uv(point):
    """Inverse of surface_point for a point on a face (used for true patch / probe locations)."""
    p = np.asarray(point, float)
    for i, (nm, _, (ax0, v0), (a1, l1, h1), (a2, l2, h2)) in enumerate(FACES):
        if abs(p[ax0]-v0) < 1e-9:
            loc = (p[a1]-l1)/(h1-l1)
            return AREA_CUM[i]+loc*(AREA_CUM[i+1]-AREA_CUM[i]), (p[a2]-l2)/(h2-l2)
    raise ValueError(point)


def time_inputs():
    """Input histories at all Newmark steps: 24 frame hats (knots 1..24), gravity step, smooth ramp."""
    ts = np.arange(round(TF[-1]/DT)+1)*DT
    hats = np.clip(1-np.abs(ts[:, None]*48-np.arange(1, NFR)[None]), 0, None)
    step = np.ones((len(ts), 1)); s = np.clip(ts/RAMP, 0, 1); ramp = (.5*(1-np.cos(np.pi*s)))[:, None]
    return np.concatenate([hats, step, ramp], 1)            # (241, 26)


COL_STEP, COL_RAMP, NH = 24, 25, 24


class Bank:
    def __init__(self, device='cuda', supports=True):
        self.dev = torch.device(device); dt = torch.float64
        files = [BK.OUT/f'mat{i:03d}.npz' for i in range(len(BK.materials()))]
        zs = [np.load(f) for f in files]
        T = lambda k, d=dt: torch.as_tensor(np.stack([z[k] for z in zs]), dtype=d, device=self.dev)
        self.lam, self.HPhi, self.PhiS, self.XoS = T('lam'), T('HPhi'), T('PhiS'), T('XoS')
        self.KO, self.KS, self.PhiK = T('KO'), T('KS'), T('PhiK')
        self.Css = T('Css', torch.float32) if supports else None
        self.mats = BK.materials()
        G = BK.Geometry(); self.geo = G
        self.q = torch.as_tensor(G.q, dtype=dt, device=self.dev); self.qw = torch.as_tensor(G.qw, dtype=dt, device=self.dev)
        r, c, v, shp = G.Nq; Nq = np.zeros(shp); np.add.at(Nq, (r, c), v)
        self.Nq = torch.as_tensor(Nq, dtype=dt, device=self.dev)
        self.inputs = torch.as_tensor(time_inputs(), dtype=dt, device=self.dev)
        self.nS = len(G.S); self.m = self.lam.shape[1]
        self.press_index = {x: 1+i for i, x in enumerate(BK.PRESS_X)}


class PatchModel:
    """theta layout for config (nF, nS):
       logE, xc_u in [0,9), c_u in [0, 4+1/9) (<1/9: no inclusion), nu_u in [0,3), log sigma_w (m),
       [log tau if nF>0], force j: (u, b, log s), support j: (u, b, log s, log k)."""
    SIGMA_F = 10.0; LEN_F = 0.15

    def __init__(self, bank: Bank, episodes, sigma_d=0.0, len_d=0.05, rows_obs=None):
        self.b = bank; self.dev = bank.dev; self.dt = torch.float64
        self.episodes = episodes; self.sigma_d = sigma_d
        tk = TF[1:]
        Ct = matern(tk, len_d); lt, Ut = np.linalg.eigh(Ct)
        self.Ut = torch.as_tensor(Ut, dtype=self.dt, device=self.dev); self.lt = torch.as_tensor(np.clip(lt, 0, None), dtype=self.dt, device=self.dev)
        Mf = matern(tk, self.LEN_F)
        self.Minv = torch.as_tensor(np.linalg.inv(Mf), dtype=self.dt, device=self.dev); self.logdetM = float(np.linalg.slogdet(Mf)[1])
        self.rows = list(range(60)) if rows_obs is None else list(rows_obs); self.R = len(self.rows)   # observed marker rows
        self.Y = {e['name']: torch.as_tensor(np.asarray(e['y'])[:, self.rows], dtype=self.dt, device=self.dev) for e in episodes if e.get('y') is not None}

    # ------------------------------------------------------------ layout
    def layout(self, nF, nS):
        g = [('logE', math.log(2e4), math.log(2e5)), ('xc_u', 0., 9.), ('c_u', 0., 4+1/9), ('nu_u', 0., 3.),
             ('logsw', math.log(1e-5), math.log(1e-3))]
        if nF:
            g.append(('logtau', math.log(.1), math.log(10.)))
        f = [x for j in range(nF) for x in [(f'F{j}_u', 0., 1.), (f'F{j}_b', 0., 1.), (f'F{j}_logs', math.log(.006), math.log(.03))]]
        s = [x for j in range(nS) for x in [(f'S{j}_u', 0., 1.), (f'S{j}_b', 0., 1.), (f'S{j}_logs', math.log(.006), math.log(.03)),
                                            (f'S{j}_logk', math.log(1e2), math.log(1e6))]]
        return [a for a, _, _ in g+f+s], np.array([[lo, hi] for _, lo, hi in g+f+s])

    @staticmethod
    def material_index(th):
        inu = np.clip(np.floor(th[:, 3]), 0, 2).astype(int)
        cu = th[:, 2]; none = cu < 1/9
        ic = np.clip(np.floor(cu-1/9), 0, 3).astype(int); ixc = np.clip(np.floor(th[:, 1]), 0, 8).astype(int)
        return np.where(none, inu*37, inu*37+1+ixc*4+ic)

    def split(self, th, nF, nS):
        ng = 6 if nF else 5
        F = th[:, ng:ng+3*nF].reshape(len(th), nF, 3); S = th[:, ng+3*nF:ng+3*nF+4*nS].reshape(len(th), nS, 4)
        tau = np.exp(th[:, 5]) if nF else np.ones(len(th))
        return F, S, tau

    def ordered(self, th, nF, nS):
        F, S, _ = self.split(th, nF, nS); ok = np.ones(len(th), bool)
        if nF > 1: ok &= (np.diff(F[:, :, 0], axis=1) > 0).all(1)
        if nS > 1: ok &= (np.diff(S[:, :, 0], axis=1) > 0).all(1)
        return ok

    def footprint(self, u, b, logs):
        c, _ = surface_point(u, b)
        c = torch.as_tensor(c, dtype=self.dt, device=self.dev); s = torch.as_tensor(np.exp(logs), dtype=self.dt, device=self.dev)
        gate = torch.exp(-.5*((self.b.q[None]-c[:, None])**2).sum(-1)/s[:, None]**2)*self.b.qw[None]
        return (gate/gate.sum(-1, keepdim=True))@self.b.Nq                  # (B, nS)

    # ------------------------------------------------------------ dynamics
    def modal_time(self, w2):
        """Newmark modal responses at frames 1..24 minus quasi-static part. w2 (B,m) -> D (B,m,24,26), A (24,26)."""
        inp = self.b.inputs; c0 = 4/DT**2
        Bn, m = w2.shape; nc = inp.shape[1]
        eta = torch.zeros(Bn, m, nc, dtype=self.dt, device=self.dev); v = torch.zeros_like(eta)
        acc = inp[0][None, None].expand(Bn, m, nc).clone()
        den = (w2+c0)[..., None]; out = []
        for s in range(1, inp.shape[0]):
            pc = c0*eta+4/DT*v+acc
            x = (inp[s][None, None]+pc)/den
            an = c0*x-pc; v = v+DT/2*(acc+an); acc = an; eta = x
            if s % SAVE == 0:
                out.append(eta)
        eta = torch.stack(out, 2)                                          # (B,m,24,nc)
        A = inp[SAVE::SAVE]                                                # (24,nc)
        return eta-A[None, None]/w2[..., None, None], A

    # ------------------------------------------------------------ operators per batch
    def operators(self, th, nF, nS, rows):
        """Everything affine in the force amplitudes for every episode.
        Returns dict: G (B,24,R,nA) for hidden-force episodes, d[name] (B,24,R), tau, mat, E."""
        b = self.b; Bn = len(th); dt = self.dt; dev = self.dev
        mat = self.material_index(th); mi = torch.as_tensor(mat, device=dev)
        E = torch.as_tensor(np.exp(th[:, 0]), dtype=dt, device=dev)
        F, S, tau = self.split(th, nF, nS)
        w2 = E[:, None]*b.lam[mi]
        D, A = self.modal_time(w2)                                         # (B,m,24,26)
        HPhi = b.HPhi[mi][:, rows]                                         # (B,R,m)
        nP = nF+nS
        if nP:
            W = self.footprint(np.concatenate([F[:, :, 0], S[:, :, 0]], 1).ravel(), np.concatenate([F[:, :, 1], S[:, :, 1]], 1).ravel(),
                               np.concatenate([F[:, :, 2], S[:, :, 2]], 1).ravel()).view(Bn, nP, -1)
            SO = torch.zeros(Bn, nP, 3, len(rows), dtype=dt, device=dev); PM = torch.zeros(Bn, nP, 3, self.b.m, dtype=dt, device=dev)
            CC = torch.zeros(Bn, nP, 3, nS, 3, dtype=dt, device=dev) if nS else None
            KSw = torch.zeros(Bn, nS, 3, b.KS.shape[-1], dtype=dt, device=dev) if nS else None
            rows_t = torch.as_tensor(rows, device=dev)
            for mm in torch.unique(mi):
                sel = (mi == mm).nonzero()[:, 0]; Wm = W[sel]
                SO[sel] = torch.einsum('bpn,nek->bpek', Wm, b.XoS[mm][:, :, rows_t])
                PM[sel] = torch.einsum('bpn,nek->bpek', Wm, b.PhiS[mm])
                if nS:
                    Ws = Wm[:, nF:]                                           # (b,nS,n)
                    C = b.Css[mm]                                             # (n,3,n,3) f32
                    Cw = torch.einsum('nemf,bjm->bnejf', C, Ws.float()).double()   # (b,n,3,nS,3)
                    CC[sel] = torch.einsum('bpn,bnejf->bpejf', Wm, Cw)
                    KSw[sel] = torch.einsum('bjn,nel->bjel', Ws, b.KS[mm])
        out = dict(mat=mat, E=E, tau=torch.as_tensor(tau, dtype=dt, device=dev), D=D)

        def resp(SOp, PMp):
            """obs response (B,24,R,p,3,24) to unit traction hats of patches with static SOp (B,p,3,R), modal PMp (B,p,3,m)."""
            st = torch.einsum('bper,gc->bgrpec', SOp/E[:, None, None, None], torch.eye(NH, dtype=dt, device=dev))
            dy = torch.einsum('brk,bpek,bkgc->bgrpec', HPhi, PMp, D[..., :NH])
            return st+dy

        def known(ep):
            """deterministic response (B,24,R) and footprint-averaged support displacement (B,nS,3,24) of known loads."""
            dk = torch.zeros(Bn, NFR-1, len(rows), dtype=dt, device=dev)
            us = torch.zeros(Bn, nS, 3, NFR-1, dtype=dt, device=dev)
            for col, l, amp in ep['known']:
                c = COL_STEP if col == 'step' else COL_RAMP
                KO = b.KO[mi][:, rows, l]; PK = b.PhiK[mi][:, :, l]
                dk += amp*(KO[:, None]*A[None, :, c, None]/E[:, None, None]+torch.einsum('brk,bk,bkg->bgr', HPhi, PK, D[..., c]))
                if nS:
                    us += amp*(KSw[..., l][..., None]*A[None, None, None, :, c]/E[:, None, None, None]
                               + torch.einsum('bjek,bk,bkg->bjeg', PM[:, nF:], PK, D[..., c]))
            return dk, us

        nA = 3*NH*nF
        if not nS:
            out['G'] = resp(SO[:, :nF], PM[:, :nF]).reshape(Bn, NFR-1, len(rows), nA) if nF else None
            for ep in self.episodes:
                out['d_'+ep['name']] = known(ep)[0]
            return out
        # ---- supports: exact coupled Newmark in modal coordinates, reaction solved at EVERY step.
        # Columns: force patch (p,e,hat c) unit traction hats, then each distinct known (load l, history).
        kcols = sorted({(l, col) for ep in self.episodes for col, l, _ in ep['known']})
        nK = len(kcols); ncol = nA+nK
        inp = b.inputs; c0 = 4/DT**2
        hist = torch.zeros(inp.shape[0], ncol, dtype=dt, device=dev)
        if nF:
            hist[:, :nA] = inp[:, :NH].repeat(1, 3*nF)
        for i, (l, col) in enumerate(kcols):
            hist[:, nA+i] = inp[:, COL_STEP if col == 'step' else COL_RAMP]
        Fk = torch.zeros(Bn, self.b.m, ncol, dtype=dt, device=dev)            # modal load per column
        So = torch.zeros(Bn, len(rows), ncol, dtype=dt, device=dev)           # static obs per column (x E)
        Sa = torch.zeros(Bn, 3*nS, ncol, dtype=dt, device=dev)                # static support-average per column (x E)
        if nF:
            Fk[:, :, :nA] = PM[:, :nF].permute(0, 3, 1, 2).repeat_interleave(NH, dim=3).reshape(Bn, self.b.m, nA)
            So[:, :, :nA] = SO[:, :nF].permute(0, 3, 1, 2).repeat_interleave(NH, dim=3).reshape(Bn, len(rows), nA)
            Sa[:, :, :nA] = CC[:, :nF].permute(0, 3, 4, 1, 2).reshape(Bn, 3*nS, nF, 3).repeat_interleave(NH, dim=3).reshape(Bn, 3*nS, nA)
        for i, (l, col) in enumerate(kcols):
            Fk[:, :, nA+i] = b.PhiK[mi][:, :, l]; So[:, :, nA+i] = b.KO[mi][:, rows, l]; Sa[:, :, nA+i] = KSw[..., l].reshape(Bn, 3*nS)
        PMs = PM[:, nF:].reshape(Bn, 3*nS, self.b.m)                          # (B,3nS,m)
        SOs = SO[:, nF:].reshape(Bn, 3*nS, len(rows))
        Css = CC[:, nF:].reshape(Bn, 3*nS, 3*nS)                              # w_j C w_i (pairs ordered (j,f),(i,e))
        k = torch.as_tensor(np.exp(S[:, :, 3]), dtype=dt, device=dev).repeat_interleave(3, 1)   # (B,3nS)
        den = w2+c0
        Bmat = Css/E[:, None, None]+torch.einsum('bjk,bik,bk->bji', PMs, PMs, 1/den-1/w2)
        Mx = torch.eye(3*nS, dtype=dt, device=dev)[None]+k[:, :, None]*Bmat
        LU = torch.linalg.lu_factor(Mx)
        eta = torch.zeros(Bn, self.b.m, ncol, dtype=dt, device=dev); v = torch.zeros_like(eta)
        acc = Fk*hist[0][None, None]
        obs = []
        for s_ in range(1, inp.shape[0]):
            h = hist[s_][None, None]
            pc = c0*eta+4/DT*v+acc
            fk = Fk*h
            ex = (fk+pc)/den[..., None]
            a_s = Sa*h/E[:, None, None]+torch.einsum('bjk,bkc->bjc', PMs, ex-fk/w2[..., None])
            Rs = -torch.linalg.lu_solve(*LU, k[..., None]*a_s)               # (B,3nS,ncol)
            fR = torch.einsum('bjk,bjc->bkc', PMs, Rs)
            x = ex+fR/den[..., None]
            an = c0*x-pc; v = v+DT/2*(acc+an); acc = an; eta = x
            if s_ % SAVE == 0:
                st = (So*h+torch.einsum('bjr,bjc->brc', SOs, Rs))/E[:, None, None]
                obs.append(st+torch.einsum('brk,bkc->brc', HPhi, x-(fk+fR)/w2[..., None]))
        obs = torch.stack(obs, 1)                                               # (B,24,R,ncol)
        out['G'] = obs[..., :nA] if nF else None
        for ep in self.episodes:
            dk = torch.zeros(Bn, NFR-1, len(rows), dtype=dt, device=dev)
            for col, l, amp in ep['known']:
                dk = dk+amp*obs[..., nA+kcols.index((l, col))]
            out['d_'+ep['name']] = dk
        return out

    # ------------------------------------------------------------ likelihood
    def noise_var(self, th):
        sw2 = torch.as_tensor(np.exp(2*th[:, 4]), dtype=self.dt, device=self.dev)
        return sw2[:, None]+self.sigma_d**2*self.lt[None]                   # (B,24) per time-eigencomponent

    def marginal(self, th, nF, nS, op, conditionals=False):
        Bn = len(th); dt = self.dt; v = self.noise_var(th); R = self.R
        ll = torch.zeros(Bn, dtype=dt, device=self.dev); cond = None
        wt = lambda X: torch.einsum('gi,bg...->bi...', self.Ut, X)
        for ep in self.episodes:
            if ep.get('y') is None:
                continue
            r = self.Y[ep['name']][None]-op['d_'+ep['name']]                # (B,24,R)
            rt = wt(r)/v.sqrt()[..., None]
            ll = ll-.5*((rt**2).sum((1, 2))+R*torch.log(v).sum(1)+rt[0].numel()*math.log(2*math.pi))
            if ep.get('hidden') and nF:
                G = wt(op['G'])/v.sqrt()[..., None, None]                      # (B,24,R,nA)
                nA = G.shape[-1]; Gf = G.reshape(Bn, -1, nA)
                lam = (op['tau']*self.SIGMA_F)**2
                P = torch.kron(torch.eye(3*nF, dtype=dt, device=self.dev), self.Minv)[None]/lam[:, None, None]+Gf.transpose(1, 2)@Gf
                bv = (Gf.transpose(1, 2)@rt.reshape(Bn, -1, 1))[..., 0]
                L, info = torch.linalg.cholesky_ex(P)
                z = torch.linalg.solve_triangular(L, bv[..., None], upper=False)[..., 0]
                logdetP = 2*torch.log(torch.diagonal(L, dim1=1, dim2=2)).sum(1)
                logdetLam = nA*torch.log(lam)+3*nF*self.logdetM
                ll = ll+.5*((z**2).sum(1)-logdetLam-logdetP)
                ll = torch.where(info == 0, ll, torch.full_like(ll, -math.inf))
                if conditionals:
                    cond = (torch.cholesky_solve(bv[..., None], L)[..., 0], torch.cholesky_inverse(L))
        return ll, cond

    def loglik(self, theta, nF, nS, batch=128):
        out = np.full(len(theta), -np.inf); ok = self.ordered(theta, nF, nS)
        idx = np.flatnonzero(ok); rows = self.rows
        for i in range(0, len(idx), batch):
            j = idx[i:i+batch]
            with torch.no_grad():
                op = self.operators(theta[j], nF, nS, rows)
                ll, _ = self.marginal(theta[j], nF, nS, op)
            out[j] = np.nan_to_num(ll.cpu().numpy(), nan=-np.inf)
        return out
