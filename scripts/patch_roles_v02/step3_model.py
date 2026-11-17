"""Step 3 model: step-2 patch-role model (model.py, unchanged) with
  (1) a SPATIALLY CORRELATED model-discrepancy term over the 20 markers:
        Cov = sigma_w^2 I + sigma_d^2 [Matern32_t(l_t) (x) SE_s(l_s; rest marker positions) (x) I_3]
      independent between episodes; exact whitening in the joint (time x marker) eigenbasis;
  (2) BOX force footprints: traction uniform on the part of the surface inside an axis-aligned cube of
      half-size h centred on the patch centre (soft edge eps = 1.5 mm), log h inferred in [ln 5, ln 35] mm.
      (All faces are axis-aligned, so on a face this is a square window like the protocol press.)
      Support patches keep the Gaussian footprint of step 2.
The hyperparameters (sigma_d, l_t, l_s) are fixed per case from a commanded-D0-only fit (step3_run.fit_disc).
"""
from __future__ import annotations
import math
import numpy as np
import torch
from scripts.patch_roles_v02 import model as M
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'scripts/fig1_v01'))
import block as B

EPS_BOX = 1.5e-3


def se_kernel(P, length):
    d2 = ((P[:, None]-P[None])**2).sum(-1)
    return np.exp(-.5*d2/length**2)


class Step3Model(M.PatchModel):
    def __init__(self, bank, episodes, sigma_d, len_t, len_s, force_shape='box'):
        super().__init__(bank, episodes, sigma_d=sigma_d, len_d=len_t)
        self.len_t, self.len_s, self.force_shape = len_t, len_s, force_shape
        ls, Us = np.linalg.eigh(se_kernel(B.MARKERS, len_s)+1e-9*np.eye(20))
        self.Us = torch.as_tensor(Us, dtype=self.dt, device=self.dev); self.ls = torch.as_tensor(np.clip(ls, 0, None), dtype=self.dt, device=self.dev)
        self._cur = (0, 0)

    def layout(self, nF, nS):
        names, bounds = super().layout(nF, nS)
        if self.force_shape == 'box':
            for j in range(nF):
                i = names.index(f'F{j}_logs'); names[i] = f'F{j}_logh'; bounds[i] = [math.log(.005), math.log(.035)]
        return names, bounds

    def box(self, u, b, logh):
        c, _ = M.surface_point(u, b)
        c = torch.as_tensor(c, dtype=self.dt, device=self.dev); h = torch.as_tensor(np.exp(logh), dtype=self.dt, device=self.dev)
        gate = torch.sigmoid((h[:, None, None]-(self.b.q[None]-c[:, None]).abs())/EPS_BOX).prod(-1)*self.b.qw[None]
        return (gate/gate.sum(-1, keepdim=True).clamp_min(1e-300))@self.b.Nq

    def footprint(self, u, b, logs):
        nF, nS = self._cur; nP = nF+nS
        if self.force_shape != 'box' or nF == 0:
            return super().footprint(u, b, logs)
        isF = (np.arange(len(u)) % nP) < nF
        out = torch.empty(len(u), self.b.Nq.shape[1], dtype=self.dt, device=self.dev)
        out[torch.as_tensor(isF, device=self.dev)] = self.box(u[isF], b[isF], logs[isF])
        if (~isF).any():
            out[torch.as_tensor(~isF, device=self.dev)] = super().footprint(u[~isF], b[~isF], logs[~isF])
        return out

    def operators(self, th, nF, nS, rows):
        self._cur = (nF, nS)
        return super().operators(th, nF, nS, rows)

    # ---------------- correlated noise
    def noise_var(self, th):
        sw2 = torch.as_tensor(np.exp(2*th[:, 4]), dtype=self.dt, device=self.dev)
        return sw2[:, None, None]+self.sigma_d**2*self.lt[None, :, None]*self.ls[None, None, :]     # (B,24,20)

    def whiten(self, X, v):
        """X (B,24,60,...) -> whitened (B,24,60,...) in the (time-eig, marker-eig, coord) basis, scaled by 1/sqrt(v)."""
        sh = X.shape; Xr = X.reshape(sh[0], 24, 20, 3, *sh[3:])
        Y = torch.einsum('gi,mj,bgmk...->bijk...', self.Ut, self.Us, Xr)
        s = v.sqrt().reshape(sh[0], 24, 20, 1, *([1]*(len(sh)-3)))
        return (Y/s).reshape(sh)

    def marginal(self, th, nF, nS, op, conditionals=False):
        assert self.R == 60
        Bn = len(th); dt = self.dt; v = self.noise_var(th)
        ll = torch.zeros(Bn, dtype=dt, device=self.dev); cond = None
        logdetR = 3*torch.log(v).sum((1, 2))
        for ep in self.episodes:
            if ep.get('y') is None:
                continue
            rt = self.whiten(self.Y[ep['name']][None]-op['d_'+ep['name']], v)
            ll = ll-.5*((rt**2).sum((1, 2))+logdetR+rt[0].numel()*math.log(2*math.pi))
            if ep.get('hidden') and nF:
                G = self.whiten(op['G'], v); nA = G.shape[-1]; Gf = G.reshape(Bn, -1, nA)
                lam = (op['tau']*self.SIGMA_F)**2
                P = torch.kron(torch.eye(3*nF, dtype=dt, device=self.dev), self.Minv)[None]/lam[:, None, None]+Gf.transpose(1, 2)@Gf
                bv = (Gf.transpose(1, 2)@rt.reshape(Bn, -1, 1))[..., 0]
                L, info = torch.linalg.cholesky_ex(P)
                z = torch.linalg.solve_triangular(L, bv[..., None], upper=False)[..., 0]
                logdetP = 2*torch.log(torch.diagonal(L, dim1=1, dim2=2)).sum(1)
                ll = ll+.5*((z**2).sum(1)-nA*torch.log(lam)-3*nF*self.logdetM-logdetP)
                ll = torch.where(info == 0, ll, torch.full_like(ll, -math.inf))
                if conditionals:
                    cond = (torch.cholesky_solve(bv[..., None], L)[..., 0], torch.cholesky_inverse(L))
        return ll, cond

    def loglik(self, theta, nF, nS, batch=None):
        return super().loglik(theta, nF, nS, batch=batch or (256 if nS else 128))


HYP_BOUNDS = [[math.log(1e-5), math.log(1e-3)], [math.log(.01), math.log(.5)], [math.log(.01), math.log(.3)]]


class Step3ModelInfer(Step3Model):
    """Fallback (declared after the reduced-budget gate smoke test failed): the discrepancy hyperparameters
    (log sigma_d, log l_t, log l_s) are inferred jointly in every configuration (same priors as the D0 fit),
    appended as the LAST three theta coordinates.  Per-particle exact whitening."""
    def __init__(self, bank, episodes, hyp_median, force_shape='box'):
        super().__init__(bank, episodes, hyp_median['sigma_d'], hyp_median['len_t'], hyp_median['len_s'], force_shape)
        self.tk = torch.as_tensor(M.TF[1:], dtype=self.dt, device=self.dev); self.Xm = torch.as_tensor(B.MARKERS, dtype=self.dt, device=self.dev)

    def layout(self, nF, nS):
        names, bounds = super().layout(nF, nS)
        return names+['logsd', 'loglt', 'logls'], np.vstack([bounds, HYP_BOUNDS])

    def kernels(self, th):
        lt_ = torch.as_tensor(np.exp(th[:, -2]), dtype=self.dt, device=self.dev); ls_ = torch.as_tensor(np.exp(th[:, -1]), dtype=self.dt, device=self.dev)
        r = np.sqrt(3)*(self.tk[:, None]-self.tk[None]).abs()[None]/lt_[:, None, None]
        et, Ut = torch.linalg.eigh((1+r)*torch.exp(-r))
        Cs = torch.exp(-.5*((self.Xm[:, None]-self.Xm[None])**2).sum(-1)[None]/ls_[:, None, None]**2)+1e-9*torch.eye(20, dtype=self.dt, device=self.dev)
        es, Us = torch.linalg.eigh(Cs)
        return Ut, et.clamp_min(0), Us, es.clamp_min(0)

    def noise_var(self, th):
        _, et, _, es = self.kernels(th)
        sw2 = torch.as_tensor(np.exp(2*th[:, 4]), dtype=self.dt, device=self.dev); sd2 = torch.as_tensor(np.exp(2*th[:, -3]), dtype=self.dt, device=self.dev)
        return sw2[:, None, None]+sd2[:, None, None]*et[:, :, None]*es[:, None, :]

    def marginal(self, th, nF, nS, op, conditionals=False):
        Ut, et, Us, es = self.kernels(th)
        sw2 = torch.as_tensor(np.exp(2*th[:, 4]), dtype=self.dt, device=self.dev); sd2 = torch.as_tensor(np.exp(2*th[:, -3]), dtype=self.dt, device=self.dev)
        v = sw2[:, None, None]+sd2[:, None, None]*et[:, :, None]*es[:, None, :]
        def whiten(X):
            sh = X.shape; Xr = X.reshape(sh[0], 24, 20, 3, *sh[3:])
            Y = torch.einsum('bgi,bmj,bgmk...->bijk...', Ut, Us, Xr)
            return (Y/v.sqrt().reshape(sh[0], 24, 20, 1, *([1]*(len(sh)-3)))).reshape(sh)
        self._wh = whiten; self._v = v
        return super().marginal(th, nF, nS, op, conditionals)

    def whiten(self, X, v):
        return self._wh(X)


class _LamProxy:
    def __init__(self, lam, scale):
        self.lam, self.scale = lam, scale; self.shape = lam.shape

    def __getitem__(self, idx):
        return self.lam[idx]*self.scale


class Step3ModelFreq(Step3ModelInfer):
    """Second fallback (declared after the fixed-hyperparameter gate failed): targeted modal-frequency discrepancy.
    The coarse linear mesh and the fine nonlinear truth ring at slightly different frequencies (gravity release
    excites undamped 9-12 Hz modes), a phase drift that no smooth GP absorbs.  The N_FREQ lowest model eigenvalues
    get independent nuisance factors exp(2 eps_k), eps_k ~ U[-0.05, 0.05] (+-5% frequency), inferred jointly with
    everything else (theta coordinates just before the three discrepancy hyperparameters)."""
    N_FREQ = 6

    def layout(self, nF, nS):
        names, bounds = super().layout(nF, nS)
        n = self.N_FREQ
        return names[:-3]+[f'feps{k}' for k in range(n)]+names[-3:], np.vstack([bounds[:-3], [[-.05, .05]]*n, bounds[-3:]])

    def operators(self, th, nF, nS, rows):
        n = self.N_FREQ; eps = th[:, -3-n:-3]
        s = torch.ones(len(th), self.b.m, dtype=self.dt, device=self.dev)
        s[:, :n] = torch.as_tensor(np.exp(2*eps), dtype=self.dt, device=self.dev)
        lam = self.b.lam; self.b.lam = _LamProxy(lam, s)
        try:
            return super().operators(th, nF, nS, rows)
        finally:
            self.b.lam = lam


class Step3Model2(Step3Model):
    """Dev model (step 3, after the single-kernel gate failed): two-component discrepancy sharing one time kernel,
        Cov = sigma_w^2 I + Matern32_t(l_t) (x) [a^2 I_20 + b^2 SE_s(l_s)] (x) I_3,
    a = per-marker (tracking-bias-like) part, b = spatially smooth (model-error-like) part.  Still one Kronecker
    product, so exact whitening is unchanged; hyperparameters fixed from a commanded-episode fit (step3_run.fit_disc2)."""
    def __init__(self, bank, episodes, h, force_shape='box'):
        super().__init__(bank, episodes, 1.0, h['len_t'], h['len_s'], force_shape)
        S = h['a']**2*np.eye(20)+h['b']**2*se_kernel(B.MARKERS, h['len_s'])
        ls, Us = np.linalg.eigh(S)
        self.Us = torch.as_tensor(Us, dtype=self.dt, device=self.dev); self.ls = torch.as_tensor(np.clip(ls, 0, None), dtype=self.dt, device=self.dev)
        self.h = h


def helmert(n):
    """Orthonormal n x n matrix whose first row is the mean direction 1/sqrt(n)."""
    H = np.zeros((n, n))
    if n == 0:
        return H
    H[0] = 1/np.sqrt(n)
    for k in range(1, n):
        H[k, :k] = 1/np.sqrt(k*(k+1)); H[k, k] = -k/np.sqrt(k*(k+1))
    return H


class Step3ModelShared(Step3Model):
    """Dev model: discrepancy with an episode-SHARED part (diagnostic: at 10 N 85% of the coarse-linear model error
    is common to all episodes of a world -- the gravity-response error -- and 40-60% at 30 N).
        Cov(y_e, y_e') = delta_ee' [sigma_w^2 I + Kt (x) S_ind (x) I3] + Kt (x) S_sh (x) I3,
        S_ind = a^2 I + b^2 SE(l_s),  S_sh = c^2 I + d^2 SE(l_s)   (common eigenbasis)
    Exact whitening: Helmert transform over the observed episodes (mean direction gets S_ind + n S_sh), then the
    (time, marker) eigenbasis.  The hidden-force columns enter every Helmert combination with their coefficient."""
    def __init__(self, bank, episodes, h, force_shape='box'):
        super().__init__(bank, episodes, 1.0, h['len_t'], h['len_s'], force_shape)
        SE = se_kernel(B.MARKERS, h['len_s']); ls, Us = np.linalg.eigh(SE)
        self.Us = torch.as_tensor(Us, dtype=self.dt, device=self.dev); se = np.clip(ls, 0, None)
        self.lind = torch.as_tensor(h['a']**2+h['b']**2*se, dtype=self.dt, device=self.dev)
        self.lsh = torch.as_tensor(h['c']**2+h['d']**2*se, dtype=self.dt, device=self.dev)
        self.obs = [e['name'] for e in episodes if e.get('y') is not None]
        self.H = torch.as_tensor(helmert(len(self.obs)), dtype=self.dt, device=self.dev)
        self.h = h

    PER_COMPONENT_TAU = True

    def layout(self, nF, nS):
        names, bounds = super().layout(nF, nS)
        if nF and self.PER_COMPONENT_TAU:      # logtau (index 5) = x-component; y and z scales appended at the end
            names = names+['logtau_y', 'logtau_z']; bounds = np.vstack([bounds, [[math.log(.1), math.log(10.)]]*2])
        return names, bounds

    def lam3(self, th, op):
        if not self.PER_COMPONENT_TAU:
            return ((op['tau']*self.SIGMA_F)**2)[:, None].expand(-1, 3)
        lt = np.stack([th[:, 5], th[:, -2], th[:, -1]], 1)
        return torch.as_tensor(np.exp(2*lt), dtype=self.dt, device=self.dev)*self.SIGMA_F**2

    def noise_var(self, th):      # per-episode marginal variance (used by the planner for a NEW independent episode)
        sw2 = torch.as_tensor(np.exp(2*th[:, 4]), dtype=self.dt, device=self.dev)
        return sw2[:, None, None]+self.lt[None, :, None]*(self.lind+self.lsh)[None, None, :]

    def marginal(self, th, nF, nS, op, conditionals=False):
        Bn = len(th); dt = self.dt; n = len(self.obs)
        sw2 = torch.as_tensor(np.exp(2*th[:, 4]), dtype=dt, device=self.dev)
        rot = lambda X: torch.einsum('gi,mj,bgmk...->bijk...', self.Ut, self.Us, X.reshape(X.shape[0], 24, 20, 3, *X.shape[3:]))
        R = torch.stack([rot(self.Y[e][None]-op['d_'+e]) for e in self.obs], 0)               # (n,B,24,20,3)
        Rk = torch.einsum('ke,ebijc->kbijc', self.H, R)
        vk = torch.stack([sw2[:, None, None]+self.lt[None, :, None]*(self.lind+(n if k == 0 else 0)*self.lsh)[None, None, :] for k in range(n)], 0)
        ll = -.5*((Rk**2/vk[..., None]).sum((0, 2, 3, 4))+3*torch.log(vk).sum((0, 2, 3))+n*1440*math.log(2*math.pi))
        cond = None
        hid = [i for i, e in enumerate(self.obs) if any(x['name'] == e and x.get('hidden') for x in self.episodes)]
        if hid and nF:
            i = hid[0]; G = rot(op['G']); nA = G.shape[-1]
            lam3 = self.lam3(th, op)                                                                  # (B,3) per-component prior var
            P = torch.zeros(Bn, nA, nA, dtype=dt, device=self.dev)
            for p_ in range(nF):
                for e in range(3):
                    o = (3*p_+e)*24
                    P[:, o:o+24, o:o+24] = self.Minv[None]/lam3[:, e, None, None]
            bv = torch.zeros(Bn, nA, dtype=dt, device=self.dev)
            for k in range(n):
                hk = self.H[k, i]
                if abs(float(hk)) < 1e-14:
                    continue
                Gk = (hk*G/vk[k].sqrt()[..., None, None]).reshape(Bn, -1, nA); rk = (Rk[k]/vk[k].sqrt()[..., None]).reshape(Bn, -1)
                P = P+Gk.transpose(1, 2)@Gk; bv = bv+(Gk.transpose(1, 2)@rk[..., None])[..., 0]
            L, info = torch.linalg.cholesky_ex(P)
            z = torch.linalg.solve_triangular(L, bv[..., None], upper=False)[..., 0]
            ll = ll+.5*((z**2).sum(1)-nF*24*torch.log(lam3).sum(1)-3*nF*self.logdetM-2*torch.log(torch.diagonal(L, dim1=1, dim2=2)).sum(1))
            ll = torch.where(info == 0, ll, torch.full_like(ll, -math.inf))
            if conditionals:
                cond = (torch.cholesky_solve(bv[..., None], L)[..., 0], torch.cholesky_inverse(L))
        return ll, cond


# ---------------------------------------------------------------------------------------------------------------
# Volume force patches.  Force patch = uniform body load in a layer of depth DEPTH below the surface, over the
# tangential square |dx_t| < h (soft edges), i.e. the same load class as the protocol press (4x4 cm x 1 cm box).
# Static/modal responses use the volume bank (step3_bank.py).  Force<->support static coupling uses the surface
# box of the same (c, h) as an approximation (only enters configurations with both roles).
import inspect, textwrap
DEPTH = 0.01
VOL_BANK = Path(__file__).resolve().parents[2]/'runs/patch-roles-v02/step3/bankvol'
FACE_AXES = {'left': (0, -1., (2, 1)), 'top': (1, 1., (0, 2)), 'right': (0, 1., (2, 1)), 'front': (2, 1., (0, 1)), 'back': (2, -1., (0, 1))}

_src = textwrap.dedent(inspect.getsource(M.PatchModel.operators))
_a = _src.index('\n    if nP:')+1; _b = _src.index('\n    out = dict(')+1
_new_block = '''    if nP:
        SO, PM, CC, KSw = self.patch_operators(th, F, S, nF, nS, mi, rows)
'''
_src = _src[:_a]+_new_block+_src[_b:]
_g = dict(M.__dict__); exec(_src, _g); _operators_vol = _g['operators']


class Step3ModelVol(Step3ModelShared):
    def __init__(self, bank, episodes, h):
        super().__init__(bank, episodes, h, force_shape='box')
        if not hasattr(bank, 'XoV'):
            zs = [np.load(VOL_BANK/f'mat{i:03d}.npz') for i in range(len(bank.mats))]
            T = lambda k, d=torch.float32: torch.as_tensor(np.stack([z[k] for z in zs]), dtype=d, device=self.dev)
            bank.XoV, bank.PhiV, bank.Ovol = T('XoV'), T('PhiV'), T('O', torch.float64)
            bank.fnodes = torch.as_tensor(zs[0]['fnodes'], device=self.dev)
        m = bank.geo.m
        self.SX = torch.as_tensor(m.sample_X, dtype=torch.float32, device=self.dev)                    # (E,S,3)
        self.cw0 = torch.as_tensor(m.vol[:, None]*m.sample_w[None], dtype=torch.float32, device=self.dev)
        self.sN = torch.as_tensor(m.sample_N, dtype=torch.float32, device=self.dev)                    # (S,10)
        self.elem = torch.as_tensor(m.elem, device=self.dev); self.nnode = m.n

    def vol_footprint(self, u, b, logh):
        c, face = M.surface_point(u, b); out = []
        for i0 in range(0, len(u), 32):
            sl = slice(i0, i0+32); cc = torch.as_tensor(c[sl], dtype=torch.float32, device=self.dev)
            h = torch.as_tensor(np.exp(logh[sl]), dtype=torch.float32, device=self.dev)
            ax = np.array([FACE_AXES[f][0] for f in face[sl]]); sg = np.array([FACE_AXES[f][1] for f in face[sl]])
            t = np.array([FACE_AXES[f][2] for f in face[sl]])
            D = self.SX[None]-cc[:, None, None]                                                       # (b,E,S,3)
            ar = torch.arange(len(ax), device=self.dev)
            depth = -D[ar, :, :, torch.as_tensor(ax, device=self.dev)]*torch.as_tensor(sg, dtype=torch.float32, device=self.dev)[:, None, None]
            g = torch.sigmoid((DEPTH-depth)/EPS_BOX)*torch.sigmoid(depth/EPS_BOX+3)
            for k in range(2):
                tk = torch.as_tensor(t[:, k], device=self.dev)
                g = g*torch.sigmoid((h[:, None, None]-D[ar, :, :, tk].abs())/EPS_BOX)
            we = torch.einsum('bes,sa->bea', g*self.cw0[None], self.sN)                                 # (b,E,10)
            wn = torch.zeros(len(ax), self.nnode, dtype=torch.float32, device=self.dev)
            wn.index_add_(1, self.elem.reshape(-1), we.reshape(len(ax), -1))
            wn = wn/wn.sum(1, keepdim=True).clamp_min(1e-30)
            out.append(wn[:, self.b.fnodes].double())
        return torch.cat(out)

    def patch_operators(self, th, F, S, nF, nS, mi, rows):
        b = self.b; Bn = len(th); dt = self.dt; dev = self.dev; nP = nF+nS
        rows_t = torch.as_tensor(rows, device=dev)
        SO = torch.zeros(Bn, nP, 3, len(rows), dtype=dt, device=dev); PM = torch.zeros(Bn, nP, 3, b.m, dtype=dt, device=dev)
        CC = torch.zeros(Bn, nP, 3, nS, 3, dtype=dt, device=dev) if nS else None
        KSw = torch.zeros(Bn, nS, 3, b.KS.shape[-1], dtype=dt, device=dev) if nS else None
        if nF:
            WF = self.vol_footprint(F[:, :, 0].ravel(), F[:, :, 1].ravel(), F[:, :, 2].ravel()).view(Bn, nF, -1)
        if nS:
            WS = Step3Model.footprint(self, S[:, :, 0].ravel(), S[:, :, 1].ravel(), S[:, :, 2].ravel()) if False else \
                M.PatchModel.footprint(self, S[:, :, 0].ravel(), S[:, :, 1].ravel(), S[:, :, 2].ravel())
            WS = WS.view(Bn, nS, -1)
            if nF:
                WFs = self.box(F[:, :, 0].ravel(), F[:, :, 1].ravel(), F[:, :, 2].ravel()).view(Bn, nF, -1)   # surface proxy for coupling
        for mm in torch.unique(mi):
            sel = (mi == mm).nonzero()[:, 0]
            if nF:
                SO[sel, :nF] = torch.einsum('bpn,nek->bpek', WF[sel], b.XoV[mm][:, :, rows_t].double())
                pv = torch.einsum('bpn,nek->bpek', WF[sel], b.PhiV[mm].double())
                PM[sel, :nF] = torch.einsum('kl,bpel->bpek', b.Ovol[mm], pv)
            if nS:
                Ws = WS[sel]
                SO[sel, nF:] = torch.einsum('bpn,nek->bpek', Ws, b.XoS[mm][:, :, rows_t])
                PM[sel, nF:] = torch.einsum('bpn,nek->bpek', Ws, b.PhiS[mm])
                Cw = torch.einsum('nemf,bjm->bnejf', b.Css[mm], Ws.float()).double()
                CC[sel, nF:] = torch.einsum('bpn,bnejf->bpejf', Ws, Cw)
                if nF:
                    CC[sel, :nF] = torch.einsum('bpn,bnejf->bpejf', WFs[sel], Cw)
                KSw[sel] = torch.einsum('bjn,nel->bjel', Ws, b.KS[mm])
        return SO, PM, CC, KSw

    def layout(self, nF, nS):
        names, bounds = super().layout(nF, nS)
        for j in range(nF):
            i = names.index(f'F{j}_logh'); bounds[i] = [math.log(.005), math.log(.04)]
        return names, bounds

    def operators(self, th, nF, nS, rows):
        self._cur = (nF, nS)
        return _operators_vol(self, th, nF, nS, rows)
