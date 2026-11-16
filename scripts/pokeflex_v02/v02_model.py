"""PokeFlex v02 mechanism ladder: forward model on top of the frozen v01 quasi-static FEM simulations.
Protocol: scripts/pokeflex_v02/protocol_v02.json (post-hoc, non-blind; fixed before any v02 fit).

For one poke and one nu, the v01 simulation gives the elastic force F_sim[n] (N, at E_REF) and surface surf_V[n] at every
frame n of the poke (start frame first).  Rungs (all vectorised over N parameter sets):
  M0  F = c F_sim                                   c = E/E_REF
  M1  x = c F_sim; F = QLV(x) with Prony kernel G(t) = g_inf + sum_i g_i exp(-t/tau_i), G(0)=1; unilateral clip
  M2  + series contact compliance k: body depth u = d - f/k, x = c F_sim[n] * Ta(u)/Ta(d_n) (Ta = loading-branch
      along-force table of the same simulation; exactly M1 for k -> inf); surface surf_V[n] + S(u) - S(d_n)
  M3  + support translation delta_s = r f/k (k = k_eff); force identical to M2 (r enters the surface only)
Only free inputs here (simulations, robot trajectory, rest scans); outcomes are read by v02_fit.py through the v01 gate."""
import json, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'scripts/pokeflex_v01'))
import pokeflex as P  # noqa: E402
import realfem as RF  # noqa: E402
import pfx_ours as O  # noqa: E402  (load_sim, NUS, E_GRID: imported, never modified)
from pfx_common import OBJECTS, approach_axis, contact_depth, heldout_pokes, poke  # noqa: E402

HERE = Path(__file__).resolve().parent
PROT2 = json.loads((HERE/'protocol_v02.json').read_text())
OUT2 = ROOT/'runs/pokeflex-v02'
PRED2 = OUT2/'pred'
WORK2 = OUT2/'work'
V01PRED = ROOT/'runs/pokeflex-v01/pred'

DT = 1/30.0
NUS = O.NUS
E_GRID = O.E_GRID                                 # 401 points, 1e2..1e7 Pa
TAU = np.logspace(np.log10(0.03), np.log10(30), 13)
G = 0.025+0.05*np.arange(19)                      # 0.025..0.925
K = np.logspace(2, 5, 13)                         # N/m
RFR = 0.05+0.1*np.arange(10)                      # support fraction r
DELTA_GRID = np.array([0, 2, 4, 7, 11, 16, 23, 32, 45])*1e-3
S_GRID = np.array([0, 2, 4, 7, 11, 16, 23, 32])*1e-3
E_WIN = (-0.25, 1.25)                             # decades around E_M0 for M2/M3
SURF_EVERY = 2
N_BISECT = 30


def tau_pairs():
    return [(i, j) for i in range(len(TAU)) for j in range(i+1, len(TAU))]


def g_pairs():
    return [(i, j) for i in range(len(G)) for j in range(len(G)) if G[i]+G[j] <= 0.95+1e-9]


class PokeSim:
    """One (poke, nu) simulation with its loading-branch tables."""

    def __init__(self, obj, take, k, nu, z=None):
        z = O.load_sim(obj, take, k, nu) if z is None else z
        self.ok = z is not None
        if not self.ok:
            return
        self.z = z; self.snapped = bool(z['snapped'])
        fr, dep = contact_depth(obj, take, k)
        assert np.array_equal(np.asarray(z['frames']), fr), (obj, take, k, nu)
        self.frames, self.dep = fr, dep
        self.a = approach_axis(obj, take)
        self.F = np.asarray(z['force'], float)                 # (K,3) at E_REF
        self.fa = self.F @ -self.a
        self.surfF = z['surf_F']
        # loading-branch table: frames up to max depth, strictly increasing depth, non-decreasing along-force
        jm = int(np.argmax(dep)); idx, d_last = [], -np.inf
        for j in range(jm+1):
            if dep[j] > d_last+1e-6:
                idx.append(j); d_last = dep[j]
        self.t_idx = np.array(idx)
        self.t_d = dep[self.t_idx]
        self.t_f = np.maximum.accumulate(np.maximum(self.fa[self.t_idx], 0.0))

    def Ta(self, u):
        return np.interp(u, self.t_d, self.t_f, left=0.0, right=self.t_f[-1])

    def node_weights(self, u, w):
        """Interpolation weights on table nodes for depths u (any shape) with weights w; returns (n_nodes,)."""
        u = np.clip(np.ravel(u), self.t_d[0], self.t_d[-1]); w = np.broadcast_to(w, np.shape(u)).ravel() if np.ndim(w) else np.full(u.shape, w)
        i = np.clip(np.searchsorted(self.t_d, u, side='right')-1, 0, len(self.t_d)-2) if len(self.t_d) > 1 else np.zeros(u.shape, int)
        out = np.zeros(len(self.t_d))
        if len(self.t_d) == 1:
            out[0] = w.sum(); return out
        t = (u-self.t_d[i])/(self.t_d[i+1]-self.t_d[i])
        np.add.at(out, i, w*(1-t)); np.add.at(out, i+1, w*t)
        return out

    def S(self, u):
        """Loading-branch surface at body depth u (scalar)."""
        V = self.z['surf_V']
        return np.tensordot(self.node_weights(np.array([u]), 1.0), V[self.t_idx].astype(np.float64), 1)


def forward(ps, c, taus, gs, k=None, want=()):
    """Vectorised forward model for one PokeSim over N parameter sets.
    c (N,) = E/E_REF; taus, gs (N,m) Prony (m may be 0); k (N,) series compliance (None = rigid, M0/M1).
    Returns force (N,K,3) unless 'noforce' in want, and Delta (N,K) = d - u, f_along (N,K) if 'state' in want."""
    c = np.asarray(c, float); N = len(c); Kf = len(ps.dep); a = ps.a
    taus = np.asarray(taus, float).reshape(N, -1); gs = np.asarray(gs, float).reshape(N, -1)
    ad = np.exp(-DT/taus) if taus.size else np.zeros((N, 0))
    bd = (1-ad)*taus/DT if taus.size else np.zeros((N, 0))
    ginf = 1-gs.sum(1)
    A = ginf+(gs*bd).sum(1)
    h = np.zeros((N, gs.shape[1], 3)); xprev = np.zeros((N, 3))
    keepF = 'noforce' not in want
    Fout = np.zeros((N, Kf, 3)) if keepF else None
    Dl = np.zeros((N, Kf)) if 'state' in want else None
    fl = np.zeros((N, Kf)) if 'state' in want else None
    for n in range(Kf):
        B = np.einsum('nm,nmj->nj', ad, h)-np.einsum('nm,nj->nj', gs*bd, xprev)
        if k is None:
            x = np.outer(c, ps.F[n]); Dn = np.zeros(N)
        else:
            d = ps.dep[n]; tad = ps.Ta(d)
            Ba = B @ -a
            f0 = A*c*ps.fa[n]+Ba
            live = f0 > 0
            lo = np.zeros(N); hi = np.where(live, f0/k, 0.0)
            if tad > 1e-9:
                for _ in range(N_BISECT):
                    mid = 0.5*(lo+hi)
                    fm = A*c*ps.fa[n]*(ps.Ta(d-mid)/tad)+Ba
                    up = k*mid < fm               # root is above mid
                    lo = np.where(up, mid, lo); hi = np.where(up, hi, mid)
                Dn = np.where(live, 0.5*(lo+hi), 0.0)
                rho = np.where(Dn > 0, ps.Ta(d-Dn)/tad, 1.0)
            else:  # no loading-branch force at this depth: compliance not applied at this frame (falls back to M1)
                Dn = np.zeros(N); rho = np.ones(N)
            x = (c*rho)[:, None]*ps.F[n][None, :]
        h = ad[:, :, None]*h+(gs*bd)[:, :, None]*(x-xprev)[:, None, :]
        F = ginf[:, None]*x+h.sum(1)
        fa = F @ -a
        F[fa <= 0] = 0.0; fa = np.maximum(fa, 0.0)
        if k is not None:
            Dn = np.where(fa > 0, Dn, 0.0)
        if keepF:
            Fout[:, n] = F
        if Dl is not None:
            Dl[:, n] = Dn; fl[:, n] = fa
        xprev = x
    return Fout, Dl, fl


def save_pred(rung, obj, take, k, frames, force, meta, surf_V=None, surf_F=None, risk=None):
    """pfx_common.save_pred format, rooted at runs/pokeflex-v02/pred."""
    d = PRED2/rung/obj; d.mkdir(parents=True, exist_ok=True)
    arr = dict(frames=np.asarray(frames, int), force=np.asarray(force, float))
    if surf_V is not None:
        arr.update(surf_V=np.asarray(surf_V, np.float32), surf_F=np.asarray(surf_F, np.int32))
    if risk is not None:
        arr['risk'] = np.array(float(risk))
    np.savez_compressed(d/f'{take}-p{k}.npz', **arr)
    (d/f'{take}-p{k}.json').write_text(json.dumps(meta, indent=1))
