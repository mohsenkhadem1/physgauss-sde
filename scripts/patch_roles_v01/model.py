"""K Gaussian surface patches with discrete roles {force, support, off}.

Generalises the published joint-field unknown-load model (one bottom support,
one shared-centre two-mode load) to nF force patches and nS support patches.
Imports, never modifies, scripts/joint_field_identification (FEM body, published
footprints, constants). Quasistatic, small strain, block geometry, no gravity.

Conditional on a count configuration (nF, nS) and continuous theta:
    K_theta u_t = B_theta a_t + sum_c c_{c,t} f_c          (original episode)
    y_t = M_t (G_theta a_t + D_theta c_t) + eps,  eps ~ N(0, sigma^2 I)
    a = vec_t(a_t) ~ N(0, Lambda), Lambda = tau^2 sigma_F^2 Matern32(t,t') (x) I_nA
The amplitudes are integrated out exactly in amplitude space (Woodbury):
    P = Lambda^{-1} + sum_t e_t e_t^T (x) G^T M_t G / sigma^2,  b = vec_t G^T M_t r_t / sigma^2
    log p(y|theta) = -1/2 [ r^T r/sigma^2 - b^T P^{-1} b + log|Lambda| + log|P| + n log 2 pi sigma^2 ]
    mu = P^{-1} b,  S = P^{-1}
Reset (probe) episodes are known loads only: y_p ~ N(p * g_theta, sigma^2 I).
"""
from __future__ import annotations
import math
import numpy as np
import torch
from scripts.joint_field_identification.mechanics import LinearGaussianBody
from scripts.joint_field_identification import surface as pub

TIMES = pub.TIMES                 # 8 loading times in [0,1]
NOISE = pub.NOISE                 # 0.02 mm
FORCE_STD = pub.FORCE_STD         # 0.08 N
PROBE_FORCE = pub.PROBE_FORCE     # 0.08 N
LENGTH = .30                      # Matern-3/2 correlation length (published)
PERIM = .24                       # perimeter of the y-z cross-section (m)
COARSE_CELLS = (8, 4, 4)

# ---------------------------------------------------------------- geometry

def surface_point(x, phi):
    """Centre on the four lateral faces. phi runs bottom(y:-.03->.03) -> back(z:0->.06)
    -> top(y:.03->-.03) -> front(z:.06->0). Returns (point[...,3], outward normal[...,3])."""
    x = np.asarray(x, float); phi = np.mod(np.asarray(phi, float), PERIM)
    seg = np.minimum((phi // .06).astype(int), 3); s = phi - .06 * seg
    y = np.select([seg == 0, seg == 1, seg == 2, seg == 3], [-.03 + s, .03 * np.ones_like(s), .03 - s, -.03 * np.ones_like(s)])
    z = np.select([seg == 0, seg == 1, seg == 2, seg == 3], [0 * s, s, .06 * np.ones_like(s), .06 - s])
    normals = np.array([[0, 0, -1.], [0, 1., 0], [0, 0, 1.], [0, -1., 0]])
    return np.stack([x, y, z], -1), normals[seg]


def phi_of(face, s):
    """Inverse helper: face in bottom/back/top/front, s = arc offset in that face."""
    return {'bottom': 0., 'back': .06, 'top': .12, 'front': .18}[face] + s


def obs_points():
    xs = [-.045, -.0225, 0., .0225, .045]
    front = [(x, -.03, z) for x in xs for z in (.02, .045)]
    top = [(x, y, .06) for x in xs for y in (-.015, .015)]
    back = [(x, .03, z) for x in xs for z in (.02, .045)]
    return np.array(front), np.array(top), np.array(back)


def matern(times=TIMES, length=LENGTH):
    d = np.abs(times[:, None] - times[None, :]); r = np.sqrt(3) * d / length
    return (1 + r) * np.exp(-r)


# ---------------------------------------------------------------- loads

class Surface:
    """Surface quadrature of a LinearGaussianBody, reused from mechanics._surface."""
    def __init__(self, body):
        self.body = body
        self.q = body.surface_points; self.w = body.surface_quadrature_weights
        self.N = body.surface_interpolation

    def gauss(self, centres, widths):
        """Normalised Gaussian surface footprint (ambient distance), nodal weights [B,n]."""
        c = torch.as_tensor(centres, dtype=self.q.dtype, device=self.q.device).reshape(-1, 3)
        s = torch.as_tensor(widths, dtype=self.q.dtype, device=self.q.device).reshape(-1)
        gate = torch.exp(-.5 * ((self.q[None] - c[:, None]) ** 2).sum(-1) / s[:, None] ** 2) * self.w[None]
        den = gate.sum(-1, keepdim=True)
        if bool((den <= 1e-300).any()): raise ValueError('Empty footprint')
        return (gate / den) @ self.N

    def tophat(self, centres, radius):
        c = torch.as_tensor(centres, dtype=self.q.dtype, device=self.q.device).reshape(-1, 3)
        gate = ((((self.q[None] - c[:, None]) ** 2).sum(-1)) < radius ** 2).to(self.q.dtype) * self.w[None]
        return (gate / gate.sum(-1, keepdim=True)) @ self.N


def load_vector(body, surf, spec):
    """Known load column [ndof] for spec dict (unit total force, N)."""
    if spec['kind'] == 'pub':
        return body.custom_probe_forces([dict(x=spec['x'], mode=spec['mode'])])[:, 0]
    c, _ = surface_point(spec['x'], spec['phi'])
    w = surf.gauss(c[None], [spec['width']])[0]
    d = torch.as_tensor(spec['dir'], dtype=body.dtype, device=body.device); d = d / d.norm()
    return (w[:, None] * d).flatten()


def gspec(face, x, s, direction, width=.012, name=''):
    return dict(kind='gauss', x=x, phi=phi_of(face, s), width=width, dir=list(direction), name=name)


# Held-out known commands (reset episodes; targets (b)).
TARGET_COMMANDS = [gspec('top', -.03, .03, (0, 0, -1), name='C1 top -z x=-30mm'),
                   gspec('top', .03, .03, (0, 0, -1), name='C2 top -z x=+30mm'),
                   gspec('front', 0., .03, (0, 1, 0), name='C3 front +y x=0'),
                   gspec('back', .03, .03, (0, -1, 0), name='C4 back -y x=+30mm')]
# Candidate calibrated pushes (reset episodes observed by front+top cameras).
PROBES = [gspec('top', x, .03, (0, 0, -1), name=f'P top -z x={x*1000:+.0f}mm') for x in (-.04, 0., .04)] + \
         [gspec('front', x, .03, (0, 1, 0), name=f'P front +y x={x*1000:+.0f}mm') for x in (-.04, 0., .04)]
LOOK_TIMES = (1, 4, 6)            # candidate back-camera frames of the original episode


# ---------------------------------------------------------------- model

class PatchModel:
    """Batched FEM forward + exact amplitude marginal for one family.

    family 'published': force patch = published top(-z)+side(-y) footprints at x
      (width .018), support = published bottom footprint at x. Theta per force: x;
      per support: x, log k.  Bounds = published BOUNDS.
    family 'general': patch centre (x, phi) on lateral faces, log width; force has
      3 Cartesian modes; support isotropic. Theta per force: x,phi,log s;
      per support: x,phi,log s,log k.
    Global theta: log E0, contrast m1, [log tau if hierarchical].
    """
    def __init__(self, family='general', cells=COARSE_CELLS, device='cuda', commands=(),
                 hierarchical=False, dtype=torch.float64):
        self.family = family; self.hier = hierarchical; self.device = torch.device(device)
        self.body = LinearGaussianBody('block', cells=cells, device=device, dtype=dtype)
        self.surf = Surface(self.body); self.dtype = dtype
        b = self.body
        front, top, back = obs_points()
        self.points = np.concatenate([front, top, back])
        self.interp = b.interpolation(self.points)                      # [30,n]
        # camera rows: front (x,z), top (x,y), back (x,z) for looks
        rows = []
        for i in range(10): rows += [(i, 0), (i, 2)]
        for i in range(10, 20): rows += [(i, 0), (i, 1)]
        for i in range(20, 30): rows += [(i, 0), (i, 2)]
        self.rows = rows                                                 # 60 projected rows
        self.n_visible = 40                                              # front+top rows
        self.commands = [dict(c) for c in commands]
        self.known = torch.stack([load_vector(b, self.surf, c) for c in self.commands], -1) if commands else \
            torch.zeros(b.ndof, 0, dtype=dtype, device=self.device)
        self.probe_cols = torch.stack([load_vector(b, self.surf, c) for c in PROBES], -1)
        self.target_cols = torch.stack([load_vector(b, self.surf, c) for c in TARGET_COMMANDS], -1)
        self.Minv = torch.as_tensor(np.linalg.inv(matern()), dtype=dtype, device=self.device)
        self.logdetM = float(np.linalg.slogdet(matern())[1])
        self.T = len(TIMES)
        self.modes = 2 if family == 'published' else 3
        r = torch.as_tensor(rows, device=self.device)
        self._row_point = r[:, 0]; self._row_comp = r[:, 1]

    # ---------------- parameter layout
    def layout(self, nF, nS):
        g = [('logE', math.log(8000), math.log(32000)), ('m1', -.8, .8)]
        if self.hier: g.append(('logtau', math.log(.5), math.log(2.5)))
        f, s = [], []
        if self.family == 'published':
            for j in range(nF): f.append((f'F{j}_x', -.04, .04))
            for j in range(nS): s += [(f'S{j}_x', -.025, .025), (f'S{j}_logk', math.log(4000), math.log(16000))]
        else:
            for j in range(nF): f += [(f'F{j}_x', -.05, .05), (f'F{j}_phi', 0., PERIM), (f'F{j}_logs', math.log(.006), math.log(.025))]
            for j in range(nS): s += [(f'S{j}_x', -.05, .05), (f'S{j}_phi', 0., PERIM), (f'S{j}_logs', math.log(.006), math.log(.025)),
                                      (f'S{j}_logk', math.log(1500), math.log(24000))]
        names = [a for a, _, _ in g + f + s]
        bounds = np.array([[lo, hi] for _, lo, hi in g + f + s])
        return names, bounds

    def split(self, theta, nF, nS):
        th = torch.as_tensor(np.asarray(theta), dtype=self.dtype, device=self.device)
        ng = 3 if self.hier else 2
        pf = 1 if self.family == 'published' else 3; ps = 2 if self.family == 'published' else 4
        glob = th[:, :ng]
        F = th[:, ng:ng + pf * nF].reshape(len(th), nF, pf)
        S = th[:, ng + pf * nF:ng + pf * nF + ps * nS].reshape(len(th), nS, ps)
        tau = glob[:, 2].exp() if self.hier else torch.ones(len(th), dtype=self.dtype, device=self.device)
        return glob, F, S, tau

    def ordered(self, theta, nF, nS):
        _, F, S, _ = self.split(theta, nF, nS)
        ok = torch.ones(len(F), dtype=torch.bool, device=self.device)
        if nF > 1: ok &= (F[:, 1:, 0] > F[:, :-1, 0]).all(1)
        if nS > 1: ok &= (S[:, 1:, 0] > S[:, :-1, 0]).all(1)
        return ok

    def centres(self, xs, phis):
        c, n = surface_point(xs.detach().cpu().numpy(), phis.detach().cpu().numpy())
        return torch.as_tensor(c, dtype=self.dtype, device=self.device)

    def operators(self, theta, nF, nS):
        b = self.body; B = len(theta)
        glob, F, S, tau = self.split(theta, nF, nS)
        th5 = torch.zeros(B, 5, dtype=self.dtype, device=self.device); th5[:, :2] = glob[:, :2]
        E = b.material_modulus(th5)
        K = torch.zeros(B, b.ndof * b.ndof, dtype=self.dtype, device=self.device)
        K.index_add_(1, b._assembly_index, (E[:, :, None, None] * b.local_unit_stiffness[None]).flatten(1))
        K = K.view(B, b.ndof, b.ndof)
        diag = torch.zeros(B, b.n, dtype=self.dtype, device=self.device)
        cols = []
        for j in range(nS):
            if self.family == 'published':
                w = b.footprint(S[:, j, 0], 'bottom'); k = S[:, j, 1].exp()
            else:
                w = self.surf.gauss(self.centres(S[:, j, 0], S[:, j, 1]), S[:, j, 2].exp()); k = S[:, j, 3].exp()
            diag += k[:, None] * w
        K.diagonal(dim1=1, dim2=2).add_(diag.repeat_interleave(3, dim=1))
        for j in range(nF):
            if self.family == 'published':
                top = b.footprint(F[:, j, 0], 'top'); side = b.footprint(F[:, j, 0], 'side')
                cols += [(top[:, :, None] * top.new_tensor([0., 0., -1.])).flatten(1),
                         (side[:, :, None] * side.new_tensor([0., -1., 0.])).flatten(1)]
            else:
                w = self.surf.gauss(self.centres(F[:, j, 0], F[:, j, 1]), F[:, j, 2].exp())
                for e in torch.eye(3, dtype=self.dtype, device=self.device):
                    cols.append((w[:, :, None] * e).flatten(1))
        unknown = torch.stack(cols, -1) if cols else K.new_zeros(B, b.ndof, 0)
        return K, unknown, tau, E

    def project(self, U):
        """U [B,ndof,c] -> projected rows [B,60,c] and full 3D at 30 points [B,30,3,c]."""
        B, _, c = U.shape
        P = torch.einsum('pn,bnkc->bpkc', self.interp, U.view(B, self.body.n, 3, c))
        return P[:, self._row_point, self._row_comp], P

    def forward(self, theta, nF, nS, extras=False):
        K, unknown, tau, E = self.operators(theta, nF, nS)
        B = K.shape[0]; nA = unknown.shape[-1]
        rhs = [unknown, self.known[None].expand(B, -1, -1), self.probe_cols[None].expand(B, -1, -1)]
        if extras: rhs.append(self.target_cols[None].expand(B, -1, -1))
        rhs = torch.cat(rhs, -1)
        L, info = torch.linalg.cholesky_ex(K)
        bad = info != 0
        if bool(bad.any()):
            L[bad] = torch.eye(K.shape[1], dtype=self.dtype, device=self.device)
        U = torch.cholesky_solve(rhs, L); del K, L
        rows, pts = self.project(U)
        nk = self.known.shape[-1]; npb = self.probe_cols.shape[-1]
        out = dict(G=rows[:, :, :nA], D=rows[:, :, nA:nA + nk], Pr=rows[:, :self.n_visible, nA + nk:nA + nk + npb] * PROBE_FORCE,
                   tau=tau, bad=bad, nA=nA)
        if extras:
            out['G3'] = pts[:, :, :, :nA]; out['D3'] = pts[:, :, :, nA:nA + nk]
            out['Tgt3'] = pts[:, :, :, nA + nk + npb:] * PROBE_FORCE    # target commands at .08 N
            out['E'] = E
        return out

    # ---------------- likelihood
    def marginal(self, fw, data, conditionals=False):
        """Exact Gaussian amplitude marginal. data: y [T,60], mask [T,60] bool,
        cmd [ncmd,T], probes list of (index, y[40])."""
        dev, dt = self.device, self.dtype
        y = torch.as_tensor(data['y'], dtype=dt, device=dev); mask = torch.as_tensor(data['mask'], dtype=dt, device=dev)
        cmd = torch.as_tensor(data['cmd'], dtype=dt, device=dev).reshape(-1, self.T)
        s2 = NOISE ** 2; G = fw['G']; B = G.shape[0]; nA = fw['nA']; T = self.T
        d = torch.einsum('bmc,ct->btm', fw['D'], cmd) if cmd.shape[0] else torch.zeros(B, T, y.shape[1], dtype=dt, device=dev)
        r = (y[None] - d) * mask[None]
        n = float(mask.sum())
        quad = (r ** 2).sum((1, 2)) / s2
        ll = -.5 * (quad + n * math.log(2 * math.pi * s2))
        mu = S = None
        if nA:
            GtG = torch.einsum('bmi,tm,bmj->btij', G, mask, G) / s2           # [B,T,nA,nA]
            bvec = (torch.einsum('bmi,btm->bti', G, r) / s2).reshape(B, T * nA)
            lam = (fw['tau'] ** 2 * FORCE_STD ** 2)                          # [B]
            I = torch.eye(nA, dtype=dt, device=dev)
            P = torch.kron(self.Minv, I)[None] / lam[:, None, None]
            P = P.view(B, T, nA, T, nA).clone()
            P[:, torch.arange(T), :, torch.arange(T), :] += GtG.permute(1, 0, 2, 3)
            P = P.view(B, T * nA, T * nA)
            Lp = torch.linalg.cholesky(P)
            z = torch.linalg.solve_triangular(Lp, bvec[..., None], upper=False)[..., 0]
            logdetP = 2 * torch.log(torch.diagonal(Lp, dim1=1, dim2=2)).sum(1)
            logdetLam = T * nA * torch.log(lam) + nA * self.logdetM
            ll = ll + .5 * ((z ** 2).sum(1) - logdetLam - logdetP)
            if conditionals:
                mu = torch.cholesky_solve(bvec[..., None], Lp)[..., 0].view(B, T, nA)
                S = torch.cholesky_inverse(Lp)
        for index, yp in data.get('probes', []):
            yp = torch.as_tensor(yp, dtype=dt, device=dev)
            res = yp[None] - fw['Pr'][:, :, index]
            ll = ll - .5 * ((res ** 2).sum(1) / s2 + len(yp) * math.log(2 * math.pi * s2))
        ll = torch.where(fw['bad'], torch.full_like(ll, -math.inf), ll)
        return (ll, mu, S, d) if conditionals else ll

    def loglik(self, theta, nF, nS, data, batch=256):
        out = []
        for i in range(0, len(theta), batch):
            th = theta[i:i + batch]
            with torch.no_grad():
                fw = self.forward(th, nF, nS)
                ll = self.marginal(fw, data)
                ll = torch.where(self.ordered(th, nF, nS), ll, torch.full_like(ll, -math.inf))
            out.append(ll.cpu().numpy())
        return np.concatenate(out)
