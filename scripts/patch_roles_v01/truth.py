"""Independent fine-mesh truth generator (avoids the inverse crime).

Truth uses a (16,8,8)-cell tetrahedral mesh (1377 nodes, 4131 dof, 6144 tets) versus
the (8,4,4) inference mesh (225 nodes, 768 tets); footprints are re-integrated on
the fine surface quadrature; the log-modulus offset is corrected for the mesh's
volume centring (as in joint_field_identification.fine_reference). Optional model
mismatch: top-hat (uniform disc) traction footprints, pressure-only short-correlation
force histories, and a stiffening (nonlinear) support k(u)=k0(1+|u|^2/u0^2) solved by
Picard iteration. Noise N(0, 0.02 mm) is added to camera rows only.
"""
from __future__ import annotations
import numpy as np
import torch
from scripts.joint_field_identification.mechanics import LinearGaussianBody
from scripts.patch_roles_v01 import model as M

FINE_CELLS = (16, 8, 8)


def draw_history(rng, modes, scale=1., length=M.LENGTH):
    C = M.FORCE_STD ** 2 * scale ** 2 * M.matern(length=length)
    L = np.linalg.cholesky(C + 1e-14 * np.eye(len(C)))
    return (L @ rng.standard_normal((len(C), modes)))            # [T, modes]


class FineTruth:
    def __init__(self, device='cuda', cells=FINE_CELLS):
        self.body = LinearGaussianBody('block', cells=cells, device=device)
        self.surf = M.Surface(self.body); self.dev = self.body.device; self.dt = self.body.dtype
        coarse = LinearGaussianBody('block', M.COARSE_CELLS, device='cpu')
        self.center_shift = float(self.body.material_log_center) - float(coarse.material_log_center)
        self.interp = self.body.interpolation(np.concatenate(M.obs_points()))
        rows = []
        for i in range(10): rows += [(i, 0), (i, 2)]
        for i in range(10, 20): rows += [(i, 0), (i, 1)]
        for i in range(20, 30): rows += [(i, 0), (i, 2)]
        self.rows = np.array(rows)

    def footprint(self, p):
        b = self.body
        if p.get('kind') == 'pub':
            face = 'bottom' if p['role'] == 'support' else None
            if face: return b.footprint(torch.tensor([p['x']], dtype=self.dt, device=self.dev), 'bottom')[0]
            return None
        c, _ = M.surface_point(p['x'], p['phi'])
        if p.get('kind') == 'tophat': return self.surf.tophat(c[None], p['radius'])[0]
        return self.surf.gauss(c[None], [p['width']])[0]

    def force_columns(self, p):
        b = self.body
        if p.get('kind') == 'pub':
            x = torch.tensor([p['x']], dtype=self.dt, device=self.dev)
            top = b.footprint(x, 'top')[0]; side = b.footprint(x, 'side')[0]
            return torch.stack([(top[:, None] * top.new_tensor([0., 0., -1.])).flatten(),
                                (side[:, None] * side.new_tensor([0., -1., 0.])).flatten()], -1)
        w = self.footprint(p)
        return torch.stack([(w[:, None] * e).flatten() for e in torch.eye(3, dtype=self.dt, device=self.dev)], -1)

    def solve(self, spec, F, iterations=60):
        """F [ndof, nrhs] -> displacements [nrhs, n, 3]; nonlinear supports via Picard per column."""
        b = self.body
        th5 = torch.zeros(1, 5, dtype=self.dt, device=self.dev)
        th5[0, 0] = spec['logE'] + spec['m1'] * self.center_shift; th5[0, 1] = spec['m1']
        E = b.material_modulus(th5)
        K = torch.zeros(b.ndof * b.ndof, dtype=self.dt, device=self.dev)
        K.index_add_(0, b._assembly_index, (E[0, :, None, None] * b.local_unit_stiffness).flatten())
        K = K.view(b.ndof, b.ndof)
        sup = [(p, self.footprint(p)) for p in spec['patches'] if p['role'] == 'support']
        lin = torch.zeros(b.n, dtype=self.dt, device=self.dev)
        for p, w in sup: lin += p['k'] * w
        U = torch.cholesky_solve(F, torch.linalg.cholesky(K + torch.diag(lin.repeat_interleave(3))))
        nonlin = [(p, w) for p, w in sup if p.get('u0')]
        if nonlin:
            out = []
            for c in range(F.shape[1]):
                u = U[:, c]
                for it in range(iterations):
                    mag2 = (u.view(b.n, 3) ** 2).sum(1)
                    d = torch.zeros(b.n, dtype=self.dt, device=self.dev)
                    for p, w in sup:
                        d += p['k'] * w * (1 + mag2 / p['u0'] ** 2 if p.get('u0') else 1.)
                    new = torch.linalg.solve(K + torch.diag(d.repeat_interleave(3)), F[:, c])
                    if float((new - u).norm() / (new.norm() + 1e-30)) < 1e-10: u = new; break
                    u = .5 * u + .5 * new
                out.append(u)
            U = torch.stack(out, 1)
        return U.T.reshape(-1, b.n, 3)

    def project(self, U):
        P = torch.einsum('pn,cnk->cpk', self.interp, U).cpu().numpy()      # [c,30,3]
        return P[:, self.rows[:, 0], self.rows[:, 1]], P

    def generate(self, spec, seed):
        rng = np.random.default_rng(seed)
        b = self.body; T = len(M.TIMES)
        cols, amps = [], []
        for p in spec['patches']:
            if p['role'] != 'force': continue
            Fp = self.force_columns(p); cols.append(Fp)
            a = np.asarray(p['history'], float)                            # [T, modes]
            amps.append(a)
        known = [M.load_vector(b, self.surf, c) for c in spec.get('commands', [])]
        cmd = np.asarray(spec.get('cmd', np.zeros((0, T))), float).reshape(-1, T)
        Ft = torch.zeros(b.ndof, T, dtype=self.dt, device=self.dev)
        for Fp, a in zip(cols, amps): Ft += Fp @ torch.as_tensor(a.T, dtype=self.dt, device=self.dev)
        for f, c in zip(known, cmd): Ft += f[:, None] * torch.as_tensor(c, dtype=self.dt, device=self.dev)[None]
        rows, pts = self.project(self.solve(spec, Ft))                     # [T,60], [T,30,3]
        noise = rng.normal(scale=M.NOISE, size=rows.shape)
        probes = torch.stack([M.load_vector(b, self.surf, c) for c in M.PROBES], -1) * M.PROBE_FORCE
        prow, _ = self.project(self.solve(spec, probes))
        targets = torch.stack([M.load_vector(b, self.surf, c) for c in M.TARGET_COMMANDS], -1) * M.PROBE_FORCE
        _, tpts = self.project(self.solve(spec, targets))
        return dict(rows_clean=rows, y=rows + noise, hidden=pts[:, 20:30], all3d=pts,
                    probe_clean=prow[:, :40], probe_noise=rng.normal(scale=M.NOISE, size=(len(M.PROBES), 40)),
                    target3d=tpts, cmd=cmd)
