"""Rigid spherical striker in volumetric penalty contact with a T10 body (Figure-1 scenario 3).

Contact law (identical for every method): force density on body material inside the sphere
    b(x) = kappa * (r - |x-c|) * (x-c)/|x-c|      [N/m^3],
integrated over the body's material volume; the sphere receives the opposite total force.
The sphere (mass m, no gravity: it moves on a horizontal guide) is added to the implicit Newmark
system as 3 extra unknowns; full Newton with the analytic contact tangent.
"""
from __future__ import annotations
import numpy as np
import scipy.sparse as sp
import fem


class Striker:
    def __init__(self, radius, mass, kappa, c0, v0):
        self.r, self.m, self.k = radius, mass, kappa
        self.c0, self.v0 = np.asarray(c0, float), np.asarray(v0, float)


def contact_terms(mesh, u, c, s: Striker, cand):
    """Body nodal force, sphere force and tangent blocks for candidate elements `cand`."""
    n3 = 3*mesh.n
    fb = np.zeros(n3); fs = np.zeros(3)
    if len(cand) == 0:
        return fb, fs, None
    el = mesh.elem[cand]; N = mesh.sample_N  # (S,10)
    xe = (mesh.X+u.reshape(-1, 3))[el]  # (C,10,3)
    xs = np.einsum('sa,caj->csj', N, xe)
    d = xs-c; dist = np.linalg.norm(d, axis=-1); inside = dist < s.r
    if not inside.any():
        return fb, fs, None
    ci, si = np.nonzero(inside)
    nrm = d[ci, si]/dist[ci, si, None]; delta = s.r-dist[ci, si]
    w = mesh.vol[cand][ci]*mesh.sample_w[si]
    b = s.k*delta[:, None]*nrm  # force density
    Na = N[si]  # (P,10)
    fnode = np.einsum('p,pa,pj->paj', w, Na, b)
    np.add.at(fb.reshape(-1, 3), el[ci], fnode)
    fs = -(w[:, None]*b).sum(0)
    G = s.k*(-np.einsum('pi,pj->pij', nrm, nrm)+(delta/dist[ci, si])[:, None, None]*(np.eye(3)[None]-np.einsum('pi,pj->pij', nrm, nrm)))
    # d f_body_a / d u_b = sum w Na Nb G ; d f_body_a / d c = -sum w Na G ; d f_s/du_b = -sum w G Nb ; d f_s/dc = sum w G
    nodes = el[ci]  # (P,10)
    Kuu = np.einsum('p,pa,pb,pij->paibj', w, Na, Na, G)
    rows = (3*nodes[:, :, None, None, None]+np.arange(3)[None, None, :, None, None])
    cols = (3*nodes[:, None, None, :, None]+np.arange(3)[None, None, None, None, :])
    rows, cols = np.broadcast_arrays(rows, cols)
    Kuu = sp.coo_matrix((Kuu.ravel(), (rows.ravel(), cols.ravel())), shape=(n3, n3)).tocsr()
    Kuc = -np.einsum('p,pa,pij->paij', w, Na, G)  # (P,10,3,3): row node dof i, col sphere j
    Kuc_full = np.zeros((n3, 3)); np.add.at(Kuc_full.reshape(-1, 3, 3), nodes, Kuc)
    Kcu_full = Kuc_full.T  # d f_s/du_b = -sum w G Nb = Kuc^T (G symmetric)
    Kcc = (w[:, None, None]*G).sum(0)
    return fb, fs, (Kuu, Kuc_full, Kcu_full, Kcc)


def candidates(mesh, u, c, s, margin=0.03):
    cen = (mesh.X+u.reshape(-1, 3))[mesh.elem[:, :4]].mean(1)
    return np.where(np.linalg.norm(cen-c, axis=1) < s.r+margin)[0]


def newmark_striker(model: fem.Elastic, force_fn, s: Striker, t_end, dt, save_every, nonlinear=True,
                    tol=1e-8, max_iter=30, log=None):
    """Returns saved body displacements (F,n,3), sphere centres (F,3), sphere velocities (F,3), stats."""
    m = model.m; fr = m.free; n3 = 3*m.n; nf = len(fr)
    M = model.M.tocsr()[fr][:, fr]
    u = np.zeros(n3); v = np.zeros(n3); a = np.zeros(n3)
    c = s.c0.copy(); vc = s.v0.copy(); ac = np.zeros(3)
    a[fr] = fem._factor(M)(force_fn(0.0)[fr])
    c0 = 4/dt**2; steps = int(round(t_end/dt))
    Klin = model.linear_K().tocsr()[fr][:, fr] if not nonlinear else None
    Kel = Klin; refresh = True
    U, C, VC = [u.reshape(-1, 3).copy()], [c.copy()], [vc.copy()]; stats = []; cache = {}
    for step in range(1, steps+1):
        t = step*dt; fe = force_fn(t)
        pu = c0*u[fr]+4/dt*v[fr]+a[fr]; pc = c0*c+4/dt*vc+ac
        x = u[fr]+dt*v[fr]+dt*dt/4*a[fr]; y = c+dt*vc+dt*dt/4*ac
        for it in range(max_iter):
            uu = u.copy(); uu[fr] = x
            cand = candidates(m, uu, y, s)
            fb, fs, J = contact_terms(m, uu, y, s, cand)
            if nonlinear:
                fint, _ = model.internal(uu, tangent=False); fint = fint[fr]
            else:
                fint = Klin@x
            ru = M@(c0*x-pu)+fint-fe[fr]-fb[fr]
            rc = s.m*(c0*y-pc)-fs
            nr = np.sqrt(ru@ru+rc@rc)
            if it == 0:
                r0 = max(nr, np.linalg.norm(fe[fr]), 1e-12)
            if nr <= tol*r0:
                break
            if nonlinear and (refresh or it in (5, 10, 15)):
                _, Kt = model.internal(uu); Kel = Kt.tocsr()[fr][:, fr]; refresh = False
            A = Kel+c0*M
            if J is None:
                if nonlinear or cache.get('f') is None:
                    cache['f'] = fem._factor(A)  # linear model without contact: one factorization reused
                dxu = cache['f'](-ru); dyc = -rc/(s.m*c0)
            else:
                Kuu, Kuc, Kcu, Kcc = J
                A = A-Kuu[fr][:, fr]
                top = sp.hstack([A, sp.csr_matrix(-Kuc[fr])]); bot = sp.hstack([sp.csr_matrix(-Kcu[:, fr]), sp.csr_matrix(s.m*c0*np.eye(3)-Kcc)])
                sol = fem._factor(sp.vstack([top, bot]).tocsr())(-np.concatenate([ru, rc]))
                dxu, dyc = sol[:nf], sol[nf:]
            x = x+dxu; y = y+dyc
        else:
            raise RuntimeError(f'striker Newton failed t={t:.5f} {nr:.3e}/{r0:.3e}')
        refresh = refresh or it > 4
        an = c0*x-pu; acn = c0*y-pc
        v[fr] = v[fr]+dt/2*(a[fr]+an); a[fr] = an; u[fr] = x
        vc = vc+dt/2*(ac+acn); ac = acn; c = y
        stats.append(it)
        if step % save_every == 0:
            U.append(u.reshape(-1, 3).copy()); C.append(c.copy()); VC.append(vc.copy())
            if log:
                log(f't={t:.4f} c_z={c[2]:.4f} v_z={vc[2]:+.3f} iters={np.mean(stats[-save_every:]):.1f}')
    return np.stack(U), np.stack(C), np.stack(VC), stats
