"""Quadratic-tetrahedron (T10) elastodynamics for the shared stem trial.

SI units, world frame x along the stem, y up (gravity -y), z lateral.
Nonlinear solver: total-Lagrangian compressible neo-Hookean,
  P = mu (F - F^-T) + lam ln J F^-T,
which is the same strain-energy family as GIC's MPM "elasticity" material.
Time integration: Newmark average acceleration (beta=1/4, gamma=1/2) with
full Newton iterations. Linear solver: same element on the reference
configuration with small-strain stiffness; used only as our method's forward
model on a coarser mesh. Masses are exact consistent T10 masses.
Clamp: every node with reference x <= clamp_x is fixed (displacement zero).
"""
from __future__ import annotations
from itertools import permutations
from math import factorial
import numpy as np
import scipy.sparse as sp

try:
    import pypardiso
    class _Factor:
        """Pardiso factorization; native memory is released when the object is collected
        (pypardiso otherwise keeps every factorization alive until process exit)."""
        def __init__(self, A):
            self.s = pypardiso.PyPardisoSolver(); self.s.set_matrix_type(11)
            self.A = A.tocsr(); self.s.factorize(self.A)
        def __call__(self, b):
            return self.s.solve(self.A, b)
        def __del__(self):
            try:
                self.s.free_memory(everything=True)
            except Exception:
                pass
    def _factor(A):
        return _Factor(A)
except Exception:  # pragma: no cover
    from scipy.sparse.linalg import factorized
    def _factor(A):
        return factorized(A.tocsc())

# Geometry of the trial (metres).
X0, X1, CLAMP_X = -0.24, 0.20, -0.20
Y0, Y1, Z0, Z1 = 0.56, 0.64, -0.04, 0.04
L_FREE = X1 - CLAMP_X
EDGES = [(0, 1), (1, 2), (2, 0), (0, 3), (1, 3), (2, 3)]


def _monomials():
    """T10 shape functions as {exponent tuple: coefficient} in barycentric coords."""
    funcs = []
    for i in range(4):
        e2 = [0]*4; e2[i] = 2; e1 = [0]*4; e1[i] = 1
        funcs.append({tuple(e2): 2.0, tuple(e1): -1.0})
    for i, j in EDGES:
        e = [0]*4; e[i] += 1; e[j] += 1
        funcs.append({tuple(e): 4.0})
    return funcs


def _unit_mass():
    """Exact integral of N_a N_b over a tetrahedron divided by its volume."""
    f = _monomials(); M = np.zeros((10, 10))
    for a in range(10):
        for b in range(10):
            s = 0.0
            for ea, ca in f[a].items():
                for eb, cb in f[b].items():
                    e = [x+y for x, y in zip(ea, eb)]
                    s += ca*cb*6*np.prod([factorial(k) for k in e])/factorial(sum(e)+3)
            M[a, b] = s
    return M


UNIT_MASS = _unit_mass()


def shape(lam):
    """Shape values (n,10) and barycentric derivatives (n,10,4) at barycentric points."""
    lam = np.atleast_2d(lam); n = len(lam)
    N = np.zeros((n, 10)); dN = np.zeros((n, 10, 4))
    for i in range(4):
        N[:, i] = lam[:, i]*(2*lam[:, i]-1); dN[:, i, i] = 4*lam[:, i]-1
    for k, (i, j) in enumerate(EDGES):
        N[:, 4+k] = 4*lam[:, i]*lam[:, j]
        dN[:, 4+k, i] = 4*lam[:, j]; dN[:, 4+k, j] = 4*lam[:, i]
    return N, dN


QA, QB = 0.5854101966249685, 0.1381966011250105
QUAD = np.array([[QA, QB, QB, QB], [QB, QA, QB, QB], [QB, QB, QA, QB], [QB, QB, QB, QA]])
QW = np.full(4, .25)


def lattice(order=6):
    """Symmetric equal-weight barycentric points (i+1)/(K+4), sum(i)=K; exact for linears."""
    pts = [(i, j, k, order-i-j-k) for i in range(order+1) for j in range(order+1-i)
           for k in range(order+1-i-j)]
    lam = (np.asarray(pts, float)+1)/(order+4)
    return lam, np.full(len(lam), 1/len(lam))


class Mesh:
    def __init__(self, cells):
        nx, ny, nz = cells
        ax = [np.linspace(X0, X1, nx+1), np.linspace(Y0, Y1, ny+1), np.linspace(Z0, Z1, nz+1)]
        P = np.stack(np.meshgrid(*ax, indexing='ij'), -1).reshape(-1, 3)
        ids = np.arange(len(P)).reshape(nx+1, ny+1, nz+1)
        tet = []
        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    o = np.array([i, j, k])
                    for order in permutations(range(3)):
                        v = [o.copy()]
                        for a in order:
                            v.append(v[-1].copy()); v[-1][a] += 1
                        tet.append([ids[tuple(q)] for q in v])
        tet = np.asarray(tet)
        d = np.linalg.det(np.stack([P[tet[:, a]]-P[tet[:, 0]] for a in (1, 2, 3)], -1))
        neg = d < 0; tet[neg, 1], tet[neg, 2] = tet[neg, 2].copy(), tet[neg, 1].copy()
        edges = np.sort(np.stack([tet[:, [i, j]] for i, j in EDGES], 1).reshape(-1, 2), 1)
        uniq, inv = np.unique(edges, axis=0, return_inverse=True)
        mid = (P[uniq[:, 0]]+P[uniq[:, 1]])/2
        self.X = np.vstack([P, mid])
        self.elem = np.hstack([tet, len(P)+inv.reshape(-1, 6)])
        self.n = len(self.X); self.cells = cells
        Xe = self.X[self.elem[:, :4]]
        J = np.stack([Xe[:, a]-Xe[:, 0] for a in (1, 2, 3)], -1)  # dX/d(l1,l2,l3)
        self.vol = np.linalg.det(J)/6
        assert (self.vol > 0).all()
        Jinv = np.linalg.inv(J)  # d(l1,l2,l3)/dX
        dl = np.zeros((len(self.elem), 4, 3)); dl[:, 1:] = Jinv; dl[:, 0] = -Jinv.sum(1)
        self.dlam = dl  # d lambda_i / dX  (E,4,3)
        _, dNq = shape(QUAD)
        self.G = np.einsum('qai,eij->eqaj', dNq, dl)  # (E,4,10,3) dN/dX at quad points
        dof = (3*self.elem[:, :, None]+np.arange(3)).reshape(len(self.elem), 30)
        self.dof = dof
        r = np.repeat(dof, 30, 1).ravel(); c = np.tile(dof, (1, 30)).ravel()
        pat = sp.coo_matrix((np.ones(len(r)), (r, c)), shape=(3*self.n,)*2).tocsr()
        pat.sum_duplicates(); pat.sort_indices()
        self.pattern = pat
        key = r.astype(np.int64)*(3*self.n)+c
        pkey = (np.repeat(np.arange(3*self.n), np.diff(pat.indptr)).astype(np.int64)*(3*self.n)+pat.indices)
        self.slot = np.searchsorted(pkey, key)
        assert (pkey[self.slot] == key).all()
        self.fixed = np.where(self.X[:, 0] <= CLAMP_X+1e-9)[0]
        fd = np.zeros(3*self.n, bool); fd[(3*self.fixed[:, None]+np.arange(3)).ravel()] = True
        self.free = np.where(~fd)[0]
        # Consistent unit-density mass.
        Me = self.vol[:, None, None]*UNIT_MASS
        Me3 = np.einsum('eab,ij->eaibj', Me, np.eye(3)).reshape(-1, 900)
        self.M1 = self._assemble(Me3)
        # Lattice sample points for indicator-weighted integrals.
        lam, w = lattice(6)
        Ns, _ = shape(lam)
        self.sample_N = Ns; self.sample_w = w
        Xe10 = self.X[self.elem]
        self.sample_X = np.einsum('sa,eaj->esj', Ns, Xe10)  # (E,S,3)

    def _assemble(self, data):
        v = np.bincount(self.slot, weights=np.asarray(data).ravel(), minlength=self.pattern.nnz)
        A = self.pattern.copy(); A.data = v
        return A

    def region_weights(self, lo, hi):
        """Node weights w with sum 1: w @ x = volume average of x over lo<=X_ref<hi."""
        m = (self.sample_X[..., 0] >= lo) & (self.sample_X[..., 0] < hi)
        cw = self.vol[:, None]*self.sample_w[None]*m  # (E,S)
        nw = np.einsum('es,sa->ea', cw, self.sample_N)
        w = np.bincount(self.elem.ravel(), weights=nw.ravel(), minlength=self.n)
        return w/cw.sum(), float(cw.sum())

    def body_force_region(self, lo, hi, force):
        """Nodal forces for total force vector spread uniformly over lo<=X_ref<hi."""
        w, vol = self.region_weights(lo, hi)
        return (w[:, None]*np.asarray(force, float)[None]).ravel()

    def gravity(self, rho, g=9.8):
        f = np.zeros((self.n, 3))
        ints = np.r_[np.full(4, -1/20), np.full(6, 1/5)]
        np.add.at(f[:, 1], self.elem.ravel(), (-rho*g*self.vol[:, None]*ints).ravel())
        return f.ravel()

    def interpolation(self, points):
        """Sparse matrix (P,n) evaluating the T10 field at reference points."""
        points = np.atleast_2d(points); rows, cols, vals = [], [], []
        Xc = self.X[self.elem[:, 0]]
        for p_i, p in enumerate(points):
            lam123 = np.einsum('eij,ej->ei', self.dlam[:, 1:], p-Xc)
            lam = np.column_stack([1-lam123.sum(1), lam123])
            e = int(np.argmax(lam.min(1)))
            assert lam[e].min() > -1e-9, p
            N, _ = shape(lam[e][None])
            rows += [p_i]*10; cols += list(self.elem[e]); vals += list(N[0])
        return sp.csr_matrix((vals, (rows, cols)), shape=(len(points), self.n))

    def surface_triangles(self):
        """Boundary faces subdivided into 4 triangles using midside nodes."""
        faces = [(0, 1, 2, 4, 5, 6), (0, 1, 3, 4, 8, 7), (1, 2, 3, 5, 9, 8), (2, 0, 3, 6, 7, 9)]
        # edge index k -> local midside node 4+k: (0,1)->4,(1,2)->5,(2,0)->6,(0,3)->7,(1,3)->8,(2,3)->9
        allf = np.concatenate([self.elem[:, list(f)] for f in faces])
        key = np.sort(allf[:, :3], 1)
        _, inv, cnt = np.unique(key, axis=0, return_inverse=True, return_counts=True)
        b = allf[cnt[inv] == 1]
        a, bb, c, ab, bc, ca = b.T
        tris = np.concatenate([np.stack([a, ab, ca], 1), np.stack([ab, bb, bc], 1),
                               np.stack([ca, bc, c], 1), np.stack([ab, bc, ca], 1)])
        p = self.X[tris]; nrm = np.cross(p[:, 1]-p[:, 0], p[:, 2]-p[:, 0])
        centre = np.array([(X0+X1)/2, (Y0+Y1)/2, (Z0+Z1)/2])
        flip = (nrm*(p.mean(1)-centre[None])).sum(1) < 0
        tris[flip, 1], tris[flip, 2] = tris[flip, 2].copy(), tris[flip, 1].copy()
        return tris


def lame(E, nu):
    return E/(2*(1+nu)), E*nu/((1+nu)*(1-2*nu))


def neo_hookean(F, mu, lam):
    """First Piola stress and tangent dP/dF for arrays F (...,3,3)."""
    J = np.linalg.det(F); Fi = np.linalg.inv(F); FiT = np.swapaxes(Fi, -1, -2)
    lnJ = np.log(J)
    P = mu*(F-FiT)+lam*lnJ[..., None, None]*FiT
    I = np.eye(3)
    A = (mu*np.einsum('ik,JL->iJkL', I, I)[None]
         + (mu-lam*lnJ)[..., None, None, None, None]*np.einsum('...Jk,...Li->...iJkL', Fi, Fi)
         + lam*np.einsum('...Ji,...Lk->...iJkL', Fi, Fi))
    return P, A


class Elastic:
    def __init__(self, mesh: Mesh, E, nu, rho):
        self.m = mesh; self.E, self.nu, self.rho = E, nu, rho
        self.mu, self.lam = lame(E, nu)
        self.M = rho*mesh.M1
        self.wq = mesh.vol[:, None]*QW[None]  # (E,4)

    def internal(self, u, tangent=True):
        m = self.m
        ue = u.reshape(-1, 3)[m.elem]  # (E,10,3)
        F = np.eye(3)+np.einsum('eai,eqaJ->eqiJ', ue, m.G)
        if (np.linalg.det(F) <= 0).any():
            raise FloatingPointError('inverted element')
        P, A = neo_hookean(F, self.mu, self.lam)
        fe = np.einsum('eq,eqiJ,eqaJ->eai', self.wq, P, m.G).reshape(-1, 30)
        f = np.bincount(m.dof.ravel(), weights=fe.ravel(), minlength=3*m.n)
        if not tangent:
            return f, None
        Ke = np.einsum('eq,eqaJ,eqiJkL,eqbL->eaibk', self.wq, m.G, A, m.G, optimize=True).reshape(-1, 900)
        return f, m._assemble(Ke)

    def linear_K(self):
        _, K = self.internal(np.zeros(3*self.m.n))
        return K


def newmark(model: Elastic, force_fn, t_end, dt, save_every, nonlinear=True, tol=1e-8, max_iter=25, log=None):
    """Integrate M a + f_int(u) = f_ext(t) from rest at u=0. Returns saved displacements (S,n,3)."""
    m = model.m; fr = m.free; n3 = 3*m.n
    M = model.M.tocsr()[fr][:, fr]
    u = np.zeros(n3); v = np.zeros(n3); a = np.zeros(n3)
    f0 = force_fn(0.0)
    a[fr] = _factor(M)(f0[fr])
    steps = int(round(t_end/dt)); out = [u.reshape(-1, 3).copy()]
    c0 = 4/dt**2
    if not nonlinear:
        K = model.linear_K().tocsr()[fr][:, fr]
        solve = _factor(K+c0*M)
    stats = []; fac = None; slow = True
    for s in range(1, steps+1):
        t = s*dt; fe = force_fn(t)[fr]
        un = u[fr].copy(); vn = v[fr]; an = a[fr]
        pred_const = c0*un+4/dt*vn+an  # a_{n+1} = c0*u_{n+1} - pred_const
        if nonlinear:
            # Modified Newton: the factorized tangent is reused and refreshed when
            # convergence slows; the converged residual tolerance is unchanged.
            x = un+dt*vn+dt*dt/4*an
            for it in range(max_iter):
                u[fr] = x
                fint, _ = model.internal(u, tangent=False)
                r = M@(c0*x-pred_const)+fint[fr]-fe
                nr = np.linalg.norm(r)
                if it == 0:
                    r0 = max(nr, np.linalg.norm(fe), 1e-12)
                if nr <= tol*r0:
                    break
                if fac is None or slow or it in (6, 12, 18):
                    _, K = model.internal(u)
                    fac = _factor(K.tocsr()[fr][:, fr]+c0*M); slow = False
                x = x+fac(-r)
            else:
                raise RuntimeError(f'Newton did not converge at t={t:.5f}: {nr:.3e}/{r0:.3e}')
            slow = it > 4
            stats.append(it)
        else:
            x = solve(fe+M@pred_const)
        a_new = c0*x-pred_const
        v[fr] = vn+dt/2*(an+a_new); a[fr] = a_new; u[fr] = x
        if s % save_every == 0:
            out.append(u.reshape(-1, 3).copy())
            if log:
                log(f't={t:.4f} newton_mean={np.mean(stats[-save_every:]) if stats else 0:.2f}')
    return np.stack(out), stats


def smooth_ramp(t, t_ramp):
    s = np.clip(t/t_ramp, 0, 1)
    return .5*(1-np.cos(np.pi*s))
