"""Forward model of 'ours' (protocol amendment 2026-11-14, methods_v01.ours) on real PokeFlex pokes.

Quasi-static FEM of the poke's rest scan (start frame, free input):
  geometry  voxel grid (cell h), cell kept if its centre is inside the scan (upward +y ray crosses the scan an odd
            number of times; majority of 3 rays offset by ~10 um against exact edge hits) and above the support plane
            y_s = min scan y; largest face-connected cell component kept; each hex -> 6 linear tets (Kuhn split,
            conforming); density = catalogue mass / voxel volume.
  embedding scan vertices carry barycentric weights of the tet that contains them (or the nearest tet, extrapolated)
            -> predicted surface keeps the scan topology.
  material  homogeneous compressible neo-Hookean, P = mu (F - F^-T) + lam ln J F^-T (scripts/stem_trial_v01/fem.py).
  support   grid nodes on the support plane (bottom node layer, y = y_s) fixed in all directions (bonded).
  gravity   rho g, pre-loaded: equilibrium under gravity first, then the poke.  The scan is the loaded shape, so the
            returned surface is scan + (u - u_gravity); contact geometry uses the loaded configuration X + u.
  tool      rigid flat-ended cylinder, radius 10 mm, axis = T_WT z (approach), end face at the T_WT origin,
            body extending 192 mm back up the axis.  Penalty contact on contact points = embedded scan vertices and the
            grid nodes inside the scan.  A penetrating point is pushed out along the cheaper direction (out
            of the end face along +axis, or radially), where the direction the point did NOT come from (its side of the
            tool at the previous equilibrium: below the face plane -> face, beside the body -> radial) costs K_SWITCH
            times its distance; d = min of the two weighted distances is continuous.  Penalty energy kappa*phi(d), C2 at
            onset (cubic over PEN_SMOOTH).  Point stiffness kappa = c_pen * E * A_p / h (A_p: scan vertex area, h^2 for
            grid nodes) -> pressure stiffness c_pen * E / h; proportional to E.  Frictionless.
  solve     per robot frame from start to end frame, continuation from the previous frame; tool motion between frames
            sub-stepped (<= dmax tip travel while in contact; predictor = last sub-step increment; bisection up to
            maxdepth on failure; at the deepest level a failed sub-step is discarded and the last equilibrium kept);
            Newton (exact tangent, pardiso symmetric-indefinite, shifted if not a descent direction) on the total
            potential (elastic + penalty - gravity work), Armijo backtracking line search with a positive-Jacobian
            guard (every element det F > 0, so no element ever inverts).  converged[i] = residual tolerance met at
            frame i; path_ok[i] = every sub-step to frame i converged.  Neither is hidden.
Force ON THE TOOL (world, N) = -(sum of penalty forces on the object).

API: build(obj, take, poke, h=None) -> RealMesh;  simulate(mesh, obj, take, poke, nu, E=None, frames=None) -> dict.
E=None runs at E_REF (Pa).  The forward model is the requested 'simulate(..., E)' but its default is E_REF instead of
E = 1.0 Pa: with gravity on, E = 1 Pa makes the object collapse under its own weight (rho g H / E ~ 60).  Force scales
as E and the deformation is E-independent up to gravity (checked in `dev`).

CLI:  python -I realfem.py dev [--nu 0.3 ...] [--pokes 1 2] [--part mesh|sims|escale|report|all]
      python -I realfem.py meshcheck   (build every poke of every protocol object; free inputs only)
  development take FoamDice/FoamDice_T1 only; outputs under runs/pokeflex-v01/dev/realfem/.
"""
from __future__ import annotations
import json, sys, time
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from scipy import ndimage
from scipy.spatial import cKDTree

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent/'stem_trial_v01'))
import pokeflex as P  # noqa: E402
from fem import neo_hookean, lame  # noqa: E402
import pypardiso  # noqa: E402

R_TOOL, L_TOOL, G_ACC = 0.010, 0.192, 9.81
E_REF = 3.0e4          # Pa; reference modulus of the one-simulation-per-nu runs (not fitted)
K_SWITCH = 100.0      # contact: cost factor for pushing a point out of the side it did not come from (5 let
#                       the scan surface slip around the tool rim near peak load on the dev take: force "snaps")
PEN_SMOOTH = 1e-4      # m; penalty onset smoothing length (C2 contact energy)
TOL_REL = 1e-4        # Newton: |residual| <= TOL_REL (E h^2 + m g)  (~2e-4 N for the dev take at E_REF)
N_TARGET_CELLS = 8000  # target voxel count for the automatic cell size
OBJECTS = json.loads((HERE/'objects.json').read_text())['objects']
DEV_OBJ, DEV_TAKE = 'FoamDice', 'FoamDice_T1'
OUTDIR = P.OUT/'dev/realfem'
KUHN = [(0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0)]
RAY_JITTER = [(0.0, 0.0), (1.3e-5, 0.7e-5), (-0.6e-5, 1.1e-5)]  # metres, (x, z) offsets of the 3 voting rays


# ---------------------------------------------------------------- geometry
def _column_crossings(V, F, x0, z0, h, nx, nz, jx, jz):
    """Upward (+y) rays at columns (x0+i*h+jx, z0+k*h+jz): returns (col = i*nz+k, y) of every ray-triangle hit."""
    A, B, C = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    T = np.stack([A, B, C], 1)
    mn, mx = T.min(1), T.max(1)
    i0 = np.clip(np.ceil((mn[:, 0]-x0-jx)/h), 0, nx).astype(int); i1 = np.clip(np.floor((mx[:, 0]-x0-jx)/h), -1, nx-1).astype(int)
    k0 = np.clip(np.ceil((mn[:, 2]-z0-jz)/h), 0, nz).astype(int); k1 = np.clip(np.floor((mx[:, 2]-z0-jz)/h), -1, nz-1).astype(int)
    ni, nk = np.maximum(i1-i0+1, 0), np.maximum(k1-k0+1, 0); cnt = ni*nk
    t = np.repeat(np.arange(len(F)), cnt)
    loc = np.arange(cnt.sum())-np.repeat(np.cumsum(cnt)-cnt, cnt)
    ii = i0[t]+loc % ni[t]; kk = k0[t]+loc//ni[t]
    px, pz = x0+ii*h+jx, z0+kk*h+jz
    a, b, c = A[t], B[t], C[t]
    e1x, e1z, e2x, e2z = b[:, 0]-a[:, 0], b[:, 2]-a[:, 2], c[:, 0]-a[:, 0], c[:, 2]-a[:, 2]
    det = e1x*e2z-e2x*e1z
    ok = np.abs(det) > 1e-18
    det = np.where(ok, det, 1.0)
    qx, qz = px-a[:, 0], pz-a[:, 2]
    u = (qx*e2z-e2x*qz)/det; v = (e1x*qz-qx*e1z)/det
    hit = ok & (u >= 0) & (v >= 0) & (u+v <= 1)
    y = a[:, 1]+u*(b[:, 1]-a[:, 1])+v*(c[:, 1]-a[:, 1])
    return (ii*nz+kk)[hit], y[hit]


def _inside_columns(V, F, x0, z0, h, nx, nz, jx, jz, qcol, qy):
    """Parity of upward-ray crossings above query points (qcol column index, qy height)."""
    col, y = _column_crossings(V, F, x0, z0, h, nx, nz, jx, jz)
    ylo, span = min(y.min(), qy.min())-1.0, (max(y.max(), qy.max())-min(y.min(), qy.min()))+2.0
    key = np.sort(col+(y-ylo)/span)
    above = np.searchsorted(key, qcol+1.0, 'left')-np.searchsorted(key, qcol+(qy-ylo)/span, 'right')
    return above % 2 == 1


def _inside_vote(V, F, x0, z0, h, nx, nz, qcol, qy):
    votes = sum(_inside_columns(V, F, x0, z0, h, nx, nz, jx, jz, qcol, qy).astype(int) for jx, jz in RAY_JITTER)
    return votes >= 2


def _voxelise(V, F, h, ys):
    lo, hi = V.min(0), V.max(0)
    nx = int(np.ceil((hi[0]-lo[0])/h))+2; nz = int(np.ceil((hi[2]-lo[2])/h))+2; ny = int(np.ceil((hi[1]-ys)/h))+1
    x0 = 0.5*(lo[0]+hi[0])-0.5*nx*h; z0 = 0.5*(lo[2]+hi[2])-0.5*nz*h; y0 = ys
    I, J, K = np.meshgrid(np.arange(nx), np.arange(ny), np.arange(nz), indexing='ij')
    # cell centres: column (i,k) at x0+(i+.5)h -> use a grid shifted by h/2
    qcol = (I*nz+K).ravel(); qy = (y0+(J+0.5)*h).ravel()
    ins = _inside_vote(V, F, x0+0.5*h, z0+0.5*h, h, nx, nz, qcol, qy).reshape(nx, ny, nz)
    return ins, (x0, y0, z0), (nx, ny, nz)


def _choose_h(V, F, ys):
    lo, hi = V.min(0), V.max(0)
    h0 = (np.prod(hi-lo)/N_TARGET_CELLS)**(1/3)
    ins, _, _ = _voxelise(V, F, h0, ys)
    vol = ins.sum()*h0**3
    return float(np.clip(np.round((vol/N_TARGET_CELLS)**(1/3)*2000)/2000, 0.005, 0.015))


class RealMesh:
    pass


def build(obj, take, poke, h=None):
    """Voxel tet mesh of the rest scan at the poke's start frame (free input) with embedded scan vertices."""
    t0 = time.time()
    V, F, _ = P.rest_mesh(obj, take, poke)
    ys = float(V[:, 1].min())
    if h is None:
        h = _choose_h(V, F, ys)
    ins, (x0, y0, z0), (nx, ny, nz) = _voxelise(V, F, h, ys)
    n_inside_raw = int(ins.sum())
    lab, nl = ndimage.label(ins)  # 6-connectivity = face connectivity
    if nl > 1:
        sizes = np.bincount(lab.ravel())[1:]
        ins = lab == (1+int(np.argmax(sizes)))
    cells = np.argwhere(ins)  # (nc,3) i,j,k
    # nodes and Kuhn tets
    nid = -np.ones((nx+1, ny+1, nz+1), np.int64)
    corner = np.array([[a, b, c] for a in (0, 1) for b in (0, 1) for c in (0, 1)])
    allc = (cells[:, None, :]+corner[None]).reshape(-1, 3)
    uc = np.unique(allc, axis=0)
    nid[uc[:, 0], uc[:, 1], uc[:, 2]] = np.arange(len(uc))
    X = np.c_[x0+uc[:, 0]*h, y0+uc[:, 1]*h, z0+uc[:, 2]*h]
    tets = []
    for order in KUHN:
        v = [cells.copy()]
        for ax in order:
            w = v[-1].copy(); w[:, ax] += 1; v.append(w)
        tets.append(np.stack([nid[w[:, 0], w[:, 1], w[:, 2]] for w in v], 1))
    T = np.stack(tets, 1).reshape(-1, 4)  # tets of cell c are rows 6c..6c+5
    D = np.stack([X[T[:, a]]-X[T[:, 0]] for a in (1, 2, 3)], -1)
    det = np.linalg.det(D); neg = det < 0
    T[neg, 1], T[neg, 2] = T[neg, 2].copy(), T[neg, 1].copy()
    D = np.stack([X[T[:, a]]-X[T[:, 0]] for a in (1, 2, 3)], -1)
    vol = np.linalg.det(D)/6; assert (vol > 0).all()
    Dinv = np.linalg.inv(D)  # rows: d lambda_{1..3} / dX
    Gr = np.zeros((len(T), 4, 3)); Gr[:, 1:] = Dinv; Gr[:, 0] = -Dinv.sum(1)
    m = RealMesh()
    m.obj, m.take, m.poke, m.h, m.y_s = obj, take, poke, float(h), ys
    m.X, m.T, m.vol, m.G, m.n = X, T, vol, Gr, len(X)
    m.scan_V, m.scan_F = V, F
    m.volume = float(vol.sum()); m.mass = OBJECTS[obj]['mass_g']*1e-3; m.rho = m.mass/m.volume
    # support: bottom node layer (y = y_s) fixed in all directions
    m.fixed = np.abs(X[:, 1]-ys) < 0.5*h
    fmap = -np.ones(3*m.n, np.int64); freed = np.where(~np.repeat(m.fixed, 3))[0]; fmap[freed] = np.arange(len(freed))
    m.free = freed; m.nfree = len(freed)
    # reduced sparse pattern over free dofs
    dof = (3*T[:, :, None]+np.arange(3)).reshape(-1, 12)
    rows = fmap[np.repeat(dof, 12, 1)]; cols = fmap[np.tile(dof, (1, 12))]
    keep = (rows >= 0) & (cols >= 0)
    key = rows.astype(np.int64)*m.nfree+cols
    uk = np.unique(key[keep])
    m.slot = np.where(keep, np.searchsorted(uk, key), -1)  # (E,144)
    m.pat_rows, m.pat_cols = (uk//m.nfree).astype(np.int64), (uk % m.nfree).astype(np.int64)
    m.nnz = len(uk)
    m.upper = m.pat_rows <= m.pat_cols  # symmetric storage (upper triangle, row-major sorted)
    m.up_indptr = np.r_[0, np.cumsum(np.bincount(m.pat_rows[m.upper], minlength=m.nfree))].astype(np.int32)
    m.up_indices = m.pat_cols[m.upper].astype(np.int32)
    m.dof = dof
    # gravity (lumped: linear tet consistent load = rho g V/4 per node)
    fg = np.zeros((m.n, 3)); np.add.at(fg[:, 1], T.ravel(), np.repeat(-m.rho*G_ACC*vol/4, 4)); m.f_grav = fg.ravel()
    # embedding of scan vertices
    cen = X[T].mean(1)
    tree = cKDTree(cen); kq = min(48, len(T))
    _, cand = tree.query(V, k=kq)
    lam = np.einsum('vkij,vkj->vki', Dinv[cand], V[:, None, :]-X[T[cand, 0]])
    lam = np.concatenate([1-lam.sum(-1, keepdims=True), lam], -1)  # (Nv,k,4)
    best = np.argmax(lam.min(-1), 1)
    m.emb_tet = cand[np.arange(len(V)), best]; m.emb_w = lam[np.arange(len(V)), best]
    minbary = m.emb_w.min(1)
    m.embed_inside_frac = float((minbary >= -1e-9).mean())
    xs = (X[T[m.emb_tet]]*m.emb_w[:, :, None]).sum(1)
    assert np.abs(xs-V).max() < 1e-9
    # distance of outside vertices from their tet (extrapolation length)
    m.embed_max_extrap_mm = float(np.max(-np.minimum(minbary, 0))*h*1e3*np.sqrt(3))
    # contact points: scan vertices (area-weighted) + grid nodes inside the scan (not fixed)
    Ar = 0.5*np.linalg.norm(np.cross(V[F[:, 1]]-V[F[:, 0]], V[F[:, 2]]-V[F[:, 0]]), axis=1)
    Av = np.bincount(F.ravel(), weights=np.repeat(Ar/3, 3), minlength=len(V))
    qcol = np.round((X[:, 0]-x0)/h).astype(int)*(nz+1)+np.round((X[:, 2]-z0)/h).astype(int)
    nin = _inside_vote(V, F, x0, z0, h, nx+1, nz+1, qcol, X[:, 1]) & ~m.fixed
    node_tet = np.full(m.n, -1); node_loc = np.zeros(m.n, int)
    node_tet[T.ravel()] = np.repeat(np.arange(len(T)), 4); node_loc[T.ravel()] = np.tile(np.arange(4), len(T))
    ni = np.where(nin)[0]
    wn = np.zeros((len(ni), 4)); wn[np.arange(len(ni)), node_loc[ni]] = 1.0
    m.c_tet = np.r_[m.emb_tet, node_tet[ni]]; m.c_w = np.r_[m.emb_w, wn]; m.c_area = np.r_[Av, np.full(len(ni), h*h)]
    m.n_contact_scan, m.n_contact_nodes = len(V), len(ni)
    m.stats = dict(h_mm=m.h*1e3, grid=[nx, ny, nz], cells=len(cells), cells_inside_raw=n_inside_raw,
                   components_removed=int(nl-1) if nl > 1 else 0, cells_removed=int(n_inside_raw-len(cells)),
                   nodes=int(m.n), tets=int(len(T)), free_dofs=int(m.nfree), fixed_nodes=int(m.fixed.sum()),
                   volume_L=m.volume*1e3, mass_kg=m.mass, rho=m.rho, y_s=ys,
                   scan_vertices=len(V), embed_inside_frac=m.embed_inside_frac, embed_max_extrap_mm=m.embed_max_extrap_mm,
                   contact_nodes=int(len(ni)), build_s=time.time()-t0)
    return m


# ---------------------------------------------------------------- mechanics
class _Sym:
    """Pardiso, real symmetric indefinite (mtype -2), symbolic analysis done once for the fixed pattern."""
    def __init__(self, m):
        self.m = m; self.s = pypardiso.PyPardisoSolver(mtype=-2); self.analysed = False

    def factor(self, data_up):
        m = self.m
        self.A = sp.csr_matrix((data_up, m.up_indices, m.up_indptr), shape=(m.nfree, m.nfree))
        self.s._check_A(self.A)
        z = np.zeros((m.nfree, 1))
        if not self.analysed:
            self.s.set_phase(11); self.s._call_pardiso(self.A, z); self.analysed = True
        self.s.set_phase(22); self.s._call_pardiso(self.A, z)

    def solve(self, b):
        self.s.set_phase(33)
        return self.s._call_pardiso(self.A, np.asarray(b, float).reshape(-1, 1)).ravel()

    def free(self):
        try:
            self.s.free_memory(everything=True)
        except Exception:
            pass

    __del__ = free


def _energy_density(Fm, mu, lam):
    J = np.linalg.det(Fm); lnJ = np.log(J)
    return 0.5*mu*((Fm*Fm).sum((-1, -2))-3)-mu*lnJ+0.5*lam*lnJ**2


class _Model:
    def __init__(self, m, E, nu, c_pen):
        self.m = m; self.E = E; self.mu, self.lam = lame(E, nu)
        self.kap = c_pen*E*m.c_area/m.h
        self.fg = m.f_grav
        self.lin = _Sym(m); self.n_shift = 0; self.trace = None; self.ptype = None
        up = np.where(m.upper)[0]
        self.diag = np.searchsorted(up, np.where(m.pat_rows == m.pat_cols)[0])  # diagonal positions in data_up

    def deform(self, u):
        x = self.m.X+u.reshape(-1, 3)
        return np.einsum('eai,eaJ->eiJ', x[self.m.T], self.m.G)

    def elastic(self, u, tangent):
        m = self.m; Fm = self.deform(u); J = np.linalg.det(Fm)
        if not (J > 0).all():
            return None
        e = float((m.vol*_energy_density(Fm, self.mu, self.lam)).sum())
        if not tangent:
            Fi = np.linalg.inv(Fm); Pk = self.mu*(Fm-np.swapaxes(Fi, -1, -2))+self.lam*np.log(J)[:, None, None]*np.swapaxes(Fi, -1, -2)
            A = None
        else:
            Pk, A = neo_hookean(Fm, self.mu, self.lam)
        fe = np.einsum('e,eiJ,eaJ->eai', m.vol, Pk, m.G).reshape(-1, 12)
        f = np.bincount(m.dof.ravel(), weights=fe.ravel(), minlength=3*m.n)
        Ke = None if A is None else np.einsum('e,eaJ,eiJkL,ebL->eaibk', m.vol, m.G, A, m.G, optimize=True).reshape(-1, 144)
        return e, f, Ke, float(J.min())

    def contact(self, u, tool, tangent):
        m = self.m; o, a = tool
        x = m.X+u.reshape(-1, 3)
        p = np.einsum('pai,pa->pi', x[m.T[m.c_tet]], m.c_w)
        r = p-o; s = r@a; rv = r-s[:, None]*a; rn = np.linalg.norm(rv, axis=1)
        act = np.where((s < 0) & (s > -L_TOOL) & (rn < R_TOOL))[0]
        f = np.zeros(3*m.n)
        if len(act) == 0:
            return 0.0, f, None, None, np.zeros(3), 0
        s, rv, rn, kap = s[act], rv[act], rn[act], self.kap[act]
        dface, drad = -s, R_TOOL-rn
        if self.ptype is not None:  # history bias: the direction the point did not come from costs K_SWITCH x more
            pt = self.ptype[act]
            dface = np.where(pt, dface, K_SWITCH*dface); drad = np.where(pt, K_SWITCH*drad, drad)
        face = dface <= drad
        nrm = np.where(face[:, None], a[None], rv/np.maximum(rn, 1e-15)[:, None])
        d = np.where(face, dface, drad)
        # d is continuous (min of two distances that vanish on their own boundary).  Penalty energy phi(d), C2 at
        # contact onset: kap d^3/(6 dl) for d < dl, kap (d^2/2 - dl d/2 + dl^2/6) beyond (dl = PEN_SMOOTH): removes the
        # stiffness jump that made Newton chatter on points touching the tool boundary.
        dl = PEN_SMOOTH
        lo = d < dl
        e = float((kap*np.where(lo, d**3/(6*dl), d*d/2-dl*d/2+dl*dl/6)).sum())
        g1 = kap*np.where(lo, d*d/(2*dl), d-dl/2)  # phi'(d)
        g2 = kap*np.where(lo, d/dl, 1.0)          # phi''(d)
        if self.ptype is not None:
            sc = np.where(face, np.where(pt, 1.0, K_SWITCH), np.where(pt, K_SWITCH, 1.0))
        else:
            sc = np.ones(len(act))
        fp = (g1*sc)[:, None]*nrm  # force on the object at each active point (= -dE/dp)
        tet, w = m.c_tet[act], m.c_w[act]
        nodes = m.T[tet]
        for c in range(3):
            f[c::3] += np.bincount(nodes.ravel(), weights=(w*fp[:, c:c+1]).ravel(), minlength=m.n)
        Kc = None
        if tangent:
            # exact Hessian: phi'' (grad d)(grad d)^T + phi' * Hess(d); Hess(d) = -(sc/r) t t^T for radial pushes
            tv = np.cross(a[None], nrm)
            nrm2 = nrm*sc[:, None]
            H = (g2[:, None, None]*np.einsum('pi,pk->pik', nrm2, nrm2)
                 - np.where(face, 0.0, g1*sc/np.maximum(rn, 1e-15))[:, None, None]*np.einsum('pi,pk->pik', tv, tv))
            Kc = (np.einsum('pa,pb->pab', w, w)[:, :, None, :, None]*H[:, None, :, None, :]).reshape(-1, 144)
        return e, f, Kc, tet, -fp.sum(0), len(act)

    def classify(self, u, tool):
        """Freeze each contact point's push-out direction for the next solve from the previous equilibrium u at the
        previous tool pose: penetrating points -> the cheaper direction there (penetrations are small at
        equilibrium); free points -> 'face' if they are at/below the end-face plane, else 'radial' (beside the body).
        Removes the deep-penetration ambiguity of the cheaper-direction rule inside a finite load step."""
        m = self.m; o, a = tool
        p = np.einsum('pai,pa->pi', (m.X+u.reshape(-1, 3))[m.T[m.c_tet]], m.c_w)
        r = p-o; s = r@a; rn = np.linalg.norm(r-s[:, None]*a, axis=1)
        inside = (s < 0) & (s > -L_TOOL) & (rn < R_TOOL)
        self.ptype = np.where(inside, -s <= R_TOOL-rn, s >= 0)

    def potential(self, u, tool):
        el = self.elastic(u, False)
        if el is None:
            return None
        ec, fc, _, _, ftool, nact = self.contact(u, tool, False)
        g = el[1]-self.fg-fc
        return el[0]+ec-self.fg@u, g[self.m.free], el[3], ftool, nact

    def assemble(self, Ke, Kc, ctet):
        m = self.m
        sl = m.slot.ravel(); ok = sl >= 0
        data = np.bincount(sl[ok], weights=Ke.ravel()[ok], minlength=m.nnz)
        if Kc is not None:
            slc = m.slot[ctet].ravel(); okc = slc >= 0
            data += np.bincount(slc[okc], weights=Kc.ravel()[okc], minlength=m.nnz)
        return data[m.upper]

    def newton(self, u, tool, tol, maxit=30, pred=None):
        """Returns (u, converged, iterations, info).  pred: optional predictor increment, used if it lowers the
        potential."""
        m = self.m; u = u.copy()
        cur = self.potential(u, tool)
        if cur is None:
            return u, False, 0, dict(reason='inverted start')
        if pred is not None:
            tr = self.potential(u+pred, tool)
            if tr is not None and tr[0] < cur[0]:
                u, cur = u+pred, tr
        for it in range(maxit):
            Pi, g, jmin, ftool, nact = cur
            gn = float(np.linalg.norm(g))
            if gn <= tol:
                return u, True, it, dict(res=gn, jmin=jmin, ftool=ftool, nact=nact)
            el = self.elastic(u, True)
            _, _, Kc, ctet, _, _ = self.contact(u, tool, True)
            Kd = self.assemble(el[2], Kc, ctet)
            self.lin.factor(Kd); du = -self.lin.solve(g)
            slope = float(g@du)
            if not np.isfinite(slope) or slope >= 0:  # not a descent direction: shift the tangent
                dg = np.abs(Kd[self.diag])
                Ks = Kd.copy(); Ks[self.diag] += 1e-2*dg+1e-12
                self.lin.factor(Ks); du = -self.lin.solve(g); slope = float(g@du); self.n_shift += 1
                if not np.isfinite(slope) or slope >= 0:
                    du = -g/np.maximum(dg, 1e-12); slope = float(g@du)
            alpha, acc = 1.0, None
            # cap the step so no free dof moves more than h (prevents wild first steps)
            mx = np.abs(du).max()
            if mx > m.h:
                alpha = m.h/mx
            while alpha > 1e-8:
                un = u.copy(); un[m.free] += alpha*du
                tr = self.potential(un, tool)
                if tr is not None and (tr[0] <= Pi+1e-4*alpha*slope or np.linalg.norm(tr[1]) < (1-1e-4*alpha)*gn):
                    acc = (un, tr); break
                alpha *= 0.5
            if self.trace is not None:
                Kfull = sp.csr_matrix((Kd, self.m.up_indices, self.m.up_indptr), shape=(self.m.nfree,)*2)
                Kfull = Kfull+sp.triu(Kfull, 1).T
                lres = float(np.linalg.norm(Kfull@du+g)/gn)
                self.trace.append((it, gn, alpha if acc else 0.0, nact, slope, lres))
            if acc is None:
                return u, False, it+1, dict(res=gn, jmin=jmin, ftool=ftool, nact=nact, reason='line search failed')
            u, cur = acc
        Pi, g, jmin, ftool, nact = cur
        gn = float(np.linalg.norm(g))
        return u, gn <= tol, maxit, dict(res=gn, jmin=jmin, ftool=ftool, nact=nact, reason='maxit')


class SimTimeout(RuntimeError):
    pass


def snap_frames(fa, tip, axis, drop=0.8, min_N=5.0, tol_m=5e-4):
    """Contact 'snap' (scan surface slipping around the tool rim): frame j is flagged when its push force along the
    approach axis falls below drop x the largest force at any earlier frame whose tip depth was not greater (minus
    tol_m) - for an elastic quasi-static model the force cannot drop while the tool goes deeper.  Uses the simulation
    only (no measurement), so it may be applied to held-out pokes.  Only loading frames (tip at or beyond its deepest
    point so far, within tol_m) are tested: on unloading the history-dependent push-out rule (K_SWITCH) gives a mild
    loading/unloading difference that is model behaviour, not a contact failure."""
    snap = np.zeros(len(fa), bool)
    if len(fa) == 0:
        return snap
    s = (tip-tip[0])@axis[0]
    for j in range(1, len(fa)):
        if s[j] < s[:j].max()-tol_m:
            continue
        prev = fa[:j][s[:j] <= s[j]+tol_m]
        if len(prev) and prev.max() > min_N and fa[j] < drop*prev.max():
            snap[j] = True
    return snap


def _pose(Tm):
    a = Tm[:3, 2]/np.linalg.norm(Tm[:3, 2])
    return Tm[:3, 3].copy(), a


def _lerp(t0, t1, s):
    o = (1-s)*t0[0]+s*t1[0]; a = (1-s)*t0[1]+s*t1[1]
    return o, a/np.linalg.norm(a)


def simulate(mesh, obj, take, poke, nu, E=None, frames=None, gravity=True, c_pen=100.0, dmax=3e-3, maxdepth=6, verbose=False,
             budget_s=None):
    """Quasi-static poke from the segment's start frame to its end frame.  Returns dict(frames, force (F,3) N on the
    tool in world, surf_V (F,Nv,3) m, surf_F, converged, newton_iters, seconds, ...).
    converged[i]: the equilibrium returned for frame i meets the residual tolerance; path_ok[i]: additionally every
    intermediate sub-step between frame i-1 and i converged (a failed intermediate state is kept and continued from).
    budget_s: hard wall-time limit; past it SimTimeout is raised (the caller records the run as failed).  An earlier
    version degraded the solver instead (shallower bisection, fewer Newton iterations); on the full run that kept
    stale states while the tool advanced and produced spurious forces (FoamDice_T2 p1: 566 N), so it was removed.
    'degraded' is kept in the output (always 0) for file compatibility."""
    t_start = time.time()
    E = E_REF if E is None else float(E)
    m = mesh; mod = _Model(m, E, nu, c_pen)
    if not gravity:
        mod.fg = np.zeros_like(mod.fg)
    tr = P.robot_trajectory(obj, take); q = P.segments(obj, take)['pokes'][poke-1]
    idx = np.arange(q['start_idx'], q['end_idx']+1); allfr = tr['frame'][idx]
    want = allfr if frames is None else np.array(sorted(set(int(f) for f in frames)))
    assert set(want) <= set(allfr.tolist()), 'frames outside the poke segment'
    last = int(np.searchsorted(allfr, want.max()))
    tol = TOL_REL*(E*m.h**2+m.mass*G_ACC)
    # gravity pre-load (tool ignored: placed far away)
    far = (np.array([0, 1e3, 0.0]), np.array([0, -1.0, 0]))
    u = np.zeros(3*m.n); n_it = 0; ok_g = True
    if gravity:
        u, ok_g, n_it, info = mod.newton(u, far, tol)
    u_g = u.copy()
    # start pose: if the tool already penetrates, ramp it in from a retracted pose
    pose0 = _pose(tr['T_WT'][idx[0]])
    _, _, _, _, _, nact0 = mod.contact(u, pose0, False)
    out = dict(force=[], surf=[], conv=[], path_ok=[], iters=[], substeps=[], jmin=[], nact=[], res=[], deg=[], axis=[], tip=[])
    log = []

    hist = dict(du=None, step=0.0)  # last converged sub-step increment (predictor)
    lim = dict(depth=maxdepth, maxit=30, level=0)

    def update_limits():
        if budget_s is None:
            return
        if time.time()-t_start > budget_s:
            raise SimTimeout(f'wall time > {budget_s:.0f} s')

    def advance(u, p0, p1, depth):
        """Solve at p1 from equilibrium u at p0; sub-steps by tip travel while in contact; bisection on failure."""
        _, _, _, _, _, na1 = mod.contact(u, p1, False)
        _, _, _, _, _, na0 = mod.contact(u, p0, False)
        if na1 == 0 and na0 == 0:
            hist['du'] = None
            r = mod.potential(u, p1)
            return u, True, 0, 1, dict(res=float(np.linalg.norm(r[1])), jmin=r[2], ftool=r[3], nact=r[4])
        nsub = max(1, int(np.ceil(np.linalg.norm(p1[0]-p0[0])/dmax))) if depth == 0 else 1
        its = 0; subs = 0; info = {}; allok = True
        for s in range(1, nsub+1):
            pa, pb = _lerp(p0, p1, (s-1)/nsub), _lerp(p0, p1, s/nsub)
            mod.classify(u, pa)
            step = np.linalg.norm(pb[0]-pa[0])
            pred = None
            if hist['du'] is not None and hist['step'] > 0 and depth == 0:
                pred = hist['du']*min(step/hist['step'], 1.0)
            update_limits()
            un, ok, it, info = mod.newton(u, pb, tol, maxit=lim['maxit'], pred=pred)
            if ok and depth == 0:
                hist['du'], hist['step'] = un-u, step
            its += it; subs += 1
            if not ok and depth < lim['depth']:
                pm = _lerp(pa, pb, 0.5)
                u1, ok1, it1, s1, _ = advance(u, pa, pm, depth+1)
                u2, ok2, it2, s2, info = advance(u1, pm, pb, depth+1)
                un, ok, its, subs = u2, ok1 and ok2, its+it1+it2, subs+s1+s2-1
            elif not ok:
                # deepest bisection level (tool increment dmax/2^maxdepth) still fails: discard the unconverged
                # iterate and keep the last equilibrium (the state lags the tool by that tiny increment).  Continuing
                # from unconverged iterates let the scan surface slip past the tool rim (force 'snap', dev take).
                un = u; r = mod.potential(u, pb)
                info = dict(res=float(np.linalg.norm(r[1])), jmin=r[2], ftool=r[3], nact=r[4])
            allok &= ok
            u = un
        return u, allok, its, subs, info

    if nact0 > 0:  # retract along the axis until free, then come back to the start pose
        back = (pose0[0]-0.1*pose0[1], pose0[1])
        u, ok, it, sub, info = advance(u, back, pose0, 0); log.append(f'start pose penetrates ({nact0} points): ramped in, ok={ok}')
    else:
        r = mod.potential(u, pose0); info = dict(res=float(np.linalg.norm(r[1])), jmin=r[2], ftool=r[3], nact=r[4]); ok = ok_g; it = n_it; sub = 1
    prev = pose0
    for j in range(last+1):
        pj = _pose(tr['T_WT'][idx[j]])
        if j > 0:
            u, ok, it, sub, info = advance(u, prev, pj, 0)
        prev = pj
        if allfr[j] in want:
            out['force'].append(info['ftool']); out['conv'].append(bool(info['res'] <= tol)); out['path_ok'].append(bool(ok)); out['iters'].append(int(it)); out['substeps'].append(int(sub))
            out['deg'].append(lim['level']); out['axis'].append(pj[1]); out['tip'].append(pj[0])
            out['jmin'].append(float(info['jmin'])); out['nact'].append(int(info['nact'])); out['res'].append(float(info['res']))
            du = (u-u_g).reshape(-1, 3)
            out['surf'].append(m.scan_V+np.einsum('vai,va->vi', du[m.T[m.emb_tet]], m.emb_w))
        if verbose:
            print(f'frame {allfr[j]} ok={ok} it={it} sub={sub} nact={info["nact"]} F={np.round(info["ftool"], 3)} jmin={info["jmin"]:.3f} t={time.time()-t_start:.1f}s', flush=True)
    sag = np.linalg.norm(u_g.reshape(-1, 3), axis=1)
    mod.lin.free()
    F_ = np.array(out['force']); fa = np.einsum('ij,ij->i', F_, -np.array(out['axis'])) if len(F_) else np.zeros(0)
    snap = snap_frames(fa, np.array(out['tip']), np.array(out['axis']))
    return dict(frames=np.asarray(want), force=np.array(out['force']), surf_V=np.array(out['surf']), surf_F=m.scan_F,
                converged=np.array(out['conv']), path_ok=np.array(out['path_ok']), tol=tol, newton_iters=np.array(out['iters']), substeps=np.array(out['substeps']),
                jmin=np.array(out['jmin']), n_active=np.array(out['nact']), residual=np.array(out['res']),
                gravity_converged=bool(ok_g), gravity_sag_max_mm=float(sag.max()*1e3), E=E, nu=float(nu), c_pen=c_pen,
                degraded=np.array(out['deg']), snap=snap, seconds=time.time()-t_start, log=log)


# ---------------------------------------------------------------- development checks (FoamDice/FoamDice_T1 only)
NUS = [0.0, 0.1, 0.2, 0.3, 0.4]


def _sim_file(poke, nu, E, gravity=True):
    return OUTDIR/'sims'/f'p{poke}_nu{nu:.1f}_E{E:.4g}{"" if gravity else "_nograv"}.npz'


def _cached_sim(poke, nu, E=E_REF, gravity=True, meshes={}):
    f = _sim_file(poke, nu, E, gravity)
    if not f.exists():
        assert P.is_dev(DEV_OBJ, DEV_TAKE)
        if poke not in meshes:
            meshes[poke] = build(DEV_OBJ, DEV_TAKE, poke)
        r = simulate(meshes[poke], DEV_OBJ, DEV_TAKE, poke, nu, E=E, gravity=gravity)
        f.parent.mkdir(parents=True, exist_ok=True)
        tmp = f.with_name(f'{f.stem}.tmp{np.random.default_rng().integers(1 << 30)}.npz')  # atomic: units may race
        np.savez_compressed(tmp, **{k: (v.astype(np.float32) if k == 'surf_V' else v) for k, v in r.items() if k != 'log'},
                            log=np.array(r['log'], dtype=str))
        tmp.replace(f)
        print(f'{f.name}: {r["seconds"]:.0f}s converged {int(r["converged"].sum())}/{len(r["converged"])} path_ok {int(r["path_ok"].sum())} '
              f'newton {int(r["newton_iters"].sum())} substeps {int(r["substeps"].sum())} jmin {r["jmin"].min():.3f}', flush=True)
    return dict(np.load(f, allow_pickle=False))


def _dev_mesh():
    rows = {}
    for q in P.segments(DEV_OBJ, DEV_TAKE)['pokes']:
        m = build(DEV_OBJ, DEV_TAKE, q['poke'])
        rows[q['poke']] = m.stats
    V0, F0, _ = P.rest_mesh(DEV_OBJ, DEV_TAKE, 1)
    side = 0.155  # catalogue 'L 15.5' cm cube
    out = dict(pokes=rows, catalogue_cube_volume_L=side**3*1e3, catalogue_mass_kg=OBJECTS[DEV_OBJ]['mass_g']*1e-3)
    (OUTDIR/'meshing.json').write_text(json.dumps(out, indent=1))
    return out


def _dev_escale():
    """E-scaling: same poke/nu at E_REF and 10 E_REF (gravity on); E_REF/10 as a stress test; E_REF without gravity."""
    res = {}
    poke, nu = 1, 0.3
    runs = {'10E_REF': _cached_sim(poke, nu, 10*E_REF), 'E_REF/10': _cached_sim(poke, nu, E_REF/10),
            'E_REF_nograv': _cached_sim(poke, nu, E_REF, gravity=False)}
    runs['E_REF'] = _cached_sim(poke, nu, E_REF)
    base = runs['E_REF']; V0 = P.rest_mesh(DEV_OBJ, DEV_TAKE, poke)[0]
    d0 = base['surf_V']-V0[None]; f0 = base['force']/base['E']
    for k, r in runs.items():
        if k == 'E_REF':
            continue
        d = r['surf_V']-V0[None]; f = r['force']/r['E']
        res[k] = dict(E=float(r['E']), max_rel_dev_displacement=float(np.abs(d-d0).max()/np.abs(d0).max()),
                      max_abs_dev_displacement_mm=float(np.abs(d-d0).max()*1e3),
                      rms_rel_dev_displacement=float(np.sqrt(((d-d0)**2).sum()/(d0**2).sum())),
                      p99_vertex_dev_mm=float(np.percentile(np.linalg.norm(d-d0, axis=2), 99)*1e3),
                      frac_vertex_frames_dev_gt_1mm=float((np.linalg.norm(d-d0, axis=2) > 1e-3).mean()),
                      max_dev_force_over_E_times_Eref_N=float(np.linalg.norm(f-f0, axis=1).max()*E_REF),
                      max_rel_dev_force_over_E=float(np.linalg.norm(f-f0, axis=1).max()/np.linalg.norm(f0, axis=1).max()),
                      converged=int(r['converged'].sum()), frames=len(r['converged']),
                      gravity_sag_max_mm=float(r['gravity_sag_max_mm']), seconds=float(r['seconds']))
    res['E_REF'] = dict(E=E_REF, gravity_sag_max_mm=float(base['gravity_sag_max_mm']), seconds=float(base['seconds']))
    (OUTDIR/'escale.json').write_text(json.dumps(res, indent=1))
    return res


def _shape_stats(fa, frames, thr=2.0):
    on = np.where(fa > thr)[0]
    return dict(onset_frame=int(frames[on[0]]) if len(on) else None, off_frame=int(frames[on[-1]]) if len(on) else None,
                peak_frame=int(frames[np.argmax(fa)]), peak_N=float(fa.max()))


def _dev_report():
    tr = P.robot_trajectory(DEV_OBJ, DEV_TAKE); ax = tr['T_WT'][:, :3, 2]
    W = P.forces(DEV_OBJ, DEV_TAKE); off = P.force_offset(DEV_OBJ, DEV_TAKE)
    pokes = P.segments(DEV_OBJ, DEV_TAKE)['pokes']
    sims = {(q['poke'], nu): _cached_sim(q['poke'], nu) for nu in NUS for q in pokes}
    meas, appr = {}, {}
    for q in pokes:
        fr = sims[(q['poke'], NUS[0])]['frames']
        meas[q['poke']] = W[fr-1, :3]-off[:3]; appr[q['poke']] = -ax[fr-1]/np.linalg.norm(ax[fr-1], axis=1, keepdims=True)
    sc = lambda fr, q: fr > q['start_frame']  # scored frames: start+1 .. end
    out = dict(E_REF=E_REF, force_offset_N=off[:3].tolist(), per_nu={}, null={})
    hmm = json.loads((OUTDIR/'meshing.json').read_text())['pokes']
    tol_of = lambda pk, s: float(s['tol']) if 'tol' in s else TOL_REL*(float(s['E'])*(hmm[str(pk)]['h_mm']*1e-3)**2+OBJECTS[DEV_OBJ]['mass_g']*1e-3*G_ACC)
    # null predictor
    allm = np.concatenate([meas[q['poke']][sc(sims[(q['poke'], 0.0)]['frames'], q)] for q in pokes])
    alla = np.concatenate([(meas[q['poke']]*appr[q['poke']]).sum(1)[sc(sims[(q['poke'], 0.0)]['frames'], q)] for q in pokes])
    out['null'] = dict(force_rmse_N=float(np.sqrt((allm**2).sum(1).mean())), force_along_rmse_N=float(np.sqrt((alla**2).mean())))
    for nu in NUS:
        fs = np.concatenate([sims[(q['poke'], nu)]['force'][sc(sims[(q['poke'], nu)]['frames'], q)] for q in pokes])
        k = float((fs*allm).sum()/(fs*fs).sum())  # force ~ E: closed-form least-squares scale
        res = allm-k*fs
        perpoke = {}
        for q in pokes:
            s = sims[(q['poke'], nu)]; m_ = sc(s['frames'], q)
            fq, mq = s['force'][m_], meas[q['poke']][m_]
            kq = float((fq*mq).sum()/(fq*fq).sum())
            aq = appr[q['poke']][m_]
            perpoke[q['poke']] = dict(E_fit_this_poke=kq*E_REF, rmse_N_at_pooled_E=float(np.sqrt(((mq-k*fq)**2).sum(1).mean())),
                                      along_rmse_N_at_pooled_E=float(np.sqrt((((mq-k*fq)*aq).sum(1)**2).mean())),
                                      null_rmse_N=float(np.sqrt((mq**2).sum(1).mean())),
                                      measured=_shape_stats((mq*aq).sum(1), s['frames'][m_]),
                                      simulated=_shape_stats((k*fq*aq).sum(1), s['frames'][m_]),
                                      lateral_rmse_N=float(np.sqrt((((mq-k*fq)-((mq-k*fq)*aq).sum(1, keepdims=True)*aq)**2).sum(1).mean())),
                                      converged_final=f'{int((s["residual"] <= tol_of(q["poke"], s)).sum())}/{len(s["residual"])}',
                                      substep_path_ok=f'{int(s["path_ok" if "path_ok" in s else "converged"].sum())}/{len(s["residual"])}',
                                      max_residual_N=float(s['residual'].max()), seconds=float(s['seconds']),
                                      newton_iters=int(s['newton_iters'].sum()), substeps=int(s['substeps'].sum()), jmin=float(s['jmin'].min()))
        aa = np.concatenate([appr[q['poke']][sc(sims[(q['poke'], nu)]['frames'], q)] for q in pokes])
        out['per_nu'][nu] = dict(E_fit=k*E_REF, force_rmse_N=float(np.sqrt((res**2).sum(1).mean())),
                                 force_along_rmse_N=float(np.sqrt(((res*aa).sum(1)**2).mean())), pokes=perpoke)
        print(f'nu={nu}: E*={k*E_REF:.0f} Pa  rmse={out["per_nu"][nu]["force_rmse_N"]:.2f} N  along={out["per_nu"][nu]["force_along_rmse_N"]:.2f} N', flush=True)
    # surface error at a few frames (mid-loading, deepest, mid-unloading, end)
    surf = {}
    for q in pokes:
        fr_list = sorted({(q['start_frame']+q['deepest_frame'])//2, q['deepest_frame'], (q['deepest_frame']+q['end_frame'])//2, q['end_frame']})
        V0, F0, _ = P.rest_mesh(DEV_OBJ, DEV_TAKE, q['poke'])
        for fr in fr_list:
            Vt, Ft, ct = P.mesh(DEV_OBJ, DEV_TAKE, fr)
            Tm = tr['T_WT'][fr-1]; kw = dict(tool_tip=Tm[:3, 3], tool_axis=Tm[:3, 2])
            row = dict(null=P.surface_error_mm(V0, F0, Vt, Ft, ct, **kw))
            for nu in NUS:
                s = sims[(q['poke'], nu)]; i = int(np.where(s['frames'] == fr)[0][0])
                row[nu] = P.surface_error_mm(s['surf_V'][i].astype(float), F0, Vt, Ft, ct, **kw)
            surf[f'p{q["poke"]}_f{fr}'] = row
            print(q['poke'], fr, 'null', round(row['null']['chamfer_excl_tool'], 2), ' '.join(f'{nu}:{row[nu]["chamfer_excl_tool"]:.2f}' for nu in NUS), flush=True)
    out['surface'] = surf
    out['surface_mean_excl_tool'] = {str(k): float(np.mean([v[k]['chamfer_excl_tool'] for v in surf.values()])) for k in ['null']+NUS}
    out['surface_mean_chamfer'] = {str(k): float(np.mean([v[k]['chamfer'] for v in surf.values()])) for k in ['null']+NUS}
    (OUTDIR/'fit.json').write_text(json.dumps(out, indent=1, default=float))
    _dev_plot(pokes, sims, meas, appr, out)
    return out


def _dev_plot(pokes, sims, meas, appr, out):
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    best = min(NUS, key=lambda nu: out['per_nu'][nu]['force_rmse_N'])
    fig, axs = plt.subplots(2, len(pokes), figsize=(4.2*len(pokes), 6.4), sharey='row', constrained_layout=True)
    for j, q in enumerate(pokes):
        a0, a1 = axs[0, j], axs[1, j]
        fr = sims[(q['poke'], best)]['frames']; m = meas[q['poke']]; ap = appr[q['poke']]
        for nu in NUS:
            if nu == best:
                continue
            s = sims[(q['poke'], nu)]; k = out['per_nu'][nu]['E_fit']/E_REF
            a0.plot(fr, (k*s['force']*ap).sum(1), color='#b8b8b8', lw=1)
        s = sims[(q['poke'], best)]; k = out['per_nu'][best]['E_fit']/E_REF
        a0.plot(fr, (m*ap).sum(1), color='#222222', lw=2, label='measured (offset-corrected)')
        a0.plot(fr, (k*s['force']*ap).sum(1), color='#2a6fdb', lw=2, label=f'FEM nu={best}, E={k*E_REF/1e3:.1f} kPa')
        a0.axvline(q['deepest_frame'], color='#dddddd', lw=1, zorder=0)
        a0.set_title(f'poke {q["poke"]}'); a0.set_xlabel('frame (30 fps)')
        for c, (lab, col) in enumerate([('x', '#2a6fdb'), ('z', '#d9822b')]):
            ci = 0 if lab == 'x' else 2
            a1.plot(fr, m[:, ci], color=col, lw=2, label=f'measured F{lab}')
            a1.plot(fr, k*s['force'][:, ci], color=col, lw=1.2, ls='--', label=f'FEM F{lab}')
        a1.set_xlabel('frame (30 fps)')
        for a in (a0, a1):
            a.grid(color='#eeeeee', lw=0.8); a.spines[['top', 'right']].set_visible(False)
    axs[0, 0].set_ylabel('force along -approach axis (N)'); axs[1, 0].set_ylabel('world lateral force (N)')
    axs[0, 0].legend(frameon=False, fontsize=8); axs[1, 0].legend(frameon=False, fontsize=8, ncol=2)
    fig.suptitle(f'FoamDice_T1 (development take): measured vs simulated tool force; gray = other nu at their best-fit E', fontsize=10)
    fig.savefig(OUTDIR/'force_dev.png', dpi=130); plt.close(fig)


def meshcheck():
    """Build every poke of every protocol object (free inputs only: rest meshes) -> meshing statistics."""
    rows = []
    for o in P.PROT['objects_considered']:
        for t in P.takes(o):
            for q in P.segments(o, t)['pokes']:
                try:
                    m = build(o, t, q['poke']); st = dict(m.stats)
                    V, F = m.scan_V, m.scan_F
                    E_ = np.sort(np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]]), 1)
                    _, cnt = np.unique(E_, axis=0, return_counts=True)
                    st.update(open_edges_unwelded=int((cnt == 1).sum()))
                except Exception as e:  # report, do not hide
                    st = dict(error=repr(e))
                st.update(obj=o, take=t, poke=q['poke'], frames=q['end_idx']-q['start_idx']+1); rows.append(st)
                print(json.dumps({k: st.get(k) for k in ['obj', 'take', 'poke', 'h_mm', 'cells', 'volume_L', 'rho', 'cells_removed', 'embed_inside_frac', 'error']}), flush=True)
    (P.OUT/'realfem_meshcheck.json').write_text(json.dumps(rows, indent=1))


def main(argv):
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=['dev', 'meshcheck'])
    ap.add_argument('--part', default='all', choices=['mesh', 'sims', 'escale', 'report', 'all'])
    ap.add_argument('--nu', type=float, nargs='*', default=None)
    ap.add_argument('--pokes', type=int, nargs='*', default=None)
    a = ap.parse_args(argv)
    if a.cmd == 'meshcheck':
        return meshcheck()
    OUTDIR.mkdir(parents=True, exist_ok=True)
    if a.part in ('mesh', 'all'):
        print(json.dumps(_dev_mesh(), indent=1))
    if a.part in ('sims', 'all'):
        for nu in (a.nu if a.nu is not None else NUS):
            for q in P.segments(DEV_OBJ, DEV_TAKE)['pokes']:
                if a.pokes is None or q['poke'] in a.pokes:
                    _cached_sim(q['poke'], nu)
    if a.part in ('escale', 'all'):
        print(json.dumps(_dev_escale(), indent=1))
    if a.part in ('report', 'all'):
        _dev_report()


if __name__ == '__main__':
    main(sys.argv[1:])
