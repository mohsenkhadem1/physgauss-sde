"""Per-material reduced operators for the Fig-1 block (step 2 of patch roles).

Inference mesh = Fig-1 ours mesh: block.mesh((12,5,6)), linear T10, bottom glued, rho known.
Material grid = Fig-1 ours inclusion parametrisation: inclusion centre xc in {-.08..+.08 step .02} with
contrast in {2,4,8,16}, or no inclusion; nu in {.3,.4,.45}. Soft modulus E is continuous and enters
exactly by scaling: K = E * Khat, omega^2 = E * lamhat.  For every grid material at E=1 we store

  lam   (m,)          lowest m generalised eigenvalues of (Khat, M), M-orthonormal modes Phi
  HPhi  (R,m)         modes at observation rows (20 markers x3, 27 probes x3)
  PhiS  (nS,3,m)      modes at loadable surface nodes (all non-fixed boundary nodes)
  XoS   (nS,3,R)      static adjoint  Khat^-1 H^T  at surface dofs (surface load -> rows)
  Css   (nS,3,nS,3)   static compliance Khat^-1 between surface dofs (float32; supports only)
  KO, KS, PhiK        static rows / static at surface / modal loads of KNOWN loads:
                      gravity (rho g) and 1 N downward presses on the protocol box at listed x.
Usage: bank.py build I0 I1   (material indices [I0,I1)) ; bank.py info
"""
import json, sys, time
from pathlib import Path
import numpy as np
import scipy.sparse.linalg as sla
R = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(R/'scripts/fig1_v01')); sys.path.insert(0, str(R/'scripts/stem_trial_v01'))
import fem, block as B

CELLS = (12, 5, 6)
XC = list(np.round(np.arange(-0.08, 0.0801, 0.02), 3))
CONTRAST = [2.0, 4.0, 8.0, 16.0]
NU = [0.3, 0.4, 0.45]
NMODES = 80
# known 1 N downward presses on the protocol box (top 1 cm x 4x4 cm) centred at these x (D0=0, D1=.05, D2=-.07)
PRESS_X = [0.0, 0.05, -0.07, -0.09, -0.06, -0.03, 0.03, 0.06, 0.09]
OUT = R/'runs/patch-roles-v02/bank'


def materials():
    """index -> (xc or None, contrast, nu). Index order: nu-major, then none, then (xc, contrast)."""
    out = []
    for nu in NU:
        out.append((None, 1.0, nu))
        for xc in XC:
            for k in CONTRAST:
                out.append((xc, k, nu))
    return out


def press_load(m, x, force=1.0):
    lo = [x-B.PATCH, B.TOP-0.01, -B.PATCH]; hi = [x+B.PATCH, B.TOP+1e-6, B.PATCH]
    return m.body_force_box(lo, hi, [0, -force, 0])


class Geometry:
    """Mesh, surface quadrature (linear sub-triangles of the T10 boundary, bottom excluded), rows."""
    def __init__(self):
        m = B.mesh(CELLS); self.m = m
        tr = m.surface_triangles(); X = m.X
        bot = np.all(np.abs(X[tr][:, :, 1]-B.LO[1]) < 1e-9, axis=1)
        tr = tr[~bot]
        self.S = np.setdiff1d(np.unique(tr), m.fixed)            # loadable surface nodes
        pos = -np.ones(m.n, int); pos[self.S] = np.arange(len(self.S))
        P = X[tr]; area = .5*np.linalg.norm(np.cross(P[:, 1]-P[:, 0], P[:, 2]-P[:, 0]), axis=1)
        bary = np.array([[2/3, 1/6, 1/6], [1/6, 2/3, 1/6], [1/6, 1/6, 2/3]])
        self.q = np.einsum('qa,taj->tqj', bary, P).reshape(-1, 3)
        self.qw = np.repeat(area/3, 3)
        rows, cols, vals = [], [], []
        for t in range(len(tr)):
            for qi in range(3):
                for a in range(3):
                    if pos[tr[t, a]] >= 0:
                        rows.append(3*t+qi); cols.append(pos[tr[t, a]]); vals.append(bary[qi, a])
        self.Nq = (np.array(rows), np.array(cols), np.array(vals), (len(self.q), len(self.S)))
        self.points = np.concatenate([B.MARKERS, B.PROBES])        # 20 markers + 27 probes
        self.H = m.interpolation(self.points).toarray()             # (47,n)
        self.nrow = 3*len(self.points)
        fdof = np.zeros(3*m.n, bool); fdof[m.free] = True
        self.free = m.free
        self.Sdof = (3*self.S[:, None]+np.arange(3)).ravel()        # global dofs of surface nodes
        self.g = m.gravity(B.RHO)
        self.known = np.stack([self.g]+[press_load(m, x) for x in PRESS_X], 1)   # (ndof, 1+len(PRESS_X))
        self.known_names = ['gravity']+[f'press x={x:+.2f} (1 N down)' for x in PRESS_X]


def build_one(G, idx, mat):
    xc, k, nu = mat; m = G.m; fr = G.free
    model = fem.Elastic(m, B.modulus(m, xc, contrast=k, e_soft=1.0), nu, B.RHO)
    K = model.linear_K().tocsr()[fr][:, fr]; Mm = model.M.tocsr()[fr][:, fr]
    fac = fem._factor(K)
    t0 = time.time()
    op = sla.LinearOperator(K.shape, matvec=lambda v: fac(np.asarray(v, float)), dtype=float)
    lam, Phi = sla.eigsh(K, k=NMODES, M=Mm, sigma=0.0, which='LM', OPinv=op)
    o = np.argsort(lam); lam, Phi = lam[o], Phi[:, o]
    Phi = Phi/np.sqrt(np.einsum('ik,ik->k', Phi, Mm@Phi))[None]
    full = np.zeros((3*m.n, NMODES)); full[fr] = Phi
    Hrow = np.zeros((G.nrow, 3*m.n))
    for p in range(len(G.points)):
        for d in range(3):
            Hrow[3*p+d, d::3] = G.H[p]
    HPhi = Hrow@full
    PhiS = full[G.Sdof].reshape(len(G.S), 3, NMODES)
    # static: rows adjoint, surface compliance, known loads
    fpos = -np.ones(3*m.n, int); fpos[fr] = np.arange(len(fr))
    sfree = fpos[G.Sdof]; assert (sfree >= 0).all()
    Xo = np.asarray(fac(np.ascontiguousarray(Hrow[:, fr].T)))      # (nf,R)
    XoS = Xo[sfree].reshape(len(G.S), 3, G.nrow)
    E = np.zeros((len(fr), len(sfree))); E[sfree, np.arange(len(sfree))] = 1.
    Cs = np.asarray(fac(E))[sfree].astype(np.float32).reshape(len(G.S), 3, len(G.S), 3)
    Uk = np.asarray(fac(np.ascontiguousarray(G.known[fr])))           # (nf,L)
    KO = Hrow[:, fr]@Uk; KS = Uk[sfree].reshape(len(G.S), 3, -1); PhiK = Phi.T@G.known[fr]
    sec = time.time()-t0
    np.savez(OUT/f'mat{idx:03d}.npz', lam=lam, HPhi=HPhi, PhiS=PhiS, XoS=XoS, Css=Cs, KO=KO, KS=KS, PhiK=PhiK,
             mat=np.array([np.nan if xc is None else xc, k, nu]))
    print(idx, mat, f'{sec:.1f}s', 'f1..3 Hz at E=6e4:', np.round(np.sqrt(6e4*lam[:3])/2/np.pi, 2), flush=True)


def build(i0, i1):
    OUT.mkdir(parents=True, exist_ok=True); G = Geometry(); mats = materials()
    for i in range(i0, min(i1, len(mats))):
        if not (OUT/f'mat{i:03d}.npz').exists():
            build_one(G, i, mats[i])


if __name__ == '__main__':
    if sys.argv[1] == 'build':
        build(int(sys.argv[2]), int(sys.argv[3]))
    else:
        G = Geometry(); print(len(materials()), len(G.S), len(G.q), G.nrow)
