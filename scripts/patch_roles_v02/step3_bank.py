"""Volume extension of the step-2 bank for step-3 volume force patches: per material, the static adjoint
Khat^-1 H^T and the 80 modes at ALL free nodes (float32).  Same eigen-solve settings as bank.py, so the modes
may differ from bank.py's within near-degenerate pairs, so HPhiV (the SAME basis) is stored and used with PhiV.
Usage: step3_bank.py I0 I1"""
import sys, time
import numpy as np
import scipy.sparse.linalg as sla
from scripts.patch_roles_v02 import bank as BK
sys.path.insert(0, str(BK.R/'scripts/fig1_v01')); sys.path.insert(0, str(BK.R/'scripts/stem_trial_v01'))
import fem, block as B
OUT = BK.R/'runs/patch-roles-v02/step3/bankvol'


def main(i0, i1):
    OUT.mkdir(parents=True, exist_ok=True); G = BK.Geometry(); mats = BK.materials(); m = G.m; fr = m.free
    fnodes = np.unique(fr//3); assert len(fnodes)*3 == len(fr)
    for idx in range(i0, min(i1, len(mats))):
        p = OUT/f'mat{idx:03d}.npz'
        if p.exists(): continue
        xc, k, nu = mats[idx]; t0 = time.time()
        model = fem.Elastic(m, B.modulus(m, xc, contrast=k, e_soft=1.0), nu, B.RHO)
        K = model.linear_K().tocsr()[fr][:, fr]; Mm = model.M.tocsr()[fr][:, fr]; fac = fem._factor(K)
        op = sla.LinearOperator(K.shape, matvec=lambda v: fac(np.asarray(v, float)), dtype=float)
        lam, Phi = sla.eigsh(K, k=BK.NMODES, M=Mm, sigma=0.0, which='LM', OPinv=op)
        o = np.argsort(lam); lam, Phi = lam[o], Phi[:, o]; Phi = Phi/np.sqrt(np.einsum('ik,ik->k', Phi, Mm@Phi))[None]
        ref = np.load(BK.OUT/f'mat{idx:03d}.npz')
        full = np.zeros((3*m.n, BK.NMODES)); full[fr] = Phi
        PhiS = full[G.Sdof].reshape(len(G.S), 3, -1)
        # basis change into bank.py's modes (eigsh start vectors are random, near-degenerate pairs may rotate):
        # Phi_bank ~= Phi_new @ O^T  ->  modal load in bank basis = O @ (Phi_new^T f)
        O = np.linalg.lstsq(PhiS.reshape(-1, BK.NMODES), ref['PhiS'].reshape(-1, BK.NMODES), rcond=None)[0].T
        orth_err = float(np.abs(O@O.T-np.eye(BK.NMODES)).max())
        rel = float(np.abs(lam/ref['lam']-1).max())
        Hrow = np.zeros((G.nrow, 3*m.n))
        for q in range(len(G.points)):
            for d in range(3): Hrow[3*q+d, d::3] = G.H[q]
        Xo = np.asarray(fac(np.ascontiguousarray(Hrow[:, fr].T)))
        fullp = np.zeros((3*m.n, BK.NMODES)); fullp[fr] = Phi; HPhiV = Hrow@fullp
        np.savez(p, XoV=Xo.reshape(len(fnodes), 3, -1).astype(np.float32), PhiV=Phi.reshape(len(fnodes), 3, -1).astype(np.float32),
                 fnodes=fnodes, lam_rel_diff=rel, HPhiV=HPhiV, lamV=lam, O=O, orth_err=orth_err)
        print(idx, f'{time.time()-t0:.1f}s lam rel diff {rel:.2e} O orth err {orth_err:.2e}', flush=True)


if __name__ == '__main__':
    main(int(sys.argv[1]), int(sys.argv[2]))
