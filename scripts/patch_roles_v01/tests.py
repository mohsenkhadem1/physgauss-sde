"""Exact identity checks against the published code (no SMC).

1. Published-family forward (1 force, 1 support) == mechanics.LinearGaussianBody.solve.
2. Woodbury amplitude-space marginal/conditional == continuous.MarginalGaussianLikelihood
   (dense C = R + H Lambda H^T) for published and general families, hierarchical tau,
   partial time masks (looks) and probe episodes.
3. Footprints integrate to one.
"""
import json, sys, math
from pathlib import Path
import numpy as np
import torch
from scripts.joint_field_identification.mechanics import LinearGaussianBody
from scripts.joint_field_identification.continuous import MarginalGaussianLikelihood
from scripts.joint_field_identification import surface as pub
from scripts.patch_roles_v01 import model as M

OUT = Path(__file__).resolve().parents[2] / 'runs/patch-roles-v01/tests.json'


def dense_reference(pm, theta, nF, nS, data):
    """Build full H [B, nobs, T*nA] and offset, evaluate with published dense code."""
    with torch.no_grad():
        fw = pm.forward(theta, nF, nS)
    G = fw['G'].cpu().numpy(); D = fw['D'].cpu().numpy(); nA = fw['nA']; T = pm.T
    mask = np.asarray(data['mask'], bool); y = np.asarray(data['y'])
    cmd = np.asarray(data['cmd']).reshape(-1, T)
    rows = [(t, m) for t in range(T) for m in range(y.shape[1]) if mask[t, m]]
    B = len(theta)
    H = np.zeros((B, len(rows), T * nA)); off = np.zeros((B, len(rows)))
    for i, (t, m) in enumerate(rows):
        H[:, i, t * nA:(t + 1) * nA] = G[:, m]
        off[:, i] = D[:, m] @ cmd[:, t] if cmd.shape[0] else 0
    yy = np.array([y[t, m] for t, m in rows])
    tau = fw['tau'].cpu().numpy()
    lls, mus, covs = [], [], []
    for b in range(B):
        K = (tau[b] ** 2) * M.FORCE_STD ** 2 * np.kron(M.matern(), np.eye(nA))
        lik = MarginalGaussianLikelihood(lambda th: dict(H=H[b:b + 1], offset=off[b:b + 1]), yy, K, M.NOISE ** 2)
        ll = lik(np.zeros((1, 1)))[0]
        for index, yp in data.get('probes', []):
            res = yp - fw['Pr'][b, :, index].cpu().numpy()
            ll += -.5 * (res @ res / M.NOISE ** 2 + len(yp) * math.log(2 * math.pi * M.NOISE ** 2))
        m, c = lik.conditional(np.zeros((1, 1)))
        lls.append(ll); mus.append(m[0]); covs.append(c[0])
    return np.array(lls), np.array(mus), np.array(covs)


def main():
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.set_num_threads(2)
    rng = np.random.default_rng(1)
    res = {}
    # 1. forward identity, published family
    pm = M.PatchModel('published', device=dev)
    body = pm.body
    _, bounds = pm.layout(1, 1)
    th = bounds[:, 0] + rng.random((6, len(bounds))) * (bounds[:, 1] - bounds[:, 0])
    th5 = np.stack([th[:, 0], th[:, 1], th[:, 3], th[:, 4], th[:, 2]], 1)   # PARAMETERS order
    with torch.no_grad():
        ref = body.solve(torch.tensor(th5, device=dev))['unit_response']      # [B,n,3,2]
        K, unknown, tau, E = pm.operators(th, 1, 1)
        U = torch.cholesky_solve(unknown, torch.linalg.cholesky(K)).view(len(th), body.n, 3, 2)
    res['published_forward_max_rel_error'] = float((U - ref).abs().max() / ref.abs().max())
    assert res['published_forward_max_rel_error'] < 1e-10
    # 2. likelihood identity
    for family, nF, nS, hier in [('published', 1, 1, False), ('published', 2, 1, True),
                                 ('general', 2, 1, False), ('general', 1, 2, True), ('general', 0, 2, False)]:
        pm = M.PatchModel(family, device=dev, hierarchical=hier,
                          commands=[M.gspec('top', .035, .03, (0, 0, -1))])
        _, bounds = pm.layout(nF, nS)
        th = bounds[:, 0] + rng.random((5, len(bounds))) * (bounds[:, 1] - bounds[:, 0])
        mask = np.zeros((8, 60), bool); mask[:, :40] = True; mask[4, 40:] = True; mask[2, 5] = False
        data = dict(y=rng.normal(scale=1e-4, size=(8, 60)), mask=mask, cmd=.1 * np.sin(np.pi * M.TIMES)[None],
                    probes=[(1, rng.normal(scale=1e-4, size=40))])
        with torch.no_grad():
            fw = pm.forward(th, nF, nS)
            ll, mu, S, _ = pm.marginal(fw, data, conditionals=True)
        rl, rm, rc = dense_reference(pm, th, nF, nS, data)
        key = f'{family}_{nF}F{nS}S_hier{int(hier)}'
        e = dict(loglik_abs=float(np.max(np.abs(ll.cpu().numpy() - rl))))
        if mu is not None:
            e['mean_rel'] = float(np.max(np.abs(mu.cpu().numpy().reshape(len(th), -1) - rm)) / np.max(np.abs(rm)))
            e['cov_rel'] = float(np.max(np.abs(S.cpu().numpy() - rc)) / np.max(np.abs(rc)))
        res[key] = e
        assert e['loglik_abs'] < 1e-6 and e.get('mean_rel', 0) < 1e-8 and e.get('cov_rel', 0) < 1e-8, (key, e)
    # 3. footprints
    surf = M.Surface(LinearGaussianBody('block', (8, 4, 4), device=dev))
    c, _ = M.surface_point(np.array([.01, -.05]), np.array([.15, .239]))
    w = surf.gauss(c, [.01, .02])
    res['footprint_sum_error'] = float((w.sum(1) - 1).abs().max())
    res['passed'] = True
    OUT.parent.mkdir(parents=True, exist_ok=True); OUT.write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == '__main__':
    main()
