"""Identity test for Step3ModelShared: whitened Helmert/Woodbury marginal == dense Gaussian over stacked episodes."""
import json, math
from pathlib import Path
import numpy as np, torch
from scipy.stats import multivariate_normal
from scripts.patch_roles_v02 import model as M, step3_model as S3
R = Path(__file__).resolve().parents[2]
bank = M.Bank('cuda'); rng = np.random.default_rng(1); pidx = {x: 1+i for i, x in enumerate(M.BK.PRESS_X)}
eps = [dict(name='D0', y=rng.standard_normal((24, 60))*3e-4, known=[('step', 0, 1.), ('ramp', pidx[0.0], 30.)]),
       dict(name='D1', y=rng.standard_normal((24, 60))*3e-4, known=[('step', 0, 1.)], hidden=True),
       dict(name='P', y=rng.standard_normal((24, 60))*3e-4, known=[('step', 0, 1.), ('ramp', pidx[0.06], 30.)])]
h = dict(a=4e-5, b=3e-5, c=5e-5, d=6e-5, len_t=0.2, len_s=0.05)
pm = S3.Step3ModelShared(bank, eps, h)
th = np.array([[math.log(6.3e4), 7.5, 1/9+2.5, 1.5, math.log(8e-5), math.log(0.8), 0.25, 0.5, math.log(0.02), 0.2, 0.3, math.log(0.012), math.log(2e3), math.log(1.5), math.log(0.5)]])
with torch.no_grad():
    op = pm.operators(th, 1, 1, list(range(60))); ll, _ = pm.marginal(th, 1, 1, op)
Kt = M.matern(M.TF[1:], h['len_t']); SE = S3.se_kernel(S3.B.MARKERS, h['len_s'])
Kind = 8e-5**2*np.eye(1440)+np.kron(Kt, np.kron(h['a']**2*np.eye(20)+h['b']**2*SE, np.eye(3)))
Ksh = np.kron(Kt, np.kron(h['c']**2*np.eye(20)+h['d']**2*SE, np.eye(3)))
C = np.kron(np.eye(3), Kind)+np.kron(np.ones((3, 3)), Ksh)
G = op['G'][0].cpu().numpy().reshape(1440, -1); Gs = np.zeros((4320, G.shape[1])); Gs[1440:2880] = G
C = C+Gs@(pm.SIGMA_F**2*np.kron(np.diag([0.8**2, 1.5**2, 0.5**2]), M.matern(M.TF[1:], pm.LEN_F)))@Gs.T
d = np.concatenate([op['d_'+e['name']][0].cpu().numpy().ravel() for e in eps]); y = np.concatenate([e['y'].ravel() for e in eps])
ref = multivariate_normal(d, C).logpdf(y)
res = dict(shared_marginal_abs_err=float(abs(ll.item()-ref)), value=float(ref)); res['passed'] = res['shared_marginal_abs_err'] < 1e-3
print(res); (R/'runs/patch-roles-v02/step3/tests_shared.json').write_text(json.dumps(res, indent=1))
