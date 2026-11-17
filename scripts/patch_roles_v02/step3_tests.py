"""Step-3 identity tests: (1) whitened Woodbury marginal == dense Gaussian log density with the full
spatio-temporal discrepancy covariance, for a 1F1S hidden episode + a known episode; (2) box footprint
integrates to 1 and covers the expected area.  Writes runs/patch-roles-v02/step3/tests.json"""
import json, math
from pathlib import Path
import numpy as np, torch
from scipy.stats import multivariate_normal
from scripts.patch_roles_v02 import model as M, step3_model as S3
R = Path(__file__).resolve().parents[2]


def main():
    bank = M.Bank('cuda'); rng = np.random.default_rng(0); pidx = {x: 1+i for i, x in enumerate(M.BK.PRESS_X)}
    eps = [dict(name='D0', y=rng.standard_normal((24, 60))*3e-4, known=[('step', 0, 1.), ('ramp', pidx[0.0], 30.)]),
           dict(name='D1', y=rng.standard_normal((24, 60))*3e-4, known=[('step', 0, 1.)], hidden=True)]
    sd, lt, ls = 1.2e-4, 0.04, 0.05
    pm = S3.Step3Model(bank, eps, sd, lt, ls)
    th = np.array([[math.log(6.3e4), 7.5, 1/9+2.5, 1.5, math.log(8e-5), math.log(0.8), 0.42, 0.5, math.log(0.02),
                    0.2, 0.3, math.log(0.012), math.log(2e3)]])
    with torch.no_grad():
        op = pm.operators(th, 1, 1, list(range(60))); ll, _ = pm.marginal(th, 1, 1, op)
    Ct = M.matern(M.TF[1:], lt); Cs = S3.se_kernel(S3.B.MARKERS, ls)
    Rn = 8e-5**2*np.eye(1440)+sd**2*np.kron(Ct, np.kron(Cs, np.eye(3)))
    ref = 0.
    for e in eps:
        d = op['d_'+e['name']][0].cpu().numpy().ravel(); C = Rn.copy()
        if e.get('hidden'):
            G = op['G'][0].cpu().numpy().reshape(1440, -1)
            C = C+G@((0.8*pm.SIGMA_F)**2*np.kron(np.eye(3), M.matern(M.TF[1:], pm.LEN_F)))@G.T
        ref += multivariate_normal(d, C).logpdf(e['y'].ravel())
    res = dict(marginal_abs_err=float(abs(ll.item()-ref)), marginal=float(ref))
    with torch.no_grad():
        u, b = M.surface_uv([0.05, 0.63, 0.0])
        w = pm.box(np.array([u]), np.array([b]), np.array([math.log(0.02)]))[0].cpu().numpy()
        gate = torch.sigmoid((0.02-(pm.b.q-torch.tensor([0.05, 0.63, 0.], dtype=torch.float64, device=pm.dev)).abs())/S3.EPS_BOX).prod(-1)
        area = float((gate*pm.b.qw).sum())
    res.update(box_weight_sum=float(w.sum()), box_area_cm2_h20mm=area*1e4)
    res['passed'] = bool(res['marginal_abs_err'] < 1e-3 and abs(res['box_weight_sum']-1) < 1e-9 and abs(res['box_area_cm2_h20mm']-16) < 1.5)
    out = R/'runs/patch-roles-v02/step3'; out.mkdir(parents=True, exist_ok=True)
    (out/'tests.json').write_text(json.dumps(res, indent=1)); print(res)


if __name__ == '__main__':
    main()
