"""Dev diagnostic (uses truth): what does the spurious support compensate?  Noise-free truth markers for D0+D1
(commanded) vs model at the true material; least-squares refit of (logE), (logE + 6 lowest-mode frequency
factors), (logE + one support spring at the step-3 gate MAP location)."""
import json, math, sys
import numpy as np, torch
from pathlib import Path
from scipy.optimize import minimize
from scripts.patch_roles_v02 import model as M, step3_model as S3, bank as BK
R = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(R/'scripts/fig1_v01')); import block as B
amp = sys.argv[1] if len(sys.argv) > 1 else 'full'; world = 'Bright'; sc = 1.0 if amp == 'full' else 1/3
bank = M.Bank('cuda'); pidx = {x: 1+i for i, x in enumerate(BK.PRESS_X)}
truth = {}
for ep in ('D0', 'D1'):
    f = R/f'runs/fig1-v01/block/truth/{ep}-{world}.npz' if amp == 'full' else R/f'runs/patch-roles-v02/truth/{ep}-{world}-a0.333333.npz'
    z = np.load(f); m = B.mesh(tuple(z['cells'])); H = m.interpolation(B.MARKERS).toarray()
    truth[ep] = np.einsum('mn,fnj->fmj', H, z['U'])[1:].reshape(24, 60)
eps = [dict(name='D0', y=truth['D0'], known=[('step', 0, 1.), ('ramp', pidx[0.0], 30*sc)]),
       dict(name='D1', y=truth['D1'], known=[('step', 0, 1.), ('ramp', pidx[0.05], 30*sc)])]
pm = S3.Step3ModelFreq(bank, eps, dict(sigma_d=1e-4, len_t=.1, len_s=.05))
base = np.array([math.log(6e4), 7.5, 1/9+2.5, 1.5, math.log(1e-4)])
def resid(x, nfreq, sup=None):
    th = np.concatenate([base[:1]+x[:1], base[1:], np.zeros(0)])
    fe = np.zeros(6); fe[:nfreq] = x[1:1+nfreq]
    nS = 0; extra = []
    if sup is not None:
        nS = 1; extra = [sup[0], sup[1], math.log(sup[2]), x[-1]]
    th = np.concatenate([th, extra, fe, [0, 0, 0]])[None]
    with torch.no_grad():
        op = pm.operators(th, 0, nS, list(range(60)))
    return np.concatenate([(truth[e['name']]-op['d_'+e['name']][0].cpu().numpy()).ravel() for e in eps])
out = {}
rms = lambda r: float(np.sqrt(np.mean(r**2))*1e3)
out['true_params'] = rms(resid(np.zeros(1), 0))
r1 = minimize(lambda x: np.sum(resid(x, 0)**2), np.zeros(1), method='Nelder-Mead'); out['fit_E'] = (rms(resid(r1.x, 0)), r1.x.tolist())
r2 = minimize(lambda x: np.sum(resid(x, 6)**2), np.zeros(7), method='Powell'); out['fit_E_6freq'] = (rms(resid(r2.x, 6)), r2.x.tolist())
u, b = M.surface_uv([-0.03, 0.545, -0.06])
r3 = minimize(lambda x: np.sum(resid(x, 0, (u, b, 0.008))**2), np.array([0., math.log(2500)]), method='Nelder-Mead'); out['fit_E_support'] = (rms(resid(r3.x, 0, (u, b, 0.008))), r3.x.tolist())
r = resid(r1.x, 0).reshape(2, 24, 20, 3)
out['residual_fitE_by_frame_um'] = (np.sqrt((r**2).mean((0, 2, 3)))*1e3).round(1).tolist()
out['residual_fitE_by_marker_um'] = (np.sqrt((r**2).mean((0, 1, 3)))*1e3).round(1).tolist()
out['residual_fitE_by_coord_um'] = (np.sqrt((r**2).mean((0, 1, 2)))*1e3).round(1).tolist()
print(json.dumps(out, indent=1))
(R/f'runs/patch-roles-v02/step3/diag-{amp}.json').write_text(json.dumps(out, indent=1))
