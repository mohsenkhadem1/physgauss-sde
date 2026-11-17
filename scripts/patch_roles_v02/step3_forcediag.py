"""Dev diagnostic (uses truth): force-amplitude bias of the surface patch representation.  At the true material,
least-squares amplitude factor kappa of a surface box/Gaussian traction at the D1 window that best matches
(i) the model's own exact body-box press response and (ii) the truth D1 markers (gravity part from the model)."""
import json, math, sys
import numpy as np, torch
from pathlib import Path
from scripts.patch_roles_v02 import model as M, step3_model as S3, bank as BK
R = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(R/'scripts/fig1_v01')); import block as B
bank = M.Bank('cuda'); pidx = {x: 1+i for i, x in enumerate(BK.PRESS_X)}; out = {}
for amp, sc, tg in (('full', 1., ''), ('low', 1/3, '-a0.333333')):
    for w, xc in B.WORLDS.items():
        F = 30*sc
        eps = [dict(name='G', y=None, known=[('step', 0, 1.)]), dict(name='C', y=None, known=[('step', 0, 1.), ('ramp', pidx[0.05], F)])]
        f = R/f'runs/fig1-v01/block/truth/D1-{w}.npz' if amp == 'full' else R/f'runs/patch-roles-v02/truth/D1-{w}{tg}.npz'
        z = np.load(f); m = B.mesh(tuple(z['cells'])); H = m.interpolation(B.MARKERS).toarray()
        tru = np.einsum('mn,fnj->fmj', H, z['U'])[1:].reshape(24, 60)
        u, b = M.surface_uv([0.05, B.TOP, 0.0]); ramp = .5*(1-np.cos(np.pi*np.clip(M.TF[1:]/M.RAMP, 0, 1)))
        a = np.zeros((3, 24)); a[1] = -F*ramp
        row = {}
        for shape, logw in (('vol h=20mm', math.log(0.02)), ('vol h=15mm', math.log(0.015)), ('box h=20mm', math.log(0.02)), ('box h=15mm', math.log(0.015)), ('gauss s=12mm', math.log(0.012))):
            pm = S3.Step3ModelVol(bank, eps, dict(a=1e-4, b=1e-4, c=1e-4, d=1e-4, len_t=.1, len_s=.05)) if shape.startswith('vol') else S3.Step3Model(bank, eps, 1e-4, .1, .05, force_shape='box' if shape.startswith('box') else 'gauss')
            th = np.array([[math.log(B.E_SOFT), (xc+0.08)/0.02+0.5, 1/9+2.5, 1.5, math.log(1e-4), 0., u, b, logw]])
            with torch.no_grad():
                op = pm.operators(th, 1, 0, list(range(60)))
            g = op['d_G'][0].cpu().numpy(); cmd = op['d_C'][0].cpu().numpy()-g
            pr = np.einsum('grq,q->gr', op['G'][0].cpu().numpy(), a.ravel())
            k1 = float((pr*cmd).sum()/(pr*pr).sum()); k2 = float((pr*(tru-g)).sum()/(pr*pr).sum())
            row[shape] = dict(kappa_vs_model_bodybox=k1, kappa_vs_truth=k2, kappa_bodybox_vs_truth=float((cmd*(tru-g)).sum()/(cmd*cmd).sum()))
        out[f'{amp}-{w}'] = row; print(amp, w, json.dumps(row))
(R/'runs/patch-roles-v02/step3/forcediag.json').write_text(json.dumps(out, indent=1))
