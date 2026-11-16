"""Diagnostic only (uses truth): noise-free model error of the coarse linear model at the TRUE material
vs the nonlinear fine-mesh truth at the 20 markers, per episode.  Not used by any inference."""
import json, math, sys
import numpy as np, torch
from pathlib import Path
from scripts.patch_roles_v02 import model as M, bank as BK
R = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(R/'scripts/fig1_v01')); sys.path.insert(0, str(R/'scripts/stem_trial_v01'))
import block as B
bank = M.Bank('cuda', supports=False); pidx = {x: 1+i for i, x in enumerate(BK.PRESS_X)}
out = {}
for amp, sc, tg in (('full', 1.0, ''), ('low', 1/3, '-a0.333333')):
    for w, xc in B.WORLDS.items():
        imat = 37+1+int(round((xc+0.08)/0.02))*4+2
        th = np.array([[math.log(B.E_SOFT), (xc+0.08)/0.02+0.5, 1/9+2.5, 1.5, math.log(1e-4)]]); assert M.PatchModel.material_index(th)[0] == imat
        for ep, x in (('D0', 0.0), ('D1', 0.05), ('D2', -0.07)):
            e = [dict(name=ep, y=None, known=[('step', 0, 1.0), ('ramp', pidx[x], 30*sc)])]
            pm = M.PatchModel(bank, e)
            with torch.no_grad():
                d = pm.operators(th, 0, 0, list(range(60)))['d_'+ep][0].cpu().numpy().reshape(24, 20, 3)
            f = R/f'runs/fig1-v01/block/truth/{ep}-{w}.npz' if amp == 'full' else R/f'runs/patch-roles-v02/truth/{ep}-{w}{tg}.npz'
            z = np.load(f); m = B.mesh(tuple(z['cells'])); H = m.interpolation(B.MARKERS).toarray()
            t = np.einsum('mn,fnj->fmj', H, z['U'])[1:]
            err = (d-t)*1e3
            out[f'{amp}-{w}-{ep}'] = dict(rms_mm=float(np.sqrt((err**2).mean())), per_marker_rms_mm=np.round(np.sqrt((err**2).mean((0, 2))), 3).tolist(),
                                          signal_rms_mm=float(np.sqrt((t**2).mean())*1e3), late_frames_rms_mm=float(np.sqrt((err[12:]**2).mean())))
            print(amp, w, ep, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in out[f'{amp}-{w}-{ep}'].items()})
(R/'runs/patch-roles-v02/discrepancy_diag.json').write_text(json.dumps(out, indent=1))
