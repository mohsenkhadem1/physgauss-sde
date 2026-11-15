"""Calibration-fit quality of the stage-B adapters (calibration window only; allowed outcomes):
per object and method, on the fitted calibration rollout (work/<adapter>/<Obj>/calpred_*.npz): protocol surface
error (pokeflex.surface_error_mm: chamfer_excl_tool, chamfer) on every 2nd calibration frame, force RMSE (3-vector)
and along -approach vs the offset-corrected measured force, and the null predictor (rest scan, zero force) alongside.
usage: python -I calib_quality.py  -> runs/pokeflex-v01/stageB/calib_quality.json"""
import json, sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent/'pokeflex_v01'))
import pokeflex as P
from sb_common import WORK, OUT, OBJECTS, obj_info, load_poke

ADAPTERS = {'PhysTwin': 'phystwin', 'PAC-NeRF': 'pacnerf'}


def main():
    res = {}
    for obj in OBJECTS:
        info = obj_info(obj); t, k = info['calibration_pokes'][0]; p = load_poke(obj, 'cal', t, k)
        a = p['approach']; Fm = p['Fmeas']
        rows = {}
        for meth, sub in ADAPTERS.items():
            f = WORK/sub/obj/f'calpred_{t}-p{k}.npz'
            if f.exists():
                z = np.load(f); rows[meth] = (z['force'], z['surf_V'].astype(float), z['surf_F'])
        rows['null'] = (np.zeros_like(Fm), np.repeat(p['V'][None], len(Fm), 0), p['F'])
        out = {}
        for meth, (F, SV, SF) in rows.items():
            ex, ch = [], []
            for j in range(0, len(p['frames']), 2):
                Vt, Ft, conf = P.mesh(obj, t, int(p['frames'][j])); T = p['T'][j]
                Fj = SF[np.linalg.norm(np.cross(SV[j][SF[:, 1]]-SV[j][SF[:, 0]], SV[j][SF[:, 2]]-SV[j][SF[:, 0]]), axis=1) > 0]  # drop zero-area faces
                e = P.surface_error_mm(SV[j], Fj, Vt, Ft, conf_t=conf, tool_tip=T[:3, 3], tool_axis=T[:3, 2])
                ex.append(e['chamfer_excl_tool']); ch.append(e['chamfer'])
            sel = p['frames'] > int(p['start_frame'])
            out[meth] = dict(chamfer_excl_tool_mm=float(np.mean(ex)), chamfer_mm=float(np.mean(ch)),
                             force_rmse_N=float(np.sqrt(((F-Fm)[sel]**2).sum(1).mean())),
                             force_along_rmse_N=float(np.sqrt((((F-Fm)[sel] @ -a)**2).mean())),
                             peak_pred_N=float((F @ -a).max()), peak_meas_N=float((Fm @ -a).max()))
        res[obj] = dict(calibration_poke=[t, k], **out)
        print(obj, json.dumps({m: {kk: round(v, 2) for kk, v in d.items()} for m, d in out.items()}), flush=True)
    (OUT/'stageB/calib_quality.json').write_text(json.dumps(res, indent=1))


if __name__ == '__main__':
    main()
