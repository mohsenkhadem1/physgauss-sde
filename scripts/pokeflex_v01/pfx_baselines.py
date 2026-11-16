"""Reference predictors (protocol amendment methods_v01): null and Hooke.
usage: python -I pfx_baselines.py            -> runs/pokeflex-v01/pred/{null,hooke}/... for every held-out poke"""
import json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import pokeflex as P
from pfx_common import OBJECTS, PRED, approach_axis, heldout_pokes, contact_depth, calibration_force, save_pred, poke


def fit_hooke(obj):
    cal, off, n_air = calibration_force(obj)
    num = den = 0.0
    for t, k, fr, F, dep in cal:
        a = approach_axis(obj, t); d = np.maximum(dep, 0)
        num += float((F @ -a) @ d); den += float(d @ d)
    k_hat = num/den
    res = np.concatenate([F - np.outer(k_hat*np.maximum(dep, 0), -approach_axis(obj, t)) for t, k, fr, F, dep in cal])
    fit = dict(k_N_per_m=k_hat, calibration_pokes=[[t, k] for t, k, *_ in cal], offset_N=off.tolist(), offset_frames=n_air,
               calibration_force_rmse_N=float(np.sqrt((res**2).sum(1).mean())))
    (PRED/'hooke'/obj).mkdir(parents=True, exist_ok=True); (PRED/'hooke'/obj/'fit.json').write_text(json.dumps(fit, indent=1))
    return fit


def main():
    for obj in OBJECTS:
        fit = fit_hooke(obj); n = 0
        for t, k in heldout_pokes(obj):
            fr, dep = contact_depth(obj, t, k); a = approach_axis(obj, t)
            sel = fr > poke(obj, t, k)['start_frame']  # scored frames: start+1 .. end
            save_pred('null', obj, t, k, fr[sel], np.zeros((sel.sum(), 3)), dict(method='null', surface='rest mesh of the poke'))
            save_pred('hooke', obj, t, k, fr[sel], np.outer(fit['k_N_per_m']*np.maximum(dep[sel], 0), -a),
                      dict(method='hooke', k_N_per_m=fit['k_N_per_m'], surface='rest mesh of the poke'))
            n += 1
        print(f"{obj:18s} k {fit['k_N_per_m']:8.1f} N/m  calib RMSE {fit['calibration_force_rmse_N']:.2f} N  offset frames {fit['offset_frames']}  held-out pokes {n}")


if __name__ == '__main__':
    main()
