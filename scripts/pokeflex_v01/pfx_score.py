"""Score sealed PokeFlex predictions (protocol 'scores'). Runs only after reports/pokeflex-v01/prediction-freeze.json
exists; only files recorded in the freeze are scored (pokeflex.is_prediction_sealed).
usage: python -I pfx_score.py [--surface-every N]   -> reports/pokeflex-v01/scores.json (+ per-poke rows)
Force: per scored frame |F_pred - (F_meas - offset)|; offset = pokeflex.force_offset (median over free-air frames of
the whole take). Reported: RMSE over scored frames (3-vector) and RMSE of the component along -approach, per poke;
mean over pokes per object. Surface: tool-excluded symmetric chamfer (primary), plain and confidence-weighted
chamfer; every Nth scored frame (default 1 = every frame), mean per poke, mean over pokes per object."""
import json, sys
from multiprocessing import Pool
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import pokeflex as P
from pfx_common import OBJECTS, PRED, approach_axis, heldout_pokes, poke

EVERY = int(sys.argv[sys.argv.index('--surface-every')+1]) if '--surface-every' in sys.argv else 1


def score_poke(args):
    method, obj, take, k = args
    f = PRED/method/obj/f'{take}-p{k}.npz'
    if not f.exists() or not P.is_prediction_sealed(f) or not P.is_prediction_sealed(f.with_suffix('.json')):
        return None
    z = np.load(f); fr = z['frames']; a = approach_axis(obj, take)
    Fm = P.forces(obj, take, fr)[:, :3]-P.force_offset(obj, take)[:3]
    e = z['force']-Fm; ea = e @ -a
    row = dict(method=method, object=obj, take=take, poke=k, frames=len(fr), force_rmse_N=float(np.sqrt((e**2).sum(1).mean())),
               force_along_rmse_N=float(np.sqrt((ea**2).mean())), peak_meas_N=float(np.abs(Fm @ -a).max()))
    if 'risk' in z.files:
        row['predicted_risk'] = float(z['risk'])
    T = P.robot_trajectory(obj, take)['T_WT']; Vr, Fr, _ = P.rest_mesh(obj, take, k)
    s = []
    for j in range(0, len(fr), EVERY):
        Vt, Ft, conf = P.mesh(obj, take, int(fr[j]))
        Vp, Fp = (z['surf_V'][j].astype(float), z['surf_F']) if 'surf_V' in z.files else (Vr, Fr)
        Tk = T[fr[j]-1]
        s.append(P.surface_error_mm(Vp, Fp, Vt, Ft, conf_t=conf, tool_tip=Tk[:3, 3], tool_axis=Tk[:3, 2]))
    for key in ('chamfer_excl_tool', 'chamfer', 'chamfer_confw'):
        v = [x[key] for x in s if key in x]
        if v:
            row[key+'_mm'] = float(np.mean(v))
    return row


def main():
    fz = P.verify()
    methods = sorted({Path(k).parts[3] for k in fz['files']})
    jobs = [(m, o, t, k) for m in methods for o in OBJECTS for t, k in heldout_pokes(o)]
    with Pool(24) as pool:
        rows = [r for r in pool.map(score_poke, jobs, chunksize=1) if r]
    agg = {}
    for m in methods:
        for o in OBJECTS:
            rr = [r for r in rows if r['method'] == m and r['object'] == o]
            if rr:
                agg.setdefault(m, {})[o] = {k: float(np.mean([r[k] for r in rr if k in r])) for k in rr[0] if isinstance(rr[0][k], float)} | dict(pokes=len(rr))
    out = dict(freeze_utc=fz['sealed_utc'], surface_every=EVERY, per_object=agg, rows=rows)
    (P.REP/'scores.json').write_text(json.dumps(out, indent=1))
    for m, d in agg.items():
        print(m, {o: round(v['force_rmse_N'], 2) for o, v in d.items()})


if __name__ == '__main__':
    main()
