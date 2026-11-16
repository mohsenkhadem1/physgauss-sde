"""Post-hoc contact-region metrics for PokeFlex v01 + stage B (NOT pre-registered; secondary, reported alongside the
protocol metrics).  Added after review: the primary surface metric excludes a 30 mm cylinder around the tool axis,
which is where contact mechanics matter most, and the protocol force RMSE averages over non-contact frames too.

Per held-out poke, per method (v01 files must be in the verified freeze; stage-B files must match the committed
stage-B manifest):
  force_contact_rmse_N       3-vector force RMSE over geometric-contact frames only (tool tip past the rest surface,
                             free inputs: pfx_common.contact_depth > 0)
  force_contact_along_rmse_N same, along -approach
  peak_force_err_N           |max pred along-force - max measured along-force| over the poke
  peak_force_rel             peak_force_err_N / measured peak
  tool_region_mm             surface error restricted to the excluded tool region: mean distance from measured-mesh
                             samples inside the 30 mm tool cylinder to the predicted surface, plus predicted samples
                             in that cylinder to the measured mesh (symmetric, same samplers/seeds as the protocol),
                             contact frames only.  The reconstruction under-resolves the dent (protocol audit), so the
                             null (rest-mesh) value is the reference for how much dent is visible at all.
  (full-surface chamfer without exclusion = protocol 'chamfer_mm', already in scores*.json; compare.py reports it.)
usage: python -I contact_metrics.py [--surface-every N]  -> reports/pokeflex-v01/contact-metrics.json"""
import json, sys
from multiprocessing import Pool
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent/'pokeflex_v01')); sys.path.insert(0, str(HERE))
from verify_full import verify_full
FZ = verify_full()
import score_stageb as SB  # checks the committed stage-B manifest; exposes SB.SEALED / SB.SB
import pokeflex as P
from pfx_common import OBJECTS, PRED, approach_axis, contact_depth, heldout_pokes

EVERY = int(sys.argv[sys.argv.index('--surface-every')+1]) if '--surface-every' in sys.argv else 1
V01 = set(FZ['files'])
METHODS = {'ours': PRED/'ours', 'fem_lsq': PRED/'fem_lsq', 'hooke': PRED/'hooke', 'null': PRED/'null',
           'PhysTwin': SB.SB/'PhysTwin', 'PAC-NeRF': SB.SB/'PAC-NeRF'}
EXCL_R = 0.03


def sealed(f):
    f = Path(f).resolve()
    return f.relative_to(P.R).as_posix() in V01 or str(f) in SB.SEALED


def tool_region(Vp, Fp, Vt, Ft, tip, axis, n=20000, seed=0):
    Pp, _ = P.sample_surface(Vp, Fp, n, seed); Pt, _ = P.sample_surface(Vt, Ft, n, seed+1)
    a = axis/np.linalg.norm(axis)

    def inside(X):
        r = X-tip; ax = r @ a
        return (np.linalg.norm(r-np.outer(ax, a), axis=1) < EXCL_R) & (ax < 0.01)
    ip, it = inside(Pp), inside(Pt)
    if ip.sum() < 20 or it.sum() < 20:
        return None
    dp = P.point_to_mesh(Pp[ip], Vt, Ft).mean()*1e3; dt = P.point_to_mesh(Pt[it], Vp, Fp).mean()*1e3
    return float(0.5*(dp+dt))


def score(args):
    m, obj, take, k = args
    f = METHODS[m]/obj/f'{take}-p{k}.npz'
    if not f.exists() or not sealed(f) or not sealed(f.with_suffix('.json')):
        return None
    z = np.load(f); fr = z['frames']; a = approach_axis(obj, take)
    Fm = P.forces(obj, take, fr)[:, :3]-P.force_offset(obj, take)[:3]
    cf, dep = contact_depth(obj, take, k); dmap = dict(zip(cf.tolist(), dep.tolist()))
    c = np.array([dmap.get(int(x), -1.0) > 0 for x in fr])
    row = dict(method=m, object=obj, take=take, poke=k, contact_frames=int(c.sum()))
    Fa_m, Fa_p = Fm @ -a, z['force'] @ -a
    pk = float(Fa_m.max())
    row.update(peak_meas_along_N=pk, peak_force_err_N=float(abs(Fa_p.max()-pk)), peak_force_rel=float(abs(Fa_p.max()-pk)/max(pk, 1e-6)))
    if c.any():
        e = z['force'][c]-Fm[c]
        row.update(force_contact_rmse_N=float(np.sqrt((e**2).sum(1).mean())), force_contact_along_rmse_N=float(np.sqrt(((e @ -a)**2).mean())))
        T = P.robot_trajectory(obj, take)['T_WT']; Vr, Fr, _ = P.rest_mesh(obj, take, k)
        vals = []
        for j in np.flatnonzero(c)[::EVERY]:
            Vt, Ft, _ = P.mesh(obj, take, int(fr[j]))
            Vp, Fp = (z['surf_V'][j].astype(float), z['surf_F']) if 'surf_V' in z.files else (Vr, Fr)
            Tk = T[fr[j]-1]
            v = tool_region(Vp, Fp, Vt, Ft, Tk[:3, 3], Tk[:3, 2])
            if v is not None:
                vals.append(v)
        if vals:
            row['tool_region_mm'] = float(np.mean(vals))
    return row


def main():
    jobs = [(m, o, t, k) for m in METHODS for o in OBJECTS for t, k in heldout_pokes(o)]
    with Pool(20) as pool:
        rows = [r for r in pool.map(score, jobs, chunksize=1) if r]
    out = dict(note='post-hoc secondary metrics (not pre-registered); see module docstring', freeze_utc=FZ['sealed_utc'],
               surface_every=EVERY, excl_radius_m=EXCL_R, rows=rows)
    (P.REP/'contact-metrics.json').write_text(json.dumps(out, indent=1))
    print(len(rows), 'rows')


if __name__ == '__main__':
    main()
