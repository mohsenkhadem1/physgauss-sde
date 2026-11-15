"""Export stage-B inputs through the gated loader: free inputs of every held-out poke (rest scan at its start frame,
tool trajectory, contact depth from free inputs) and the calibration window outcomes (allowed for fitting).
Calibration pokes that start in contact are dropped (same rule as pfx_ours.fit_object 'excluded_started_in_contact').
No held-out outcome is requested (the loader would raise SealedError).  usage: python -I prep_inputs.py [OBJ ...]"""
import json, sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent/'pokeflex_v01'))
import pokeflex as P
import pfx_common as C
from sb_common import INP, tool_pose, tool_keep

N_SAMPLES = 4000


def poke_arrays(obj, t, k):
    q = C.poke(obj, t, k); fr, dep = C.contact_depth(obj, t, k)
    T = P.robot_trajectory(obj, t)['T_WT'][fr-1]
    V, F, _ = P.rest_mesh(obj, t, k)
    return dict(frames=fr, T=T, dep=dep, V=V, F=F, start_frame=q['start_frame'], end_frame=q['end_frame'],
                approach=C.approach_axis(obj, t), y_s=float(V[:, 1].min()))


def main(objs):
    for obj in objs:
        d = INP/obj; d.mkdir(parents=True, exist_ok=True)
        cal, off, n_air = C.calibration_force(obj)
        sic = [[t, k] for t, k, fr, Fm, dep in cal if dep[0] > 0]
        cal = [c for c in cal if c[4][0] <= 0]
        for t, k, fr, Fm, dep in cal:
            a = poke_arrays(obj, t, k); assert np.array_equal(a['frames'], fr)
            S = np.zeros((len(fr), N_SAMPLES, 3)); keep = np.zeros((len(fr), N_SAMPLES), bool)
            for j, f in enumerate(fr):
                Vt, Ft, conf = P.mesh(obj, t, int(f))
                S[j], _ = P.sample_surface(Vt, Ft, N_SAMPLES, seed=int(f))
                tip, ax = tool_pose(a['T'][j]); keep[j] = tool_keep(S[j], tip, ax)
            np.savez_compressed(d/f'cal_{t}-p{k}.npz', **a, Fmeas=Fm, S=S.astype(np.float32), Skeep=keep)
        ho = C.heldout_pokes(obj)
        for t, k in ho:
            np.savez_compressed(d/f'ho_{t}-p{k}.npz', **poke_arrays(obj, t, k))
        info = dict(obj=obj, mass_kg=json.loads((HERE.parent/'pokeflex_v01/objects.json').read_text())['objects'][obj]['mass_g']*1e-3,
                    calibration_pokes=[[t, k] for t, k, *_ in cal], excluded_started_in_contact=sic, offset_N=off.tolist(),
                    offset_frames=n_air, heldout=[[t, k] for t, k in ho])
        (d/'object.json').write_text(json.dumps(info, indent=1))
        print(obj, 'cal', info['calibration_pokes'], 'excluded', sic, 'held-out', len(ho), flush=True)


def dev():
    """Development take FoamDice/FoamDice_T1 (debugging only, never used for fitting): poke 1, cal-format export with
    raw-window offset, written to inputs/_dev."""
    obj, t, k = 'FoamDice', 'FoamDice_T1', 1
    assert P.is_dev(obj, t)
    d = INP/'_dev'; d.mkdir(parents=True, exist_ok=True)
    a = poke_arrays(obj, t, k); fr = a['frames']
    W = P.forces(obj, t, fr)[:, :3]; air = W[a['dep'] < -0.010]; off = np.median(air, 0) if len(air) else np.zeros(3)
    S = np.zeros((len(fr), N_SAMPLES, 3)); keep = np.zeros((len(fr), N_SAMPLES), bool)
    for j, f in enumerate(fr):
        Vt, Ft, conf = P.mesh(obj, t, int(f))
        S[j], _ = P.sample_surface(Vt, Ft, N_SAMPLES, seed=int(f))
        tip, ax = tool_pose(a['T'][j]); keep[j] = tool_keep(S[j], tip, ax)
    np.savez_compressed(d/f'cal_{t}-p{k}.npz', **a, Fmeas=W-off, S=S.astype(np.float32), Skeep=keep)
    print('dev export', t, k, len(fr), 'frames')


if __name__ == '__main__':
    if sys.argv[1:] == ['--dev']:
        dev()
    else:
        main(sys.argv[1:] or C.OBJECTS)
