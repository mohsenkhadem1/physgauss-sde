"""Shared helpers for PokeFlex predictions (protocol amendment 2026-11-14, methods_v01).

Prediction files: runs/pokeflex-v01/pred/<method>/<Obj>/<take>-p<poke>.npz (one per held-out poke) with
  frames  (K,)    1-based frame numbers scored for that poke: start_frame+1 .. end_frame
  force   (K,3)   predicted contact force ON THE TOOL, world frame, N (zero out of contact)
  surf_V  (K,N,3) predicted surface vertices (m), optional; absent => the poke's rest mesh is the surface prediction
  surf_F  (M,3)   faces for surf_V
  risk    ()      optional predicted probability that this poke's force RMSE exceeds 2 N
and a <take>-p<poke>.json with method metadata. Fitted per-object parameters go to <method>/<Obj>/fit.json.
Only free inputs and calibration-window outcomes are read here (the loader gates everything else)."""
import json, sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import pokeflex as P

PRED = P.OUT/'pred'
PROT = json.loads((HERE/'protocol.json').read_text())
OBJECTS = PROT['objects_considered']


def approach_axis(obj, take):
    return np.array(P.segments(obj, take)['approach_axis_world'])


def heldout_pokes(obj):
    return [(t, q['poke']) for t in P.takes(obj) for q in P.segments(obj, t)['pokes'] if P.role(obj, t, q['poke']) == 'held_out']


def calibration_pokes(obj):
    """All pokes in the calibration window (the first complete poke of the designated take and any earlier poke)."""
    t, q = P.calibration_poke(obj)
    return [(t, k) for k in range(1, q+1)]


def poke(obj, take, k):
    return P.segments(obj, take)['pokes'][k-1]


def contact_depth(obj, take, k):
    """Per frame of the poke (start..end, 1-based frames): tool-tip depth (m) along the approach axis past the point
    where the tip meets the poke's rest surface (negative = not yet in geometric contact). Free inputs only."""
    q = poke(obj, take, k); seg = P.segments(obj, take)
    s = np.array(seg['depth_along_approach_mm'])*1e-3
    tip = P.robot_trajectory(obj, take)['T_WT'][:, :3, 3]
    V, F, _ = P.rest_mesh(obj, take, k)
    d, inside = P.signed_tip_gap(tip[q['start_idx']], V, F); gap = (-d if inside else d)*1e-3
    s_contact = s[q['start_idx']]+gap
    idx = np.arange(q['start_idx'], q['end_idx']+1)
    return idx+1, s[idx]-s_contact


def calibration_force(obj):
    """Offset-corrected measured forces over the calibration window (free: calibration frames are not sealed).
    Offset = median wrench over free-air frames (tip >= 10 mm outside the rest mesh) INSIDE the calibration window,
    so no held-out frame is touched. Returns list of (take, poke, frames, F (K,3))."""
    tip_rows = []
    out = []
    for t, k in calibration_pokes(obj):
        fr, dep = contact_depth(obj, t, k)
        W = P.forces(obj, t, fr)[:, :3]
        out.append((t, k, fr, W, dep))
        tip_rows.append(W[dep < -0.010])
    air = np.concatenate(tip_rows) if sum(len(r) for r in tip_rows) else np.zeros((0, 3))
    off = np.median(air, 0) if len(air) else np.zeros(3)
    return [(t, k, fr, W-off, dep) for t, k, fr, W, dep in out], off, len(air)


def save_pred(method, obj, take, k, frames, force, meta, surf_V=None, surf_F=None, risk=None):
    d = PRED/method/obj; d.mkdir(parents=True, exist_ok=True)
    arr = dict(frames=np.asarray(frames, int), force=np.asarray(force, float))
    if surf_V is not None:
        arr.update(surf_V=np.asarray(surf_V, np.float32), surf_F=np.asarray(surf_F, np.int32))
    if risk is not None:
        arr['risk'] = np.array(float(risk))
    np.savez_compressed(d/f'{take}-p{k}.npz', **arr)
    (d/f'{take}-p{k}.json').write_text(json.dumps(meta, indent=1))
