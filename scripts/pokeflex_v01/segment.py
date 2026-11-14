"""Poke segmentation from the robot tool trajectory ONLY (T_WT; T_WE/jointPos recorded for reference).
Forces and meshes are NOT read here, except in `validate` on the declared development take.

Approach axis a = mean tool z-axis (unit, world).  Depth along approach s(t) = (p_tip(t) - p_tip(0)) . a.
Pokes = peaks of s with prominence >= PROM_MM; poke boundaries = argmin of s between consecutive peaks
(turning point of the retract/approach), first/last poke bounded by the take ends.  Within a poke:
press = [start, deepest], retract = [deepest, end]; dwell = frames with s within DWELL_MM of the peak.

usage: python -I segment.py [OBJ ...]        -> runs/pokeflex-v01/segments/<Obj>/<take>.json for every extracted take
                                                (+ <take>.flags.json: completeness flags from the rest meshes, see annotate)
       python -I segment.py --validate        -> force-peak agreement on the development take (dev only)
"""
import json, sys
from pathlib import Path

import numpy as np
from scipy.signal import find_peaks

R = Path(__file__).resolve().parents[2]
OUT = R/'runs/pokeflex-v01'
PROT = json.loads((Path(__file__).resolve().parent/'protocol.json').read_text())
PROM_MM, DWELL_MM, FPS = 20.0, 1.0, 30.0


def load_traj(obj, take):
    rd = json.loads((OUT/'takes/poking'/obj/take/'robot_traj.json').read_text())
    assert all('forces' not in r for r in rd)
    T = np.array([r['T_WT'] for r in rd], float); E = np.array([r['T_WE'] for r in rd], float)
    J = np.array([r['jointPos'] for r in rd], float); fr = [int(r['frame']) for r in rd]
    return fr, T, E, J


def segment(obj, take):
    fr, T, E, J = load_traj(obj, take)
    p = T[:, :3, 3]; zax = T[:, :3, 2]
    a = zax.mean(0); a /= np.linalg.norm(a)
    s = (p - p[0]) @ a * 1e3  # mm, positive = deeper along approach
    pk, _ = find_peaks(s, prominence=PROM_MM)
    n = len(s); bounds = [0]
    for i, j in zip(pk[:-1], pk[1:]):
        bounds.append(i + int(np.argmin(s[i:j+1])))
    bounds.append(n-1)
    pokes = []
    for k, d in enumerate(pk):
        st, en = bounds[k], bounds[k+1]
        dwell = np.where(np.abs(s[st:en+1] - s[d]) <= DWELL_MM)[0] + st
        v = p[d] - p[st]; mdir = v/np.linalg.norm(v)
        pokes.append(dict(
            poke=k+1, start_idx=int(st), deepest_idx=int(d), end_idx=int(en),
            start_frame=fr[st], deepest_frame=fr[d], end_frame=fr[en],
            press_frames=[fr[st], fr[d]], retract_frames=[fr[d], fr[en]],
            dwell_frames=[fr[int(dwell.min())], fr[int(dwell.max())]],
            stroke_mm=float(s[d] - s[st]), retract_mm=float(s[d] - s[en]),
            contact_point_nominal_m=p[d].tolist(),          # tool tip at deepest point
            approach_dir_tool_z=(zax[d]/np.linalg.norm(zax[d])).tolist(),
            approach_dir_motion=mdir.tolist(),               # unit(p_tip[deepest] - p_tip[start])
            tool_tip_path_m=p[st:en+1].tolist()))
    return dict(object=obj, take=take, method='tool-trajectory only (T_WT); forces/meshes not read',
                params=dict(prominence_mm=PROM_MM, dwell_mm=DWELL_MM), fps=FPS, frames=fr,
                approach_axis_world=a.tolist(), depth_along_approach_mm=s.round(3).tolist(), pokes=pokes)


def run_all(objs=None):
    for d in sorted((OUT/'takes/poking').glob('*/*')):
        if not (d/'robot_traj.json').exists() or objs and d.parent.name not in objs:
            continue
        seg = segment(d.parent.name, d.name)
        o = OUT/'segments'/d.parent.name/f'{d.name}.json'; o.parent.mkdir(parents=True, exist_ok=True)
        o.write_text(json.dumps(seg))
        fl = annotate(d.parent.name, d.name)
        print(d.name, 'pokes', len(seg['pokes']), [(q['start_frame'], q['deepest_frame'], q['end_frame'], round(q['stroke_mm']), f['start_gap_mm'], f['end_margin_mm'], f['complete'])
                                               for q, f in zip(seg['pokes'], fl)])


def validate():
    """Development take only: compare trajectory segmentation with force peaks / force-contact intervals."""
    rep = {}
    for obj, take in [tuple(x.split('/')) for x in PROT['development_takes']]:
        seg = segment(obj, take)
        F = np.load(OUT/'sealed/poking'/obj/take/'forces.npy')[:, :3]; f = np.linalg.norm(F, axis=1)
        fpk, _ = find_peaks(f, prominence=10.0)
        base = float(np.median(f[:5])); on = f > base + 5.0  # force-contact frames (5 N above the free-air baseline)
        rows = []
        for q in seg['pokes']:
            inside = [int(i) for i in fpk if q['start_idx'] <= i <= q['end_idx']]
            cf = np.where(on[q['start_idx']:q['end_idx']+1])[0] + q['start_idx']
            rows.append(dict(poke=q['poke'], seg=[q['start_idx']+1, q['deepest_idx']+1, q['end_idx']+1],
                             force_peaks_frames=[i+1 for i in inside], force_peak_N=[round(float(f[i]), 1) for i in inside],
                             lag_force_peak_minus_deepest=[i - q['deepest_idx'] for i in inside],
                             force_contact_frames=[int(cf.min())+1, int(cf.max())+1] if len(cf) else None))
        outside = int(on.sum() - sum(on[q['start_idx']:q['end_idx']+1].sum() for q in seg['pokes']))
        agree = len(fpk) == len(seg['pokes']) and all(len(r['force_peaks_frames']) == 1 for r in rows)
        rep[f'{obj}/{take}'] = dict(n_pokes_traj=len(seg['pokes']), n_force_peaks=len(fpk), one_to_one=agree,
                                    force_contact_frames_outside_segments=outside, pokes=rows)
    o = OUT/'audit/segmentation_validation_dev.json'; o.parent.mkdir(parents=True, exist_ok=True)
    o.write_text(json.dumps(rep, indent=1)); print(json.dumps(rep, indent=1))


def annotate(obj, take, contact_tol_mm=2.0):
    """Completeness flags from FREE inputs only (trajectory + rest mesh at each poke's start frame):
    start_out_of_contact: tool tip >= tol outside the rest mesh at the start frame;
    end_out_of_contact: at the end frame the tip has retracted (along the approach axis) above the depth at
    which it met the rest surface (s_contact = s_start + start gap), by >= tol;  complete = both."""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import pokeflex as P
    seg = json.loads((OUT/'segments'/obj/f'{take}.json').read_text())
    tip = np.array([r['T_WT'] for r in json.loads((OUT/'takes/poking'/obj/take/'robot_traj.json').read_text())])[:, :3, 3]
    s = np.array(seg['depth_along_approach_mm']); rows = []
    for q in seg['pokes']:
        V, F, _ = P.rest_mesh(obj, take, q['poke'])
        d, ins = P.signed_tip_gap(tip[q['start_idx']], V, F); g = -d if ins else d
        s_c = s[q['start_idx']] + g
        rows.append(dict(poke=q['poke'], start_gap_mm=round(g, 2), start_out_of_contact=bool(g >= contact_tol_mm),
                         end_margin_mm=round(float(s_c - s[q['end_idx']]), 2),
                         end_out_of_contact=bool(g >= contact_tol_mm and s_c - s[q['end_idx']] >= contact_tol_mm),
                         complete=bool(g >= contact_tol_mm and s_c - s[q['end_idx']] >= contact_tol_mm)))
    (OUT/'segments'/obj/f'{take}.flags.json').write_text(json.dumps(dict(object=obj, take=take, contact_tol_mm=contact_tol_mm,
        inputs='robot trajectory + rest mesh at poke start frames (free inputs)', pokes=rows), indent=1))
    return rows


if __name__ == '__main__':
    if '--validate' in sys.argv:
        validate()
    else:
        run_all(sys.argv[1:] or None)
