"""Geometry / units audit on the DEVELOPMENT take and the dropping (passive) takes only.
-> runs/pokeflex-v01/audit/audit.json
usage: python -I audit.py
"""
import json, sys
from pathlib import Path

import numpy as np
from scipy.signal import find_peaks
from scipy.spatial import cKDTree

HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
import pokeflex as P

OUT = P.OUT/'audit'; OUT.mkdir(parents=True, exist_ok=True)
OBJS = P.PROT['objects_considered']


def topology(frames):
    """frames: list of (V,F).  Counts, and whether vertex i keeps its identity between consecutive frames."""
    nv = [len(V) for V, _ in frames]; nf = [len(F) for _, F in frames]
    same_faces, idx_disp, nn_disp = [], [], []
    for (V0, F0), (V1, F1) in zip(frames[:-1], frames[1:]):
        nn_disp.append(float(np.median(cKDTree(V1).query(V0)[0])*1e3))
        if len(V0) == len(V1):
            idx_disp.append(float(np.median(np.linalg.norm(V1-V0, axis=1))*1e3))
            same_faces.append(bool(len(F0) == len(F1) and np.array_equal(F0, F1)))
    return dict(n_frames=len(frames), n_vertices_min=min(nv), n_vertices_max=max(nv), n_unique_vertex_counts=len(set(nv)),
                n_faces_min=min(nf), n_faces_max=max(nf),
                consecutive_pairs_same_vertex_count=len(idx_disp), consecutive_pairs_identical_faces=int(sum(same_faces)),
                median_same_index_disp_mm=float(np.median(idx_disp)) if idx_disp else None,
                median_nearest_neighbour_disp_mm=float(np.median(nn_disp)))


signed_tip_gap = P.signed_tip_gap


def dev_audit(obj, take):
    tr = P.robot_trajectory(obj, take); T = tr['T_WT']; E = tr['T_WE']; tip = T[:, :3, 3]
    seg = P.segments(obj, take); n = len(tip)
    meshes = [P.mesh(obj, take, i+1) for i in range(n)]
    W = P.forces(obj, take); Fw = W[:, :3]; Tq = W[:, 3:]; fmag = np.linalg.norm(Fw, axis=1)
    base = np.median(W[:5], 0); fc = np.linalg.norm(Fw-base[:3], axis=1)
    res = dict(take=f'{obj}/{take}')
    res['units'] = dict(mesh_extent_mm=(np.ptp(meshes[0][0], 0)*1e3).round(1).tolist(),
                        note='OBJ vertices are in mm (extent of FoamDice ~155 mm edge); robot poses in m.')
    res['topology'] = topology([(V, F) for V, F, _ in meshes])
    conf = [c for _, _, c in meshes]
    res['confidence'] = dict(per_vertex=all(c is not None and len(c) == len(V) for (V, _, _), c in zip(meshes, conf)),
                             mean=float(np.mean([c.mean() for c in conf])), frac_zero=float(np.mean([(c == 0).mean() for c in conf])))
    # tool tip vs surface
    gaps = [signed_tip_gap(tip[i], meshes[i][0], meshes[i][1]) for i in range(n)]
    g = np.array([d for d, _ in gaps]); ins = np.array([b for _, b in gaps]); sg = np.where(ins, -g, g)
    rest_gap = [signed_tip_gap(tip[i], meshes[q['start_idx']][0], meshes[q['start_idx']][1]) for q in seg['pokes'] for i in [q['deepest_idx']]]
    per = []
    on = fc > 5.0
    for q in seg['pokes']:
        s, d, e = q['start_idx'], q['deepest_idx'], q['end_idx']
        # geometric contact onset: first frame in press where tip is within 3 mm of / inside the REST mesh of the poke
        Vr, Fr, _ = meshes[s]
        gr = np.array([P.point_to_mesh(tip[i][None], Vr, Fr)[0]*1e3 for i in range(s, d+1)])
        ir = np.array([signed_tip_gap(tip[i], Vr, Fr)[1] for i in range(s, d+1)])
        sgr = np.where(ir, -gr, gr)
        geo_on = int(np.argmax(sgr <= 3.0)) + s if (sgr <= 3.0).any() else None
        f_on = int(np.argmax(on[s:d+1])) + s if on[s:d+1].any() else None
        # surface height under the tool (top of mesh within 8 mm laterally of the tip, along approach axis)
        a = np.array(seg['approach_axis_world'])
        def surf_depth(i):
            V = meshes[i][0]; r = V-tip[i]; lat = np.linalg.norm(r-np.outer(r @ a, a), axis=1)
            m = lat < 0.008
            return float((-(r[m] @ a)).min()*1e3) if m.any() else np.nan  # mm from tip to surface, + = surface ahead of tip
        sd = np.array([surf_depth(i) for i in range(s, e+1)])
        surf_pos = np.array([(tip[i] @ a)*1e3 + sd[i-s] for i in range(s, e+1)])  # surface location along approach
        dpk = int(np.nanargmax(surf_pos)) + s
        per.append(dict(poke=q['poke'], frames=[s+1, d+1, e+1],
                        gap_tip_to_surface_at_deepest_mm=float(sg[d]),
                        gap_tip_to_surface_during_force_contact_mm=dict(median=float(np.median(sg[s:e+1][on[s:e+1]])) if on[s:e+1].any() else None,
                                                                        min=float(sg[s:e+1][on[s:e+1]].min()) if on[s:e+1].any() else None,
                                                                        max=float(sg[s:e+1][on[s:e+1]].max()) if on[s:e+1].any() else None),
                        tip_depth_into_rest_mesh_at_deepest_mm=float(sgr[-1]),
                        contact_onset_frame_geometric=None if geo_on is None else geo_on+1,
                        contact_onset_frame_force=None if f_on is None else f_on+1,
                        surface_under_tool_deepest_frame=dpk+1, tool_deepest_frame=d+1,
                        force_peak_frame=int(np.argmax(fmag[s:e+1]))+s+1,
                        surface_minus_tip_along_approach_mm_at_deepest=float(sd[d-s])))
    res['tool_vs_mesh'] = dict(per_poke=per, signed_gap_mm_all_frames=sg.round(2).tolist(),
                               note='signed distance tool tip (T_WT origin) -> reconstructed surface; negative = tip inside the mesh')
    # force frame: compare world vs tool-frame hypotheses at the deepest frames
    a = np.array(seg['approach_axis_world']); rows = []
    for q in seg['pokes']:
        d = q['deepest_idx']; f = Fw[d]-base[:3]; R = T[d, :3, :3]
        rows.append(dict(poke=q['poke'], F_raw=f.round(2).tolist(), cos_raw_vs_minus_approach=float(f @ -a/np.linalg.norm(f)),
                         cos_toolframe_rotated_vs_minus_approach=float((R @ f) @ -a/np.linalg.norm(f)),
                         cos_flangeframe_rotated_vs_minus_approach=float((E[d, :3, :3] @ f) @ -a/np.linalg.norm(f))))
    # torque consistency: tau ?= (c - o) x F for o = flange (E) or tip (T), c = tip
    taus = {}
    for name, o in (('about_flange', E[:, :3, 3]), ('about_tip', tip)):
        pred = np.cross(tip-o, Fw-base[:3]); m = on
        taus[name] = float(np.linalg.norm((Tq-base[3:])[m]-pred[m], axis=1).mean())
    taus['measured_mean_norm_Nm'] = float(np.linalg.norm((Tq-base[3:])[on], axis=1).mean())
    res['force_frame'] = dict(per_poke=rows, torque_residual_Nm=taus, baseline_wrench_first5=base.round(3).tolist(),
                              peak_N=float(fc.max()))
    # timing
    T_flange_z = E[:, :3, 3]
    res['timing'] = dict(robot_frames=[int(tr['frame'][0]), int(tr['frame'][-1])], n_mesh_frames=len(meshes),
                         note='robot_data frame k <-> mesh-f0000k (same 1-based index, 30 fps)')
    # support plane: lowest mesh points (along -approach) in the rest frame
    V0 = meshes[0][0]; h = V0 @ -a
    res['support'] = dict(lowest_point_along_up_mm=float(h.min()*1e3), p1_mm=float(np.percentile(h, 1)*1e3),
                          world_y_min_mm=float(V0[:, 1].min()*1e3), up_axis_world=(-a).tolist())
    return res


def drop_audit():
    out = {}
    for o in OBJS:
        for t in P.takes(o, 'dropping'):
            fr = P.drop_meshes(o, t)
            if not fr:
                continue
            rec = dict(topology=topology([(V, F) for _, V, F in fr]), frames=[fr[0][0], fr[-1][0]])
            Vl = fr[-1][1]
            # resting support: lowest y in the last frames (object at rest), world y up
            ys = [float(V[:, 1].min()*1e3) for _, V, _ in fr[-5:]]
            cen = np.array([V.mean(0) for _, V, _ in fr])*1e3
            rec.update(last5_min_y_mm=ys, centroid_y_first_last_mm=[float(cen[0, 1]), float(cen[-1, 1])],
                       extent_last_mm=(np.ptp(Vl, 0)*1e3).round(1).tolist(),
                       lowest_band_flatness_mm=float(np.ptp(np.sort(Vl[:, 1])[:50])*1e3))
            out[f'{o}/{t}'] = rec
            print(o, t, rec['topology']['n_vertices_min'], rec['topology']['n_vertices_max'], ys[-1], flush=True)
    return out


if __name__ == '__main__':
    rep = dict(dev={k: dev_audit(*k.split('/')) for k in P.PROT['development_takes']})
    (OUT/'audit.json').write_text(json.dumps(rep, indent=1))
    rep['dropping'] = drop_audit()
    (OUT/'audit.json').write_text(json.dumps(rep, indent=1)); print('done')
