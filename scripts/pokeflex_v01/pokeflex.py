"""PokeFlex loader with a sealed outcome store (protocol: scripts/pokeflex_v01/protocol.json).

Free (inputs): robot trajectory, camera parameters, segments, passive dropping meshes, and the reconstructed
mesh / confidence / images at the START frame of every segment (rest state).  Calibration poke (poke 1 of the
designated take) and development takes: all outcomes free.
Gated (held-out outcomes): forces, meshes, confidences, images of frames inside held-out pokes -- returned only
if reports/pokeflex-v01/prediction-freeze.json exists and verifies (sha256 of predictions, protocol, code and
sealed manifests), otherwise SealedError.

CLI:  python -I pokeflex.py freeze PRED_DIR   |   python -I pokeflex.py verify   |   python -I pokeflex.py summary
Meshes are returned in metres (stored in mm).  Always read data with `python -I`.
"""
import hashlib, json, sys, time
from functools import lru_cache
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent; R = HERE.parents[1]
OUT = R/'runs/pokeflex-v01'; REP = R/'reports/pokeflex-v01'; FREEZE = REP/'prediction-freeze.json'
PROT = json.loads((HERE/'protocol.json').read_text())
DEV = set(PROT['development_takes'])
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()


class SealedError(PermissionError):
    pass


# ---------------------------------------------------------------- inventory / roles
def objects():
    return sorted({p.name for p in (OUT/'takes/poking').glob('*')} | {p.name for p in (OUT/'takes/dropping').glob('*')})


def takes(obj, kind='poking'):
    d = OUT/'takes'/kind/obj
    key = (lambda p: int(p.name.rsplit('_T', 1)[1]))
    return [p.name for p in sorted(d.glob(f'{obj}_T*'), key=key) if (p/'.extracted').exists()]


def is_dev(obj, take):
    return f'{obj}/{take}' in DEV


def segments(obj, take):
    return json.loads((OUT/'segments'/obj/f'{take}.json').read_text())


def flags(obj, take):
    """Per-poke completeness flags (segment.py annotate; from rest meshes + trajectory only)."""
    return json.loads((OUT/'segments'/obj/f'{take}.flags.json').read_text())['pokes']


def designated_take(obj):
    """Lowest-numbered non-development poking take that contains a complete poke (protocol rule).
    Refuses unless every poking take ETH lists for the object (expected_takes.json) is extracted, so the
    choice cannot depend on download progress."""
    exp = json.loads((Path(__file__).parent/'expected_takes.json').read_text())[obj]
    missing = sorted(set(exp)-set(takes(obj)))
    assert not missing, f'{obj}: takes not yet extracted {missing}; designation would depend on download state'
    for x in takes(obj):
        if not is_dev(obj, x) and any(f['complete'] for f in flags(obj, x)):
            return x
    return None


def calibration_poke(obj):
    t = designated_take(obj)
    return None if t is None else (t, next(f['poke'] for f in flags(obj, t) if f['complete']))


def role(obj, take, poke):
    """development | calibration | held_out | excluded (incomplete poke: sealed, never scored)."""
    if is_dev(obj, take):
        return 'development'
    cp = calibration_poke(obj)
    if cp and take == cp[0] and poke <= cp[1]:
        return 'calibration'  # the first complete poke of the designated take and anything before it in time
    return 'held_out' if flags(obj, take)[poke-1]['complete'] else 'excluded'


def heldout_frames(obj, take):
    """Set of frame numbers whose outcomes are sealed: frames of held-out pokes, excluding their start frame."""
    s = set()
    if is_dev(obj, take):
        return s
    pk = segments(obj, take)['pokes']
    for q in pk:
        if role(obj, take, q['poke']) in ('held_out', 'excluded'):
            s |= set(range(q['start_frame']+1, q['end_frame']+1))
    # the start frame of every poke is its rest/initial state (free input), even when it is also the end
    # frame of the preceding held-out poke
    return s - {q['start_frame'] for q in pk}


# ---------------------------------------------------------------- freeze / verify
def _code_hashes():
    return {p.name: sha(p) for p in sorted(HERE.glob('*.py'))}


def _manifests():
    return {p.relative_to(OUT).as_posix(): sha(p) for p in sorted((OUT/'sealed').rglob('MANIFEST.sha256.json'))}


def freeze(pred_dir):
    pred_dir = Path(pred_dir).resolve()
    assert not FREEZE.exists(), 'already frozen'
    files = {p.relative_to(R).as_posix(): sha(p) for p in sorted(pred_dir.rglob('*')) if p.is_file()}
    assert files, 'no prediction files'
    rec = dict(sealed_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), prediction_dir=pred_dir.relative_to(R).as_posix(),
               files=files, protocol_sha256=sha(HERE/'protocol.json'), objects_sha256=sha(HERE/'objects.json'),
               code_sha256=_code_hashes(), sealed_manifests=_manifests())
    REP.mkdir(parents=True, exist_ok=True); FREEZE.write_text(json.dumps(rec, indent=1))
    print('sealed', len(files), 'prediction files')


@lru_cache(1)
def verify():
    """Raise SealedError unless the freeze exists and every recorded hash still matches.  Returns the record."""
    if not FREEZE.exists():
        raise SealedError(f'held-out outcome requested but {FREEZE.relative_to(R)} does not exist')
    fz = json.loads(FREEZE.read_text())
    bad = [k for k, h in fz['files'].items() if not (R/k).exists() or sha(R/k) != h]
    bad += ['protocol.json'] if sha(HERE/'protocol.json') != fz['protocol_sha256'] else []
    bad += [k for k, h in fz['sealed_manifests'].items() if sha(OUT/k) != h]
    if bad:
        raise SealedError(f'changed after freeze: {bad[:5]}')
    return fz


def is_prediction_sealed(path):
    """True only for files recorded in the freeze (anything else must not be scored)."""
    k = Path(path).resolve().relative_to(R).as_posix()
    return k in verify()['files']


def _gate(obj, take, frames):
    held = heldout_frames(obj, take)
    if any(f in held for f in frames):
        verify()
        return True
    return False


def _checked(obj, take, path, gated):
    """Read a sealed file; if gated, also check its sha against the take manifest."""
    sealed = OUT/'sealed/poking'/obj/take
    if gated:
        man = json.loads((sealed/'MANIFEST.sha256.json').read_text())
        k = Path(path).relative_to(sealed).as_posix()
        if man.get(k) != sha(path):
            raise SealedError(f'sealed file does not match manifest: {k}')
    return path


# ---------------------------------------------------------------- free inputs
def robot_trajectory(obj, take):
    rd = json.loads((OUT/'takes/poking'/obj/take/'robot_traj.json').read_text())
    return dict(frame=np.array([int(r['frame']) for r in rd]), T_WT=np.array([r['T_WT'] for r in rd], float),
                T_WE=np.array([r['T_WE'] for r in rd], float), T_WB=np.array([r['T_WB'] for r in rd], float),
                jointPos=np.array([r['jointPos'] for r in rd], float))


def camera_params(obj, take):
    d = OUT/'takes/poking'/obj/take/'cameras'
    return {f'{p.parent.parent.name}/{p.parent.name}': json.loads(p.read_text()) for p in sorted(d.glob('*/*/camera_parameters.json'))}


def read_obj(path):
    """Minimal OBJ reader (v / f only, triangles).  Returns V (n,3) float64 in file units, F (m,3) int64."""
    txt = Path(path).read_text().splitlines()
    V = np.array(' '.join(l[2:] for l in txt if l.startswith('v ')).split(), float).reshape(-1, 3)
    fl = [l.split()[1:] for l in txt if l.startswith('f ')]
    assert all(len(x) == 3 for x in fl), 'non-triangle face'
    F = np.array([[int(t.split('/')[0]) for t in x] for x in fl], np.int64) - 1
    return V, F


def drop_meshes(obj, take, frames=None):
    """Passive footage (free): list of (frame, V[m], F)."""
    d = OUT/'takes/dropping'/obj/take/'meshes'
    out = []
    for p in sorted(d.glob('mesh-f*.obj')):
        fr = int(p.stem[6:])
        if frames is None or fr in frames:
            V, F = read_obj(p); out.append((fr, V*1e-3, F))
    return out


# ---------------------------------------------------------------- outcomes (gated per frame)
def mesh(obj, take, frame):
    """Reconstructed mesh at a poking frame: (V [m], F, confidence per vertex or None)."""
    p = OUT/'sealed/poking'/obj/take/'meshes'/f'mesh-f{frame:05d}.obj'
    g = _gate(obj, take, [frame])
    V, F = read_obj(_checked(obj, take, p, g))
    c = OUT/'sealed/poking'/obj/take/'mesh_confidence'/f'{frame:05d}.npy'
    conf = np.load(_checked(obj, take, c, g), allow_pickle=False) if c.exists() else None
    return V*1e-3, F, conf


def rest_mesh(obj, take, poke):
    """Mesh at the start frame of a poke segment (free input; needs no role information)."""
    fr = segments(obj, take)['pokes'][poke-1]['start_frame']
    V, F = read_obj(OUT/'sealed/poking'/obj/take/'meshes'/f'mesh-f{fr:05d}.obj')
    c = OUT/'sealed/poking'/obj/take/'mesh_confidence'/f'{fr:05d}.npy'
    return V*1e-3, F, (np.load(c, allow_pickle=False) if c.exists() else None)


def forces(obj, take, frames=None):
    """Wrench rows [Fx,Fy,Fz (N), Tx,Ty,Tz (Nm)] for 1-based frames (default: all).  World-frame forces (see audit)."""
    fr = list(range(1, 151)) if frames is None else list(frames)
    g = _gate(obj, take, fr)
    W = np.load(_checked(obj, take, OUT/'sealed/poking'/obj/take/'forces.npy', g), allow_pickle=False)
    return W[np.array(fr)-1]


def free_air_frames(obj, take, min_gap_mm=10.0):
    """Frames whose tool tip is >= min_gap_mm outside the rest mesh of its poke (free inputs only)."""
    tip = robot_trajectory(obj, take)['T_WT'][:, :3, 3]; out = []
    for q in segments(obj, take)['pokes']:
        V, F, _ = rest_mesh(obj, take, q['poke'])
        for i in range(q['start_idx'], q['end_idx']+1):
            d, ins = signed_tip_gap(tip[i], V, F)
            if not ins and d >= min_gap_mm:
                out.append(i+1)
    return sorted(set(out))


def force_offset(obj, take):
    """Sensor offset (protocol force_baseline): median wrench over free-air frames (gated like forces)."""
    fr = free_air_frames(obj, take)
    return np.median(forces(obj, take, fr), 0) if fr else np.zeros(6)


def image(obj, take, rig, cam, channel, frame):
    """Path to an extracted camera image (extract.py --images); gated per frame."""
    p = OUT/'sealed/poking'/obj/take/'images'/rig/str(cam)/channel/f'{frame:05d}.png'
    if not p.exists():
        raise FileNotFoundError(f'{p} (run extract.py --images)')
    return _checked(obj, take, p, _gate(obj, take, [frame]))


# ---------------------------------------------------------------- surface metric
def _tri_point_dist(P, A, B, C):
    """Exact distance from points P (n,3) to triangles (n,3 each), vectorised (Ericson, RTCD 5.1.5)."""
    ab, ac, ap = B-A, C-A, P-A
    d1, d2 = (ab*ap).sum(-1), (ac*ap).sum(-1)
    bp = P-B; d3, d4 = (ab*bp).sum(-1), (ac*bp).sum(-1)
    cp = P-C; d5, d6 = (ab*cp).sum(-1), (ac*cp).sum(-1)
    va = d3*d6-d5*d4; vb = d5*d2-d1*d6; vc = d1*d4-d3*d2
    den = np.where(np.abs(va+vb+vc) < 1e-30, 1e-30, va+vb+vc)
    v = vb/den; w = vc/den; Q = A+ab*v[:, None]+ac*w[:, None]  # interior
    def setq(m, q):
        Q[m] = q[m]
    with np.errstate(divide='ignore', invalid='ignore'):
        # edges
        m = (vc <= 0) & (d1 >= 0) & (d3 <= 0); t = d1/(d1-d3); setq(m, A+ab*t[:, None])
        m = (vb <= 0) & (d2 >= 0) & (d6 <= 0); t = d2/(d2-d6); setq(m, A+ac*t[:, None])
        m = (va <= 0) & ((d4-d3) >= 0) & ((d5-d6) >= 0); t = (d4-d3)/((d4-d3)+(d5-d6)); setq(m, B+(C-B)*t[:, None])
    # vertices last (take precedence)
    setq((d1 <= 0) & (d2 <= 0), A); setq((d3 >= 0) & (d4 <= d3), B); setq((d6 >= 0) & (d5 <= d6), C)
    return np.linalg.norm(P-Q, axis=-1)


def point_to_mesh(P, V, F, k=24):
    """Distance from points P to the surface (V,F): exact point-triangle distance over the k triangles with
    nearest centroids (exact for meshes whose triangles are small relative to the distances involved)."""
    from scipy.spatial import cKDTree
    cen = V[F].mean(1); k = min(k, len(F))
    _, idx = cKDTree(cen).query(P, k=k)
    idx = idx.reshape(len(P), k)
    T = V[F[idx]]  # (n,k,3,3)
    Pk = np.repeat(P[:, None], k, 1).reshape(-1, 3)
    d = _tri_point_dist(Pk, T[..., 0, :].reshape(-1, 3), T[..., 1, :].reshape(-1, 3), T[..., 2, :].reshape(-1, 3))
    return d.reshape(len(P), k).min(1)


def signed_tip_gap(tip, V, F):
    """Unsigned distance tip->surface (mm) and whether the tip is inside the closed-ish mesh (ray parity along +x)."""
    d = float(point_to_mesh(tip[None], V, F)[0]*1e3)
    A, B, C = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    # Moller-Trumbore ray along +x
    e1, e2 = B-A, C-A; dirv = np.array([1.0, 0.31, 0.17]); dirv /= np.linalg.norm(dirv)
    h = np.cross(dirv, e2); a = (e1*h).sum(1); ok = np.abs(a) > 1e-14
    f = np.where(ok, 1/np.where(ok, a, 1), 0); s = tip-A; u = f*(s*h).sum(1)
    q = np.cross(s, e1); v = f*(q @ dirv); t = f*(e2*q).sum(1)
    hits = ok & (u >= 0) & (v >= 0) & (u+v <= 1) & (t > 0)
    return d, bool(hits.sum() % 2 == 1)


def sample_surface(V, F, n, seed=0, vals=None):
    """Area-weighted surface samples (and barycentric interpolation of per-vertex vals)."""
    rng = np.random.default_rng(seed)
    A = 0.5*np.linalg.norm(np.cross(V[F[:, 1]]-V[F[:, 0]], V[F[:, 2]]-V[F[:, 0]]), axis=1)
    t = rng.choice(len(F), n, p=A/A.sum())
    u, v = rng.random(n), rng.random(n); fl = u+v > 1; u[fl], v[fl] = 1-u[fl], 1-v[fl]
    w = np.stack([1-u-v, u, v], 1); P = (V[F[t]]*w[..., None]).sum(1)
    return P, (None if vals is None else (vals[F[t]]*w).sum(1))


def surface_error_mm(Vp, Fp, Vt, Ft, conf_t=None, n=20000, seed=0, tool_tip=None, tool_axis=None, excl_radius=0.03):
    """Symmetric point-to-mesh chamfer (metres in, mm out).  Returns dict with pred->truth, truth->pred,
    symmetric mean, (if conf_t) the confidence-weighted truth->pred and symmetric values, and (if tool_tip and
    tool_axis are given) the chamfer with samples within excl_radius of the tool axis (above the tip) removed on
    both sides -- the reconstruction does not resolve the dent under the probe (audit)."""
    Pp, _ = sample_surface(Vp, Fp, n, seed)
    Pt, ct = sample_surface(Vt, Ft, n, seed+1, conf_t)
    dp = point_to_mesh(Pp, Vt, Ft)*1e3; d_pt = dp.mean()
    dt = point_to_mesh(Pt, Vp, Fp)*1e3; d_tp = dt.mean()
    out = dict(pred_to_truth=float(d_pt), truth_to_pred=float(d_tp), chamfer=float(0.5*(d_pt+d_tp)))
    if ct is not None:
        w = np.clip(ct, 0, None); wt = float((dt*w).sum()/max(w.sum(), 1e-12))
        out.update(truth_to_pred_confw=wt, chamfer_confw=float(0.5*(d_pt+wt)))
    if tool_tip is not None:
        a = np.asarray(tool_axis, float); a = a/np.linalg.norm(a); c = np.asarray(tool_tip, float)
        def keep(X):
            r = X-c; ax = r @ a
            return ~((np.linalg.norm(r-np.outer(ax, a), axis=1) < excl_radius) & (ax < 0.01))  # cylinder from 1 cm below tip upwards
        kp, kt = keep(Pp), keep(Pt)
        out.update(chamfer_excl_tool=float(0.5*(dp[kp].mean()+dt[kt].mean())), excl_frac=[float(1-kp.mean()), float(1-kt.mean())])
    return out


# ---------------------------------------------------------------- CLI
def summary():
    rows = {}
    for o in objects():
        pk = {}
        for t in takes(o):
            sp = OUT/'segments'/o/f'{t}.json'
            n = len(json.loads(sp.read_text())['pokes']) if sp.exists() else None
            pk[t] = dict(pokes=n, roles=[role(o, t, i+1) for i in range(n or 0)])
        rows[o] = dict(dropping=takes(o, 'dropping'), poking=pk, designated=designated_take(o))
    print(json.dumps(rows, indent=1))


if __name__ == '__main__':
    cmd = sys.argv[1] if len(sys.argv) > 1 else 'summary'
    if cmd == 'freeze':
        freeze(sys.argv[2])
    elif cmd == 'verify':
        print('ok', len(verify()['files']), 'files')
    else:
        summary()
