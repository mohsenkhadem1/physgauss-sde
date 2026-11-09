"""RGB-only marker measurements for our method: same RGB frames GIC/PAC-NeRF receive (byte-identical),
pyramidal Lucas-Kanade tracking in two fitting cameras per marker (chosen at rest: visible in both by a
depth test on the known rest geometry, widest ray angle), linear two-view triangulation. No depth used.
Usage: rgb_track.py EPISODE WORLD [MARKER_SET]  (MARKER_SET: stem16 default)"""
import json, sys
from pathlib import Path
import numpy as np
import cv2
from PIL import Image
sys.path.insert(0, str(Path(__file__).resolve().parent))
import fem, render
from ours import MARKERS_RGBD

R = Path(__file__).resolve().parents[2]
FIT_CAMS = list(range(1, 11))


def proj_matrix(c2w, K):
    w2c = np.linalg.inv(c2w); F = np.diag([1, -1, -1])
    return K@np.hstack([F@w2c[:3, :3], (F@w2c[:3, 3])[:, None]])


def triangulate(P1, P2, u1, u2):
    A = np.stack([u1[0]*P1[2]-P1[0], u1[1]*P1[2]-P1[1], u2[0]*P2[2]-P2[0], u2[1]*P2[2]-P2[1]])
    X = np.linalg.svd(A)[2][-1]; return X[:3]/X[3]


def choose_pairs(markers, X_rest, tris, cams, K, normals):
    vis = {}
    for c in FIT_CAMS:
        _, a, d = render.render(cams[c], K, X_rest, X_rest, tris, return_depth=True)
        uv, dm = render.project(cams[c], K, markers)
        ok = []
        for (u, v), z in zip(uv, dm):
            iu, iv = int(u), int(v)
            ok.append(0 <= iu < 800 and 0 <= iv < 800 and a[iv, iu] > .99 and abs(d[iv, iu]-z) < 2e-3)
        vis[c] = np.array(ok)
    pairs = []
    for m, X in enumerate(markers):
        best = None
        for i in FIT_CAMS:
            for j in FIT_CAMS:
                if i < j and vis[i][m] and vis[j][m]:
                    ri = cams[i][:3, 3]-X; rj = cams[j][:3, 3]-X
                    ang = np.degrees(np.arccos(ri@rj/np.linalg.norm(ri)/np.linalg.norm(rj)))
                    face = min(normals[m]@ri/np.linalg.norm(ri), normals[m]@rj/np.linalg.norm(rj))
                    if ang >= 20 and (best is None or face > best[3]):
                        best = (i, j, ang, face)  # >=20 deg baseline, then the most face-on worse view
        assert best is not None, f'marker {m} not visible in two fitting cameras'
        pairs.append(best)
    return pairs


def main(episode, world, scene_root='runs/stem-trial-v01/scenes', truth_root='runs/stem-trial-v01/truth',
         out_root='runs/stem-trial-v01/rgb', markers=MARKERS_RGBD, cells=None, normals=None):
    src = 'W1000' if episode == 'D0' and 'stem' in scene_root else world
    z = np.load(R/truth_root/f'{episode}-{src}.npz')
    mesh = fem.Mesh(tuple(int(c) for c in z['cells'])) if cells is None else cells(z)
    tris = mesh.surface_triangles(); cams, K = render.cameras(); F = len(z['U'])
    if normals is None:
        normals = np.array([[0, 0, 1.]]*8+[[0, 1., 0]]*8)  # stem16: side (+z) then top (+y)
    pairs = choose_pairs(markers, mesh.X, tris, cams, K, normals)
    scene = R/scene_root/f'{episode}-{src}/data'
    tracks = {}
    for c in sorted({p[0] for p in pairs} | {p[1] for p in pairs}):
        idx = [m for m, p in enumerate(pairs) if c in p[:2]]
        uv0, _ = render.project(cams[c], K, markers[idx])
        pts = uv0.astype(np.float32).reshape(-1, 1, 2); prev = None; seq = []
        for f in range(F):
            gray = cv2.cvtColor(np.asarray(Image.open(scene/f'm_{c}_{f}.png')), cv2.COLOR_RGB2GRAY)
            if prev is not None:
                pts, st, _ = cv2.calcOpticalFlowPyrLK(prev, gray, pts, None, winSize=(21, 21), maxLevel=3,
                                                      criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 50, 1e-3))
                assert st.all(), (c, f)
            prev = gray; seq.append(pts.reshape(-1, 2).copy())
        tracks[c] = (idx, np.stack(seq))
    y = np.zeros((F, len(markers), 3))
    for m, (i, j, *_) in enumerate(pairs):
        ui = tracks[i][1][:, tracks[i][0].index(m)]; uj = tracks[j][1][:, tracks[j][0].index(m)]
        Pi, Pj = proj_matrix(cams[i], K), proj_matrix(cams[j], K)
        y[:, m] = [triangulate(Pi, Pj, ui[f], uj[f]) for f in range(F)]
    H = mesh.interpolation(markers).toarray(); true = markers[None]+np.einsum('mn,fnj->fmj', H, z['U'])
    err = np.linalg.norm(y-true, axis=-1)
    out = R/out_root; out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out/f'obs-{episode}-{world}.npz', y=y, status=np.ones(y.shape[:2], bool), pairs=np.array([p[:2] for p in pairs]))
    diag = dict(episode=episode, world=world, pairs=[list(map(float, p)) for p in pairs], abs_err_mm_median=float(np.median(err)*1e3),
                abs_err_mm_p95=float(np.percentile(err, 95)*1e3), frame0_err_mm_max=float(err[0].max()*1e3))
    (out/f'diag-{episode}-{world}.json').write_text(json.dumps(diag, indent=1)); print(diag)


if __name__ == '__main__':
    main(*sys.argv[1:3])
