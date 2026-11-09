"""RGB-D marker measurements for our method (protocol_v02 part B). Usage: rgbd_track.py EPISODE WORLD
RGB is re-rendered and asserted byte-identical to the scene GIC receives; depth gets iid noise.
Markers are tracked in RGB with pyramidal Lucas-Kanade from the projected known rest positions,
depth is read as a 3x3 median at the tracked pixel and back-projected. The truth trajectory is
used only to render the images and, separately, to report the measurement error as a diagnostic."""
import json, sys
from pathlib import Path
import numpy as np
import cv2
from PIL import Image
sys.path.insert(0, str(Path(__file__).resolve().parent))
import fem, render
from ours import MARKERS_RGBD

R = Path(__file__).resolve().parents[2]
P2 = json.loads((Path(__file__).parent/'protocol_v02.json').read_text())
SD = 1.5e-3
CAM = np.array([1]*8+[9]*8)


def main(episode, world):
    src = 'W1000' if episode == 'D0' else world
    z = np.load(R/f'runs/stem-trial-v01/truth/{episode}-{src}.npz')
    mesh = fem.Mesh(tuple(int(c) for c in z['cells'])); tris = mesh.surface_triangles()
    cams, K = render.cameras(); F = len(z['U'])
    rng = np.random.default_rng(20261109+{'D0': 0, 'D1': 1}[episode]*100+int(world[1:]))
    scene = R/f'runs/stem-trial-v01/scenes/{episode}-{src}/data'
    y = np.zeros((F, 16, 3)); status = np.ones((F, 16), bool)
    for cam in (1, 9):
        idx = np.where(CAM == cam)[0]
        uv0, _ = render.project(cams[cam], K, MARKERS_RGBD[idx])
        pts = uv0.astype(np.float32).reshape(-1, 1, 2); prev = None
        for f in range(F):
            img, alpha, depth = render.render(cams[cam], K, z['X']+z['U'][f], z['X'], tris, return_depth=True)
            rgb = np.round(img*255).astype(np.uint8)
            assert np.array_equal(rgb, np.asarray(Image.open(scene/f'm_{cam}_{f}.png'))), 'RGB differs from GIC input'
            gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
            if prev is not None:
                pts, st, _ = cv2.calcOpticalFlowPyrLK(prev, gray, pts, None, winSize=(21, 21), maxLevel=3,
                                                      criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 50, 1e-3))
                status[f, idx] &= st.ravel().astype(bool)
            prev = gray
            noisy = np.where(depth > 0, depth+SD*rng.standard_normal(depth.shape), 0.0)
            for k, (u, v) in zip(idx, pts.reshape(-1, 2)):
                iu, iv = int(round(u-.5)), int(round(v-.5))
                patch = noisy[max(iv-1, 0):iv+2, max(iu-1, 0):iu+2]; patch = patch[patch > 0]
                if patch.size == 0:
                    status[f, k] = False; y[f, k] = np.nan; continue
                y[f, k] = render.unproject(cams[cam], K, np.array([u]), np.array([v]), np.array([np.median(patch)]))[0]
    # diagnostic only: error against the true material marker positions (not used by inference)
    H = mesh.interpolation(MARKERS_RGBD).toarray(); true = MARKERS_RGBD[None]+np.einsum('mn,fnj->fmj', H, z['U'])
    err = np.linalg.norm(y-true, axis=-1)
    out = R/'runs/stem-trial-v01/rgbd'; out.mkdir(parents=True, exist_ok=True)
    disp = y-y[0:1]  # displacement from the measured rest pose
    np.savez_compressed(out/f'obs-{episode}-{world}.npz', y=y, disp=disp, status=status, cam=CAM)
    diag = dict(episode=episode, world=world, frames=F, lost=int((~status).sum()),
                abs_err_mm_median=float(np.nanmedian(err)*1e3), abs_err_mm_p95=float(np.nanpercentile(err, 95)*1e3),
                disp_err_rms_mm=float(np.sqrt(np.nanmean(((disp-(true-true[0:1]))**2).sum(-1)))*1e3))
    (out/f'diag-{episode}-{world}.json').write_text(json.dumps(diag, indent=1)); print(diag)


if __name__ == '__main__':
    main(*sys.argv[1:3])
