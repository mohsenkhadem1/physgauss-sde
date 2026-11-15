"""PAC-NeRF adapter, data: native PAC-NeRF dataset (data.pt: rays_o/rays_d/viewdirs (V,HW,3), rgb_all (V,1,HW,3) with a
white background, ray_mask_all) for the REST frame (start frame) of every stage-B poke, from the 6 PokeFlex colour
cameras (kinect 0/1 4K, realsense 0/1 wrist cameras with per-frame extrinsics, volucam 0/1).  Free inputs only
(rest-frame images via the gated loader, camera parameters, rest scan).
  * extrinsics are world->camera (verified by projecting the dev-take scan), OpenCV distortion -> cv2.undistort
  * foreground mask: silhouette of the rest scan projected into each view, dilated by 4 px, minus green-screen pixels
    (chroma key; PokeFlex ships no masks; native PAC-NeRF mattes against a background plate, which PokeFlex does not
    provide); background -> white
  * each view: square crop around the silhouette (+25 % margin) resampled to S x S, intrinsics adjusted
Env: runs/gap-crossmethod-v02/pacnerf-env (cv2, torch).  usage: python -I pac_prep.py [OBJ ...] | --dev"""
import json, sys
from pathlib import Path
import numpy as np
import cv2
cv2.setNumThreads(2)
import torch
HERE = Path(__file__).resolve().parent
R = HERE.parents[1]
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(R/'scripts/pokeflex_v01'))
import pokeflex as P
from sb_common import WORK, INP, OBJECTS, obj_info, load_poke


def get_rays_of_a_view(H, W, K, c2w, inverse_y=True):
    """= lib.nerf.dvgo.get_rays_of_a_view(H, W, K, c2w, inverse_y=True, mode='center') (copied to avoid building the
    CUDA extension in the data step)."""
    assert inverse_y
    j, i = torch.meshgrid(torch.arange(H, dtype=torch.float32)+0.5, torch.arange(W, dtype=torch.float32)+0.5, indexing='ij')
    dirs = torch.stack([(i-K[0][2])/K[0][0], (j-K[1][2])/K[1][1], torch.ones_like(i)], -1)
    rays_d = torch.sum(dirs[..., None, :]*c2w[:3, :3], -1)
    rays_o = c2w[:3, 3].expand(rays_d.shape)
    return rays_o, rays_d, rays_d/rays_d.norm(dim=-1, keepdim=True)

S = 256
CAMS = ['kinect/0', 'kinect/1', 'realsense/0', 'realsense/1', 'volucam/0', 'volucam/1']


def cam_params(c, rig, frame):
    if rig == 'kinect':
        return np.array(c['color_intrinsics']), np.array(c['color_dist_coeffs']), np.array(c['color_extrinsics'])
    if rig == 'volucam':
        return np.array(c['intrinsics']), np.array(c['dist_coeffs']), np.array(c['extrinsics'])
    return np.array(c['color_intrinsics']), np.array(c['color_dist_coeffs']), np.array(c['extrinsics'][f'{frame:05d}'])


def build(obj, take, frame, V, F, out):
    out.mkdir(parents=True, exist_ok=True)
    cams = P.camera_params(obj, take)
    ro, rd, vd, rgb, meta = [], [], [], [], []
    for name in CAMS:
        rig, ci = name.split('/')
        K, D, E = cam_params(cams[name], rig, frame)
        img = cv2.imread(str(P.image(obj, take, rig, ci, 'color', frame)))[..., ::-1]
        und = cv2.undistort(img, K, D, None, K)
        Xc = V @ E[:3, :3].T+E[:3, 3]; uv = Xc[:, :2]/Xc[:, 2:]*np.array([K[0, 0], K[1, 1]])+K[:2, 2]
        H, W = und.shape[:2]
        mask = np.zeros((H, W), np.uint8)
        cv2.fillPoly(mask, [t for t in np.round(uv[F]).astype(np.int32)], 255)
        ys, xs = np.nonzero(mask)
        if len(xs) == 0:
            meta.append(dict(cam=name, skipped='scan not visible')); continue
        cx, cy = 0.5*(xs.min()+xs.max()), 0.5*(ys.min()+ys.max()); side = 1.25*max(xs.max()-xs.min(), ys.max()-ys.min())
        x0, y0 = cx-side/2, cy-side/2; sc = S/side
        A = np.array([[sc, 0, -x0*sc], [0, sc, -y0*sc]])
        im = cv2.warpAffine(und, A, (S, S), flags=cv2.INTER_AREA if sc < 1 else cv2.INTER_LINEAR, borderValue=(255, 255, 255))
        mk = cv2.warpAffine(mask, A, (S, S), flags=cv2.INTER_NEAREST, borderValue=0)
        mk = cv2.dilate(mk, np.ones((9, 9), np.uint8)) > 0
        # chroma key against the green studio (PokeFlex scenes are shot in a green-screen room): drop green pixels
        # inside the dilated silhouette unless that would remove most of the object (green objects)
        I = im.astype(np.int32); green = (I[..., 1] > I[..., 0]+25) & (I[..., 1] > I[..., 2]+15)
        keyed = mk & ~green
        chroma = bool(keyed.sum() > 0.5*mk.sum())
        mk = keyed if chroma else mk
        im = im.copy(); im[~mk] = 255
        Kp = np.array([[K[0, 0]*sc, 0, (K[0, 2]-x0)*sc], [0, K[1, 1]*sc, (K[1, 2]-y0)*sc], [0, 0, 1]])
        c2w = np.linalg.inv(E)[:3, :4]
        o, d, v = get_rays_of_a_view(S, S, torch.tensor(Kp, dtype=torch.float32), torch.tensor(c2w, dtype=torch.float32), inverse_y=True)
        ro.append(o.reshape(-1, 3)); rd.append(d.reshape(-1, 3)); vd.append(v.reshape(-1, 3))
        rgb.append(torch.from_numpy(im.reshape(1, -1, 3).astype(np.float32)/255.))
        cv2.imwrite(str(out/f'view_{rig}{ci}.jpg'), im[..., ::-1])
        meta.append(dict(cam=name, K=Kp.tolist(), c2w=c2w.tolist(), crop=[float(x0), float(y0), float(side)], fg_frac=float(mk.mean()), chroma_key=chroma))
    torch.save(dict(rays_o=torch.stack(ro), rays_d=torch.stack(rd), viewdirs=torch.stack(vd), rgb_all=torch.stack(rgb),
                    ray_mask_all=torch.ones(len(ro), S*S)), out/'data.pt')
    (out/'meta.json').write_text(json.dumps(dict(obj=obj, take=take, frame=int(frame), S=S, views=meta), indent=1))
    return sum('K' in m for m in meta)


def main(objs):
    for obj in objs:
        info = obj_info(obj)
        for kind, lst in (('cal', info['calibration_pokes']), ('ho', info['heldout'])):
            for t, k in lst:
                p = load_poke(obj, kind, t, k)
                out = WORK/'pacnerf'/obj/f'{kind}_{t}-p{k}'
                if (out/'data.pt').exists():
                    continue
                nv = build(obj, t, int(p['start_frame']), p['V'], p['F'], out)
                print(obj, kind, t, k, 'views', nv, flush=True)


if __name__ == '__main__':
    if sys.argv[1:] == ['--dev']:
        p = dict(np.load(INP/'_dev/cal_FoamDice_T1-p1.npz'))
        print('dev views', build('FoamDice', 'FoamDice_T1', int(p['start_frame']), p['V'], p['F'], WORK/'pacnerf/_dev/cal_FoamDice_T1-p1'))
    else:
        main(sys.argv[1:] or OBJECTS)
