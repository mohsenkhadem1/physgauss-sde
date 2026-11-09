"""Independent multi-view renderer for the stem trial (PAC-NeRF dataset format).

Exact triangle rasterization with a z-buffer on the GPU, 2x2 supersampling,
procedural texture evaluated at each pixel's reference-configuration point,
Lambert shading, white background. Cameras use the Blender/PAC-NeRF c2w
convention (camera looks along -z, y up) with pinhole intrinsics.
This renderer shares no code with GIC, PAC-NeRF or 3D Gaussian splatting.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import torch
from PIL import Image

W = H = 800
FOCAL = 1000.0
# Cameras must aim at a point on the world y-axis: GIC's PAC-NeRF reader (b523851) uses T = -t,
# negating the x component of the world->camera translation. That is harmless only when every
# camera's translation has t_x = 0, which holds for PAC-NeRF's origin-facing cameras and for
# look-at targets with zero x and z. Checked by gic_camera_check() below.
TARGET = np.array([0.0, 0.585, 0.0])


def look_at(pos, target=TARGET, up=(0, 1, 0)):
    z = pos-target; z = z/np.linalg.norm(z)
    x = np.cross(up, z); x /= np.linalg.norm(x); y = np.cross(z, x)
    c2w = np.eye(4); c2w[:3, 0], c2w[:3, 1], c2w[:3, 2], c2w[:3, 3] = x, y, z, pos
    return c2w


def cameras(radius=1.15):
    """11 views; camera 0 is the held-out view. Azimuth measured from +z toward +x."""
    spec = [(35, 22), (0, 15), (60, 30), (-45, 25), (110, 35), (-100, 30), (160, 40),
            (-160, 25), (20, 55), (-20, 60), (90, 5)]
    out = []
    for az, el in spec:
        a, e = np.radians(az), np.radians(el)
        pos = TARGET+radius*np.array([np.sin(a)*np.cos(e), np.sin(e), np.cos(a)*np.cos(e)])
        out.append(look_at(pos))
    K = np.array([[FOCAL, 0, W/2], [0, FOCAL, H/2], [0, 0, 1.]])
    return out, K


def gic_camera_check():
    """Max deviation between GIC's reader world->camera and the correct one over all cameras."""
    cams, _ = cameras(); dev = 0.0
    for c2w in cams:
        w2c = np.linalg.inv(c2w)
        R = -w2c[:3, :3].T.copy(); R[:, 0] = -R[:, 0]; T = -w2c[:3, 3]
        G = np.eye(4); G[:3, :3] = R.T; G[:3, 3] = T
        dev = max(dev, float(np.abs(G-np.diag([1, -1, -1, 1])@w2c).max()))
    return dev


def project(c2w, K, X):
    """World points (N,3) -> pixel (u,v) and positive depth; Blender convention."""
    w2c = np.linalg.inv(c2w)
    pc = X@w2c[:3, :3].T+w2c[:3, 3]
    d = -pc[:, 2]
    u = K[0, 0]*pc[:, 0]/d+K[0, 2]; v = -K[1, 1]*pc[:, 1]/d+K[1, 2]
    return np.stack([u, v], 1), d


def texture(Xr):
    """Deterministic non-white procedural colours from reference coordinates (torch)."""
    x, y, z = Xr[:, 0], Xr[:, 1], Xr[:, 2]
    s = (x+.24)/.44
    r = .45+.25*torch.sin(2*np.pi*(x/.045))*torch.cos(2*np.pi*(z/.05+y/.06))
    g = .40+.25*torch.sin(2*np.pi*(y/.035+.3))*torch.sin(2*np.pi*(x/.07+z/.03))
    b = .35+.25*torch.cos(2*np.pi*(z/.04-x/.09))+.15*s
    band = (torch.remainder(torch.floor(x/.02), 2) == 0).float()
    c = torch.stack([r, g, b], 1)*(0.75+0.25*band[:, None])
    return c.clamp(.05, .88)


@torch.no_grad()
def render(c2w, K, verts, verts_ref, tris, ss=2, device='cuda', return_depth=False):
    """verts (n,3) deformed world positions; verts_ref reference positions; tris (T,3)."""
    Ws, Hs = W*ss, H*ss
    Kss = K.copy(); Kss[:2] *= ss
    uv, d = project(c2w, Kss, verts)
    uv = torch.tensor(uv, device=device); d = torch.tensor(d, device=device)
    T = torch.tensor(tris, device=device)
    p = uv[T]; dz = d[T]  # (T,3,2),(T,3)
    lo = torch.floor(p.min(1).values).long(); hi = torch.ceil(p.max(1).values).long()
    win = int((hi-lo).max().item())+1
    assert win <= 64, win
    oy, ox = torch.meshgrid(torch.arange(win, device=device), torch.arange(win, device=device), indexing='ij')
    px = lo[:, None, 0]+ox.reshape(1, -1); py = lo[:, None, 1]+oy.reshape(1, -1)
    cx = px.double()+.5; cy = py.double()+.5
    a, b, c = p[:, 0], p[:, 1], p[:, 2]
    den = (b[:, 1]-c[:, 1])*(a[:, 0]-c[:, 0])+(c[:, 0]-b[:, 0])*(a[:, 1]-c[:, 1])
    l0 = ((b[:, None, 1]-c[:, None, 1])*(cx-c[:, None, 0])+(c[:, None, 0]-b[:, None, 0])*(cy-c[:, None, 1]))/den[:, None]
    l1 = ((c[:, None, 1]-a[:, None, 1])*(cx-c[:, None, 0])+(a[:, None, 0]-c[:, None, 0])*(cy-c[:, None, 1]))/den[:, None]
    l2 = 1-l0-l1
    inside = (l0 >= 0) & (l1 >= 0) & (l2 >= 0) & (px >= 0) & (px < Ws) & (py >= 0) & (py < Hs) & (den.abs()[:, None] > 1e-12)
    # perspective-correct depth: 1/z interpolates linearly in screen space
    inv = l0/dz[:, None, 0]+l1/dz[:, None, 1]+l2/dz[:, None, 2]
    depth = 1/inv
    tid = torch.arange(len(T), device=device)[:, None].expand_as(px)
    sel = inside.nonzero(as_tuple=True)
    pix = (py[sel]*Ws+px[sel]); dep = depth[sel]
    zbuf = torch.full((Ws*Hs,), float('inf'), device=device, dtype=dep.dtype)
    zbuf.scatter_reduce_(0, pix, dep, reduce='amin')
    win_mask = dep <= zbuf[pix]
    pix = pix[win_mask]; t = tid[sel][win_mask]
    L = torch.stack([l0[sel][win_mask], l1[sel][win_mask], l2[sel][win_mask]], 1)
    # perspective-correct barycentrics
    Lp = L/dz[t]; Lp = Lp/Lp.sum(1, keepdim=True)
    Vr = torch.tensor(verts_ref, device=device); Vd = torch.tensor(verts, device=device)
    Xr = (Lp[:, :, None]*Vr[T[t]]).sum(1)
    tri_w = Vd[T[t]]; nrm = torch.cross(tri_w[:, 1]-tri_w[:, 0], tri_w[:, 2]-tri_w[:, 0], dim=1)
    nrm = nrm/nrm.norm(dim=1, keepdim=True)
    Xw = (Lp[:, :, None]*tri_w).sum(1)
    cam = torch.tensor(c2w[:3, 3], device=device)
    view = cam[None]-Xw; view = view/view.norm(dim=1, keepdim=True)
    light = torch.tensor([.3, .9, .4], device=device, dtype=view.dtype); light = light/light.norm()
    shade = .45+.35*(nrm*view).sum(1).abs()+.2*(nrm*light).sum(1).abs()
    col = texture(Xr)*shade[:, None]
    img = torch.ones(Ws*Hs, 3, device=device, dtype=col.dtype)
    alpha = torch.zeros(Ws*Hs, device=device, dtype=col.dtype)
    img[pix] = col.clamp(0, .9); alpha[pix] = 1
    img = img.reshape(Hs, Ws, 3).reshape(H, ss, W, ss, 3).mean((1, 3))
    alpha = alpha.reshape(Hs, Ws).reshape(H, ss, W, ss).mean((1, 3))
    if return_depth:
        # camera-space depth (m) along the optical axis; mean of covered subpixels, 0 = background
        zb = zbuf.reshape(Hs, Ws); cov = torch.isfinite(zb)
        zs = torch.where(cov, zb, torch.zeros_like(zb)).reshape(H, ss, W, ss).sum((1, 3))
        n = cov.reshape(H, ss, W, ss).sum((1, 3))
        depth = torch.where(n > 0, zs/n.clamp(min=1), torch.zeros_like(zs))
        return img.cpu().numpy(), alpha.cpu().numpy(), depth.cpu().numpy()
    return img.cpu().numpy(), alpha.cpu().numpy()


def write_scene(folder, frames_xyz, ref_xyz, tris, fps, cams=None, K=None):
    """frames_xyz (F,n,3) node positions. Writes PAC-NeRF all_data.json, r_/m_ images."""
    folder = Path(folder); (folder/'data').mkdir(parents=True, exist_ok=True)
    if cams is None:
        cams, K = cameras()
    entries = []
    blank = np.full((H, W, 3), 255, np.uint8)
    for c, c2w in enumerate(cams):
        for f in [-1]+list(range(len(frames_xyz))):
            name = f'r_{c}_{f}.png'
            if f < 0:
                im = blank
            else:
                img, _ = render(c2w, K, frames_xyz[f], ref_xyz, tris)
                im = np.round(img*255).astype(np.uint8)
            Image.fromarray(im).save(folder/'data'/name)
            Image.fromarray(im).save(folder/'data'/name.replace('r_', 'm_'))
            entries.append(dict(file_path=f'./data/{name}', time=f/fps, c2w=c2w[:3].tolist(), intrinsic=K.tolist()))
    (folder/'all_data.json').write_text(json.dumps(entries))
    return entries


def unproject(c2w, K, u, v, depth):
    """Pixel (u,v) with camera-space depth -> world point (Blender convention)."""
    pc = np.stack([(u-K[0, 2])*depth/K[0, 0], -(v-K[1, 2])*depth/K[1, 1], -depth], -1)
    return pc@c2w[:3, :3].T+c2w[:3, 3]
