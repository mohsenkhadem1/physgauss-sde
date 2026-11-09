"""Top-down renders of the predicted stem shape under the 20 N push (D2) at peak bending.
Truth body = black outline; rest pose = dashed gray outline; GIC = its predicted particles (orange);
ours = our predicted deformed mesh (blue). Run with pacnerf-env (torch + CUDA)."""
import json, sys
from pathlib import Path
import numpy as np
import torch
from scipy import ndimage
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0, str(Path(__file__).resolve().parent))
import fem, render

R = Path(__file__).resolve().parents[2]
T = R/'runs/stem-trial-v01'; OUT = R/'output/paper-figures'
S1 = json.loads((R/'reports/stem-trial-v01/scores.json').read_text())['rows']
OURS, GIC, INK, MUTED = '#2a78d6', '#eb6834', '#0b0b0b', '#8a8984'
W = ['W500', 'W1000', 'W2000']; RHO = {'W500': 500, 'W1000': 1000, 'W2000': 2000}

target = np.array([-0.02, 0.60, 0.0])
c2w = render.look_at(target+np.array([0, 0.75, 0]), target, up=(0, 0, -1))  # image right = +x, image down = +z
K = np.array([[1000., 0, 400], [0, 1000., 400], [0, 0, 1]])
mesh = fem.Mesh((33, 6, 6)); tris = mesh.surface_triangles()


def mesh_img(X):
    img, a = render.render(c2w, K, X, mesh.X, tris)
    return img, a > .5


def particle_img(P):
    """Disc splats with a z-buffer (nearest particle wins), Lambert-like shading by depth."""
    uv, d = render.project(c2w, K, P); H = Wd = 800; r = 6
    zb = np.full((H, Wd), np.inf); dep = np.zeros((H, Wd))
    oy, ox = np.mgrid[-r:r+1, -r:r+1]; disk = (ox**2+oy**2) <= r*r
    for (u, v), z in zip(uv, d):
        ys, xs = int(v)+oy[disk], int(u)+ox[disk]
        ok = (ys >= 0) & (ys < H) & (xs >= 0) & (xs < Wd); ys, xs = ys[ok], xs[ok]
        closer = z < zb[ys, xs]; zb[ys[closer], xs[closer]] = z; dep[ys[closer], xs[closer]] = z
    m = np.isfinite(zb)
    return m, dep


def edge(mask):
    return mask & ~ndimage.binary_erosion(mask, iterations=2)


def tint(mask, shade, color):
    rgb = np.array(matplotlib.colors.to_rgb(color))
    out = np.ones(mask.shape+(3,))
    out[mask] = (0.55+0.45*shade[mask, None])*rgb+(0.45-0.45*shade[mask, None])*0
    return out


def rms(method, evidence, w):
    return next(r['rms_mm'] for r in S1 if r['world'] == w and r['method'] == method and r['evidence'] == evidence)


def main(mag=1.0):
    rest_edge = edge(mesh_img(mesh.X)[1])
    cols = [('GIC', 'passive video only', 'D0 only'), ('ours', 'passive video only', 'D0 only'),
            ('GIC', 'after one 40 N poke', 'after D1 poke refit'), ('ours', 'after one 40 N poke', 'D0 + D1 poke')]
    fig, axes = plt.subplots(3, 4, figsize=(15, 6.6), gridspec_kw=dict(wspace=.03, hspace=.18))
    crop = (slice(260, 560), slice(80, 740)) if mag == 1 else (slice(300, 690), slice(80, 770))
    for i, w in enumerate(W):
        z = np.load(T/f'truth/D2-{w}.npz'); U = z['U']
        tip = U[:, mesh.X[:, 0] > .199, 2].mean(1); f = int(np.argmax(tip))  # frame of peak lateral bending
        truth_edge = edge(mesh_img(mesh.X+mag*U[f])[1])
        ours = np.load(T/f'ours-fields/tracks-{w}-D2.npz')
        for j, (meth, stage, ev) in enumerate(cols):
            ax = axes[i, j]
            if meth == 'GIC':
                P = np.load(T/('gic/forecast-D0.npz' if ev == 'D0 only' else f'gic/forecast-D1-{w}.npz'))['xyz']; p = P[0]+mag*(P[f]-P[0])
                m, dep = particle_img(p.astype(float)); sh = np.zeros_like(dep); sh[m] = 1-(dep[m]-dep[m].min())/max(np.ptp(dep[m]), 1e-9)
                img = tint(m, sh, GIC); err = rms('GIC', ev, w)
            else:
                Uo = ours['D0' if ev == 'D0 only' else 'D0pD1'][f]
                rgb, m = mesh_img(mesh.X+mag*Uo); sh = rgb.mean(-1)/rgb[m].mean(-1).max()
                img = tint(m, sh, OURS); err = rms('ours posterior mean', ev, w)
            img[rest_edge] = matplotlib.colors.to_rgb('#c9c8c3')
            img[truth_edge] = 0
            ax.imshow(img[crop]); ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(False)
            ax.text(.99, .97, f'{err:.1f} mm', transform=ax.transAxes, ha='right', va='top', fontsize=10, weight='bold',
                    color=INK if err > 3 else '#1f7a3a')
            if i == 0:
                ax.set_title(f'{"GIC" if meth == "GIC" else "Ours"} — {stage}', fontsize=10.5, weight='bold',
                             color=GIC if meth == 'GIC' else OURS)
            if j == 0:
                ax.set_ylabel(f'ρ = {RHO[w]} kg/m³\npeak t = {f/48:.2f} s', fontsize=10)
    fig.text(.5, .955, 'Predicted stem shape under the held-out 20 N push at 0.75 L (top view, push direction ↓)'+('' if mag == 1 else f'  —  displacements ×{mag:g}'),
             ha='center', fontsize=12, weight='bold')
    fig.text(.5, .03, 'black outline: true shape    gray: rest shape    orange: GIC predicted particles    blue: our predicted mesh'
             '    number: 3D error over the whole push (RMS of 4 material stations), green = within 3 mm',
             ha='center', fontsize=9, color='#52514e')
    name = 'fig4-predicted-shapes' if mag == 1 else f'fig4b-predicted-shapes-x{mag:g}'
    fig.savefig(OUT/f'{name}.png', dpi=170, bbox_inches='tight'); fig.savefig(OUT/f'{name}.pdf', bbox_inches='tight')
    print(OUT/f'{name}.png')


if __name__ == '__main__':
    main(1.0); main(3.0)
