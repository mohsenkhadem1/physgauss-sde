"""PokeFlex comparison videos (LOCAL ONLY: derived from licensed PokeFlex data, never commit or share).

One video per object: its median-error held-out poke (by ours force RMSE in scores.json).  Panels: measured mesh, then
ours / fem_lsq (v01, sealed) and PhysTwin / PAC-NeRF (stage B, manifest-checked).  Each panel is a side view
(horizontal = in-plane axis perpendicular to the approach axis, vertical = -approach) of surface points, coloured by
distance to the measured mesh at that frame (mm); grey underlay = measured surface; black bar = tool.  Bottom: force
along -approach, measured vs each method, with a frame cursor.
usage: python -I videos.py [OBJ ...] [--all] [--fps 8]   -> runs/pokeflex-v01/videos/<obj>_<take>-p<k>.mp4"""
import json, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import animation
import imageio_ffmpeg

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent/'pokeflex_v01')); sys.path.insert(0, str(HERE))
import score_stageb as SB  # verifies the stage-B manifest and installs its sealed check for stage-B files
import pokeflex as P
from pfx_common import OBJECTS, PRED, approach_axis

plt.rcParams['animation.ffmpeg_path'] = imageio_ffmpeg.get_ffmpeg_exe()
OUTD = P.OUT/'videos'; OUTD.mkdir(parents=True, exist_ok=True)
args = sys.argv[1:]
FPS = float(args[args.index('--fps')+1]) if '--fps' in args else 8.0
OBJS = [a for a in args if a in OBJECTS] or OBJECTS
V01_SEALED = set(P.verify()['files'])
METHODS = [('ours', PRED/'ours', '#2a6fdb'), ('fem_lsq', PRED/'fem_lsq', '#7a7a7a'),
           ('PhysTwin', SB.SB/'PhysTwin', '#d9822b'), ('PAC-NeRF', SB.SB/'PAC-NeRF', '#3fa34d')]
NPTS = 5000


def sealed(f):
    return Path(f).resolve().relative_to(P.R).as_posix() in V01_SEALED or str(Path(f).resolve()) in SB.SEALED


def load_pred(d, obj, take, k):
    f = d/obj/f'{take}-p{k}.npz'
    if not f.exists() or not sealed(f):
        return None
    z = np.load(f)
    return {k_: z[k_] for k_ in z.files}


def pick_pokes(obj):
    S = json.loads((P.REP/'scores.json').read_text())['rows']
    rr = sorted([r for r in S if r['method'] == 'ours' and r['object'] == obj], key=lambda r: r['force_rmse_N'])
    return [(r['take'], r['poke']) for r in rr] if '--all' in args else [(rr[len(rr)//2]['take'], rr[len(rr)//2]['poke'])]


def frame_basis(a):
    up = -a/np.linalg.norm(a)
    h = np.cross(up, [0, 0, 1.0]) if abs(up[2]) < 0.9 else np.cross(up, [1.0, 0, 0])
    return h/np.linalg.norm(h), up


def make(obj, take, k):
    a = approach_axis(obj, take); h, up = frame_basis(a)
    preds = [(n, load_pred(d, obj, take, k), c) for n, d, c in METHODS]
    ref = next(p for _, p, _ in preds if p is not None)
    fr = ref['frames']
    Fm = (P.forces(obj, take, fr)[:, :3]-P.force_offset(obj, take)[:3]) @ -a
    T = P.robot_trajectory(obj, take)['T_WT']
    meas = [P.mesh(obj, take, int(f)) for f in fr]
    proj = lambda X: np.stack([X @ h, X @ up], 1)*1e3
    Vr = meas[0][0]; c0 = proj(Vr).mean(0)
    allp = proj(Vr)-c0; lim = np.abs(allp).max()*1.25
    panels = [('measured', None, 'k')]+[(n, p, c) for n, p, c in preds]
    fig = plt.figure(figsize=(3.0*len(panels), 6.2), dpi=110)
    gs = fig.add_gridspec(2, len(panels), height_ratios=[2.2, 1])
    axs = [fig.add_subplot(gs[0, i]) for i in range(len(panels))]
    axf = fig.add_subplot(gs[1, :])
    rng = np.random.default_rng(0)
    sub = {}
    for n, p, _ in preds:
        if p is not None:
            sub[n] = rng.choice(p['surf_V'].shape[1], min(NPTS, p['surf_V'].shape[1]), replace=False)
    axf.plot(range(len(fr)), Fm, 'k-', lw=2, label='measured')
    for n, p, c in preds:
        if p is not None:
            axf.plot(range(len(fr)), p['force'] @ -a, '-', color=c, lw=1.5, label=n)
    cur = axf.axvline(0, color='0.5', lw=1)
    axf.set_xlabel('frame in poke'); axf.set_ylabel('force along -approach (N)'); axf.legend(fontsize=8, ncol=5, loc='upper left')
    axf.grid(alpha=0.3)
    sm = plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(0, 15))
    fig.colorbar(sm, ax=axs, fraction=0.02, pad=0.01, label='distance to measured (mm)')
    fig.suptitle(f'{take} poke {k} (held-out) - PokeFlex, local use only', fontsize=10)

    def draw(j):
        Vm, Fmesh, _ = meas[j]
        Q = Vm[rng.choice(len(Vm), min(NPTS, len(Vm)), replace=False)]
        tip = T[fr[j]-1][:3, 3]; tp = proj(tip[None])[0]-c0
        for ax, (n, p, c) in zip(axs, panels):
            ax.cla(); ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_aspect('equal'); ax.set_xticks([]); ax.set_yticks([])
            ax.scatter(*(proj(Q)-c0).T, s=0.6, c='0.82', lw=0)
            if n == 'measured':
                ax.scatter(*(proj(Q)-c0).T, s=0.8, c='k', lw=0); title = 'measured'
            elif p is None:
                title = f'{n}\n(no prediction)'
            else:
                X = p['surf_V'][j][sub[n]].astype(float)
                d = np.sqrt(((X[:, None, :]-Q[None, ::4, :])**2).sum(-1).min(1))*1e3
                ax.scatter(*(proj(X)-c0).T, s=0.8, c=d, cmap='viridis', vmin=0, vmax=15, lw=0)
                e = abs(p['force'][j] @ -a-Fm[j])
                title = f'{n}\n|dF| {e:.1f} N'
            ax.plot([tp[0]-10, tp[0]+10], [tp[1], tp[1]], 'k-', lw=3)
            ax.plot([tp[0], tp[0]], [tp[1], tp[1]+lim], 'k-', lw=1)
            ax.set_title(title, fontsize=9, color=c if n != 'measured' else 'k')
        cur.set_xdata([j, j])
        return []

    out = OUTD/f'{take}-p{k}.mp4'
    animation.FuncAnimation(fig, draw, frames=len(fr), blit=False).save(out, fps=FPS, dpi=110)
    plt.close(fig)
    print(out)


if __name__ == '__main__':
    for o in OBJS:
        for t, k in pick_pokes(o):
            make(o, t, k)
