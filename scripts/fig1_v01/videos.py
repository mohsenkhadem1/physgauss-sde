"""Figure-1 videos: the held-out interaction over time, reference + each method after calibration, side by side.
Same inputs, view, colouring and palette as figure1.py (predictions read only through the sealed manifests): particles
coloured by motion error |u_method - u_true| (mm, nearest reference node to the rest position), grey underlay =
reference shape at that frame, striker circle for S3 (dashed = reference, solid = method's own striker).
usage: python videos.py [S1|S2|S3 ...] [--world W] [--fps 12]  -> output/paper-figures/videos/fig1-<scen>-<world>.mp4"""
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import animation
import imageio_ffmpeg
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import figure1 as F

plt.rcParams['animation.ffmpeg_path'] = imageio_ffmpeg.get_ffmpeg_exe()
VID = F.OUT/'videos'; VID.mkdir(parents=True, exist_ok=True)
args = sys.argv[1:]
WORLD = args[args.index('--world')+1] if '--world' in args else None
FPS_OUT = float(args[args.index('--fps')+1]) if '--fps' in args else 12.0
SCENS = [a for a in args if a in F.SCEN] or list(F.SCEN)
NPTS = 6000


def make(scen, w):
    cfg = F.SCEN[scen]; a, b = cfg['plane']
    X, U, _, sphere = F.truth(scen, w)
    n = len(U)
    shapes = {m: F.shape(scen, m, w) for m in F.METHODS}
    rng = np.random.default_rng(0)
    sel = {m: rng.permutation(len(r))[:NPTS] for m, (r, p, _) in shapes.items()}
    errs = {m: np.stack([F.motion_error(r[sel[m]], p[:, sel[m]], k, X, U) for k in range(n)]) for m, (r, p, _) in shapes.items()}
    vmax = float(np.ceil(np.percentile(np.concatenate([e.ravel() for e in errs.values()]), 95)))
    norm = F.Normalize(0, vmax)
    # fixed view: reference over all frames (+ striker while near the object)
    pts = [(X[None]+U)[:, :, [a, b]].reshape(-1, 2)]
    if sphere is not None:
        near = np.array([np.linalg.norm(X+U[k]-sphere[k], axis=1).min() < .1 for k in range(n)])
        pts += [sphere[near][:, [a, b]]+.03, sphere[near][:, [a, b]]-.03]
    for m, (r, p, _) in shapes.items():  # include each method's own reconstruction (robust 0.5-99.5 % extent)
        q = p[:, sel[m]][:, :, [a, b]].reshape(-1, 2); pts.append(np.percentile(q, [.5, 99.5], axis=0))
    pts = np.vstack(pts); lo, hi = pts.min(0), pts.max(0); pad = .04*(hi-lo).max()
    lo, hi = lo-pad, hi+pad; asp = (hi[1]-lo[1])/(hi[0]-lo[0])
    ncol = 3 if asp > .6 else 2; nrow = 6//ncol
    pw = 4.6 if ncol == 3 else 6.2; ph = pw*asp+.45; top_in, cb_in = .95, .9
    strip = 2.5; W = ncol*pw+cb_in+.3; H = nrow*ph+top_in+.15+strip
    W, H = 2*round(W*75)/150, 2*round(H*75)/150  # even pixel size at 150 dpi
    fig = plt.figure(figsize=(W, H), dpi=150)
    gs = fig.add_gridspec(nrow, ncol+1, width_ratios=[1]*ncol+[.04], wspace=.05, hspace=.45/ph*1.6,
                          left=.15/W, right=1-.75/W, top=1-top_in/H, bottom=(strip+.1)/H)
    names = ['reference']+F.METHODS
    axes = [fig.add_subplot(gs[j//ncol, j % ncol]) for j in range(6)]
    under, dots, circ_ref, circ_m, meds = [], [], [], [], []
    for j, (ax, m) in enumerate(zip(axes, names)):
        ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1]); ax.set_aspect('equal')
        ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.set_title(F.LABEL.get(m, 'reference (truth)'), fontsize=10, color=F.INK, loc='left', pad=5)
        ax.patch.set_alpha(0)
        under.append(ax.scatter([], [], s=1.2, c=F.UNDER if m != 'reference' else F.INK2, lw=0))
        if m != 'reference':
            ax.plot([0, 1], [1.0, 1.0], transform=ax.transAxes, color=F.COL[m], lw=3, solid_capstyle='butt', clip_on=False)
            dots.append(ax.scatter([], [], s=1.6, c=[], cmap=F.ERR_CMAP, norm=norm, lw=0))
            meds.append(ax.text(1, 1.035, '', transform=ax.transAxes, fontsize=8.5, color=F.INK2, ha='right', va='bottom'))
            circ_m.append(ax.add_patch(plt.Circle((0, 0), .03, fill=False, ec=F.INK, lw=1.1, visible=False)))
        circ_ref.append(ax.add_patch(plt.Circle((0, 0), .03, fill=False, ec=F.MUTED, lw=1, ls='--', visible=False)))
    cax = fig.add_subplot(gs[:, ncol].subgridspec(3, 1, height_ratios=[.2, .6, .2])[1])
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=F.ERR_CMAP), cax=cax); cb.outline.set_visible(False)
    cax.set_title('motion\nerror (mm)', fontsize=8.5, color=F.INK2, loc='center', pad=6)
    fig.text(.15/W, 1-.38/H, cfg['title'], fontsize=13, fontweight='bold', color=F.INK)
    # error-over-time strip: RMS error of the scored stations/probes (exactly the scored quantity), this world
    tg = F.truth(scen, w)[2]
    sx = fig.add_axes([.75/W, .5/H, 1-1.75/W, (strip-1.0)/H])
    for m in F.METHODS:
        e = np.sqrt(((F.trajectory(scen, m, w, 'D1')-tg)**2).sum(-1).mean(-1))*1e3
        sx.plot(np.arange(len(e))/F.FPS, np.maximum(e, 1e-2), color=F.COL[m], lw=2.2 if m == 'ours' else 1.3, label=F.LABEL[m],
                zorder=3 if m == 'ours' else 2)
    sx.set_yscale('log'); sx.set_xlim(0, (n-1)/F.FPS); sx.set_ylim(.05, 40); sx.set_xlabel('time (s)', fontsize=8.5)
    sx.set_ylabel('error (mm)', fontsize=8.5); sx.tick_params(labelsize=8)
    sx.legend(ncol=5, fontsize=8, loc='lower right', bbox_to_anchor=(1, 1.0), handlelength=1.5, columnspacing=1.2)
    cursor = sx.axvline(0, color=F.INK, lw=1)
    clock = fig.text(.15/W, 1-.68/H, '', fontsize=9.5, color=F.INK2)
    view = 'side view (x-y)' if b == 1 else 'top view (x-z)'

    def draw(k):
        ref = X+U[k]
        sph_on = sphere is not None and np.linalg.norm(ref-sphere[k], axis=1).min() < .1
        for j, m in enumerate(names):
            under[j].set_offsets(ref[::2][:, [a, b]])
            circ_ref[j].set_visible(bool(sph_on))
            if sph_on:
                circ_ref[j].center = (sphere[k][a], sphere[k][b])
            if m == 'reference':
                continue
            i = j-1; r, p, sc = shapes[m]; e = errs[m][k]; o = np.argsort(e)
            dots[i].set_offsets(p[k][sel[m]][o][:, [a, b]]); dots[i].set_array(e[o])
            meds[i].set_text(f'median {np.median(e):.1f} mm')
            show = sc is not None and sph_on
            circ_m[i].set_visible(bool(show))
            if show:
                circ_m[i].center = (sc[k][a], sc[k][b])
        cursor.set_xdata([k/F.FPS]*2)
        clock.set_text(f'world {w}   ·   t = {k/F.FPS:.3f} s  (frame {k+1}/{n})   ·   {view}   ·   playback {FPS_OUT/F.FPS:.2g}x real time   ·   after one calibration interaction')
        return []

    frames = list(range(n))+[n-1]*int(FPS_OUT)  # hold the last frame ~1 s before looping
    anim = animation.FuncAnimation(fig, draw, frames=frames, blit=False)
    out = VID/f'fig1-{scen}-{w}.mp4'
    anim.save(out, writer=animation.FFMpegWriter(fps=FPS_OUT, codec='libx264', bitrate=4000, extra_args=['-pix_fmt', 'yuv420p']))
    plt.close(fig)
    print(out, f'{len(frames)/FPS_OUT:.1f} s')


if __name__ == '__main__':
    for s in SCENS:
        for w in ([WORLD] if WORLD else F.SCEN[s]['worlds']):
            make(s, w)
