"""Compose the Blender panel renders (blender_render.py) into one labelled video per scene.
usage: python blender_compose.py S3 W1000 [--fps 12]  -> output/paper-figures/blender/fig1-blender-<scen>-<world>.mp4"""
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.colors import Normalize
import imageio_ffmpeg
HERE = Path(__file__).resolve().parent; R = HERE.parents[1]
sys.path.insert(0, str(HERE))
from figure1 import ERR_CMAP, INK, INK2, MUTED, GRID, FPS

plt.rcParams['animation.ffmpeg_path'] = imageio_ffmpeg.get_ffmpeg_exe()
scen, w = sys.argv[1], sys.argv[2]
FPS_OUT = float(sys.argv[sys.argv.index('--fps')+1]) if '--fps' in sys.argv else 12.0
Z = np.load(R/f'runs/fig1-v01/blender/{scen}-{w}.npz')
FR = R/f'runs/fig1-v01/blender/frames/{scen}-{w}'
OUT = R/'output/paper-figures/blender'; OUT.mkdir(parents=True, exist_ok=True)
n = int(Z['n']); labels = ['reference (truth)']+list(Z['labels']); colors = [None]+list(Z['colors'])
plt.rcParams.update({'font.family': 'DejaVu Sans', 'text.color': INK, 'axes.edgecolor': MUTED, 'xtick.color': INK2, 'ytick.color': INK2})

fig = plt.figure(figsize=(19.2, 10.8), dpi=100, facecolor='white')
gs = fig.add_gridspec(2, 4, width_ratios=[1, 1, 1, .035], left=.012, right=.955, top=.885, bottom=.235, wspace=.025, hspace=.13)
ims, meds = [], []
for j in range(6):
    ax = fig.add_subplot(gs[j//3, j % 3]); ax.set_axis_off()
    ims.append(ax.imshow(plt.imread(FR/f'p{j}/000.png')))
    ax.set_title(labels[j], loc='left', fontsize=15, color=INK, pad=9)
    if colors[j] is not None:
        ax.plot([0, 1], [1.01, 1.01], transform=ax.transAxes, color=colors[j], lw=5, solid_capstyle='butt', clip_on=False)
        meds.append(ax.text(1, 1.035, '', transform=ax.transAxes, ha='right', va='bottom', fontsize=12.5, color=INK2))
cax = fig.add_subplot(gs[:, 3].subgridspec(3, 1, height_ratios=[.2, .6, .2])[1])
cb = fig.colorbar(plt.cm.ScalarMappable(norm=Normalize(0, float(Z['vmax'])), cmap=ERR_CMAP), cax=cax); cb.outline.set_visible(False)
cax.set_title('motion\nerror (mm)', fontsize=12, color=INK2, pad=8); cax.tick_params(labelsize=11)
fig.text(.012, .955, str(Z['title']), fontsize=22, fontweight='bold')
clock = fig.text(.012, .918, '', fontsize=13.5, color=INK2)
fig.text(.012, .012, 'Each method: surface built the same way from its own predicted points, coloured by motion error.  ' +
         ('Grey ghost: true surface.  Solid ball: method\'s striker; translucent ball: true striker.' if scen == 'S3' else
          'Grey ghost: true surface.  Dark arrow: the applied force (known command), length following its ramp.'),
         fontsize=11.5, color=INK2, ha='left')
if scen == 'S3':
    fig.text(.012, .034, 'PAC-NeRF: its reconstructed stem is ~2x too thick, so its striker starts inside the body and is ejected at t = 0 (report caveat 4.5).',
             fontsize=11.5, color=INK2, ha='left')
sx = fig.add_axes([.055, .1, .9, .1])
for i, (lab, c) in enumerate(zip(Z['labels'], Z['colors'])):
    e = Z[f'm{i}_rms']; sx.plot(np.arange(len(e))/FPS, np.maximum(e, 1e-2), color=c, lw=3 if lab.startswith('ours') else 1.8, label=lab)
sx.set_yscale('log'); sx.set_xlim(0, (n-1)/FPS); sx.set_ylim(.05, 40); sx.grid(True, color=GRID, lw=.8)
for s in ('top', 'right'):
    sx.spines[s].set_visible(False)
sx.set_xlabel('time (s)', fontsize=12, color=INK2); sx.set_ylabel('scored error (mm)', fontsize=12, color=INK2); sx.tick_params(labelsize=11)
sx.legend(ncol=5, fontsize=12, frameon=False, loc='lower right', bbox_to_anchor=(1, 1.0))
cursor = sx.axvline(0, color=INK, lw=1.5)


def draw(k):
    for j in range(6):
        ims[j].set_data(plt.imread(FR/f'p{j}/{k:03d}.png'))
    for i in range(5):
        meds[i].set_text(f"median {np.median(Z[f'm{i}_err'][k]):.1f} mm")
    cursor.set_xdata([k/FPS]*2)
    clock.set_text(f'world {w}   ·   t = {k/FPS:.3f} s  (frame {k+1}/{n})   ·   {FPS_OUT/FPS:.2g}x real time   ·   after one calibration interaction   ·   rendered in Blender (Cycles)')
    return []


frames = list(range(n))+[n-1]*int(FPS_OUT)
out = OUT/f'fig1-blender-{scen}-{w}.mp4'
animation.FuncAnimation(fig, draw, frames=frames).save(out, writer=animation.FFMpegWriter(fps=FPS_OUT, codec='libx264', bitrate=9000, extra_args=['-pix_fmt', 'yuv420p']))
draw(int(np.argmax(np.linalg.norm(Z['U'], axis=2).max(1)[:int(.3*FPS)+1]))); fig.savefig(OUT/f'fig1-blender-{scen}-{w}-still.png', dpi=100)
print(out)
