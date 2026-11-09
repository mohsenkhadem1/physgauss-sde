"""One figure linking rendering and physics on the stem: observed-motion fit, held-out miss, poke recovery."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

R = Path(__file__).resolve().parents[2]
T = R/'runs/stem-trial-v01'
P = dict(np.load(T/'render-panels.npz'))
S = json.loads((R/'reports/stem-trial-v01/scores-round2.json').read_text())
out = R/'output/stem-trial-v01'; out.mkdir(parents=True, exist_ok=True)
W = ['W500', 'W1000', 'W2000']; RHO = {'W500': 500, 'W1000': 1000, 'W2000': 2000}
K = 1  # index into kept frames [12, 24, 36] -> frame 24 (0.5 s)


def crop(imgs):
    m = np.any([(i.astype(int).sum(-1) > 0) for i in imgs], 0)
    ys, xs = np.nonzero(m); pad = 18
    return slice(max(ys.min()-pad, 0), ys.max()+pad), slice(max(xs.min()-pad, 0), xs.max()+pad)


def psnr(ep, w, method, appearance='GIC D0'):
    for r in S['rendering']:
        if r['episode'] == ep and r['world'] == w and r['method'] == method and r['appearance'] == appearance:
            return r['fg_psnr_db']


def rms(ep, w, method, evidence):
    for r in S['physics']:
        if r['episode'] == ep and r['world'] == w and r['method'] == method and r['evidence'] == evidence:
            return r['rms_mm']


w = 'W500'
obs = [('reference (camera 0, never fitted)', P['ref|D0-W1000'][K]),
       ('GIC deformation network', P['GIC D0|GIC deformation network|D0 observed|all'][K]),
       ('GIC physics', P['GIC D0|GIC physics|D0 observed|all'][K])]
held = [('held-out reference', P[f'ref|D2b-{w}'][K], None),
        ('GIC, passive fit', P[f'GIC D0|GIC forecast|D2b|{w}'][K], psnr('D2b', w, 'GIC forecast')),
        ('GIC, refit after poke*', P[f'GIC D1-{w}|GIC forecast|D2b|{w}'][K], psnr('D2b', w, 'GIC forecast', f'GIC D1-{w}')),
        ('ours, passive only', P[f'GIC D0|ours rgbd D0|D2b|{w}'][K], psnr('D2b', w, 'ours rgbd D0')),
        ('ours, after poke', P[f'GIC D0|ours rgbd D0pD1|D2b|{w}'][K], psnr('D2b', w, 'ours rgbd D0pD1')),
        ('true motion (ceiling)', P[f'GIC D0|truth motion|D2b|{w}'][K], psnr('D2b', w, 'truth motion'))]
cy, cx = crop([o[1] for o in obs]+[h[1] for h in held])

fig = plt.figure(figsize=(15, 10.8))
gs = fig.add_gridspec(4, 6, height_ratios=[1, 1, 1, 1.6], hspace=.18, wspace=.05)
fig.text(.125, .915, 'A  Observed passive motion (D0) from camera 0, never used for fitting (frame 24)', weight='bold', fontsize=11)
psn_obs = [None, next(r['fg_psnr_db'] for r in S['rendering'] if r['episode'] == 'D0 observed' and r['method'] == 'GIC deformation network'),
           next(r['fg_psnr_db'] for r in S['rendering'] if r['episode'] == 'D0 observed' and r['method'] == 'GIC physics')]
for i, ((t, im), v) in enumerate(zip(obs, psn_obs)):
    ax = fig.add_subplot(gs[0, i]); ax.imshow(im[cy, cx]); ax.set_title(t, fontsize=9); ax.axis('off')
    if v:
        ax.text(.02, .04, f'{v:.1f} dB', transform=ax.transAxes, color='w', fontsize=9, weight='bold')
ax = fig.add_subplot(gs[0, 3:]); ax.axis('off')
ax.text(0, .55, 'GIC reproduces the motion it was fitted to: its deformation\nnetwork reaches 24.9 dB on the unseen camera, and its\nphysics replay 20.6 dB. Fitting quality does not reveal\nwhether a new force will be predicted correctly (B-D).', fontsize=10, va='center')
fig.text(.125, .725, 'B  Held-out push D2b (15 N down + 20 N sideways at 0.75 L), density 500 kg/m³, frame 24; bottom row: |prediction - reference| x4', weight='bold', fontsize=11)
for i, (t, im, v) in enumerate(held):
    ax = fig.add_subplot(gs[1, i]); ax.imshow(im[cy, cx]); ax.set_title(t, fontsize=9); ax.axis('off')
    if v:
        ax.text(.02, .04, f'{v:.1f} dB', transform=ax.transAxes, color='w', fontsize=9, weight='bold')
    ax = fig.add_subplot(gs[2, i]); ax.axis('off')
    if i:
        err = np.abs(im[cy, cx].astype(float)-held[0][1][cy, cx].astype(float)).mean(-1)/255*4
        ax.imshow(np.clip(err, 0, 1), cmap='magma', vmin=0, vmax=1)
    else:
        ax.text(.5, .5, 'Same GIC-learned appearance\nfor every motion source;\n*refit uses its own refit\nappearance. Ours: RGB-D.', ha='center', va='center', fontsize=9)
ref = held[0][1].astype(float)
methods = [('GIC forecast', 'GIC', 'D0 only', 'GIC, passive fit', '#c0392b', 'GIC D0'),
           ('GIC forecast', 'GIC', 'after D1 poke refit', 'GIC, refit after poke*', '#e59866', 'refit'),
           ('ours rgbd D0', 'ours RGB-D', 'D0 only', 'ours RGB-D, passive only', '#7fb3d5', 'GIC D0'),
           ('ours rgbd D0pD1', 'ours RGB-D', 'D0 + D1 poke', 'ours RGB-D, after poke', '#1f618d', 'GIC D0')]
x = np.arange(3); bw = .2
sub = gs[3, :].subgridspec(1, 2, wspace=.22); ax1 = fig.add_subplot(sub[0]); ax2 = fig.add_subplot(sub[1])
for j, (rm, pm, ev, lab, col, app) in enumerate(methods):
    ps = [psnr('D2b', ww, rm, f'GIC D1-{ww}' if app == 'refit' else 'GIC D0') for ww in W]
    ms = [rms('D2b', ww, pm, ev) for ww in W]
    ax1.bar(x+(j-1.5)*bw, ps, bw, color=col, label=lab); ax2.bar(x+(j-1.5)*bw, ms, bw, color=col, label=lab)
ceil = [psnr('D2b', ww, 'truth motion') for ww in W]
for i, c in enumerate(ceil):
    ax1.hlines(c, i-.45, i+.45, color='k', ls='--', lw=1.2, label='true motion, same appearance' if i == 0 else None)
ax2.axhline(3, color='k', ls=':', lw=1, label='3 mm failure threshold')
for a, yl in ((ax1, 'camera-0 foreground PSNR (dB) ↑'), (ax2, '3D station error RMS (mm) ↓')):
    a.set_xticks(x); a.set_xticklabels([f'ρ = {RHO[ww]}' for ww in W]); a.set_ylabel(yl); a.grid(axis='y', alpha=.3)
ax1.set_ylim(15, 23); ax1.set_title('C  Rendering of the held-out push (prospective)', fontsize=10, loc='left', weight='bold')
ax2.set_title('D  Physics of the same push (prospective)', fontsize=10, loc='left', weight='bold')
ax2.legend(fontsize=8, loc='upper center', bbox_to_anchor=(.5, -.1), ncol=3, frameon=False); ax1.legend(fontsize=8, loc='upper center', bbox_to_anchor=(.5, -.1), ncol=3, frameon=False)
fig.savefig(out/'stem-render-physics.png', dpi=150, bbox_inches='tight'); fig.savefig(out/'stem-render-physics.pdf', bbox_inches='tight')
print(out/'stem-render-physics.png')
