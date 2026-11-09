"""Paper figures: why rendering alone and passive-video physics alone are not enough.
Reads only saved results (stem trial rounds 1-2, torus cross-method audit). Writes output/paper-figures/."""
import json, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0, str(Path(__file__).resolve().parent))
from score import truth_stations

R = Path(__file__).resolve().parents[2]
T = R/'runs/stem-trial-v01'; REP = R/'reports/stem-trial-v01'
OUT = R/'output/paper-figures'; OUT.mkdir(parents=True, exist_ok=True)
S1 = json.loads((REP/'scores.json').read_text()); S2 = json.loads((REP/'scores-round2.json').read_text())
TOR = json.loads((R/'reports/gap-crossmethod-v02/crossmethod-independent-audit.json').read_text())['paired_foreground_image']['rows']

# Roles. Methods are categorical (validated reference slots 1-3); density is a magnitude (one ramp).
OURS, GIC, PAC = '#2a78d6', '#eb6834', '#1baf7a'
INK, INK2, MUTED, GRID = '#0b0b0b', '#52514e', '#8a8984', '#e4e3df'
DENS = {'W500': '#b9b8b2', 'W1000': '#7a7973', 'W2000': '#2f2e2b'}
RHO = {'W500': 500, 'W1000': 1000, 'W2000': 2000}
W = ['W500', 'W1000', 'W2000']
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9, 'axes.edgecolor': MUTED, 'axes.labelcolor': INK2,
                     'xtick.color': INK2, 'ytick.color': INK2, 'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': .6, 'axes.axisbelow': True,
                     'axes.titlesize': 10, 'axes.titleweight': 'bold', 'axes.titlelocation': 'left', 'legend.frameon': False})
t = np.arange(37)/48; t0 = np.arange(25)/48


def row(scores, **k):
    return next(r for r in scores if all(r.get(a) == b for a, b in k.items()))


def ours_passive_band():
    """Posterior 5-95% band of our passive-only prediction of D2 lateral motion at 0.75 L."""
    b = np.load(T/'ours/bank.npz'); w = np.load(T/'ours/round2-tracks-W1000.npz')['w_D0']
    from ours import RHO_GRID
    traj = b['st_g'][:, :, None, :, 2, 2]+(20/RHO_GRID)[None, None, :, None]*b['st_h'][:, :, None, :, 2, 2]
    ww = w.ravel(); tr = traj.reshape(-1, traj.shape[-1]); keep = ww > 1e-8; ww, tr = ww[keep]/ww[keep].sum(), tr[keep]
    q = []
    for f in range(tr.shape[1]):
        o = np.argsort(tr[:, f]); c = np.cumsum(ww[o])
        q.append([tr[o, f][np.searchsorted(c, p)] for p in (.05, .5, .95)])
    return np.array(q)


def label_end(ax, x, y, s, c, dy=0):
    ax.annotate(s, (x, y), xytext=(4, dy), textcoords='offset points', color=c, fontsize=8, va='center')


# ---------- Figure 1: identical evidence, different futures ----------
def fig_identical(ax0, ax1):
    for w in W:
        d0, _ = truth_stations('D0', 'W1000' if w != 'W500' else 'W500')
        ax0.plot(t0, d0[:, 3, 1]*1e3, color=DENS[w], lw=2.6 if w == 'W500' else 1.4, ls='-' if w != 'W2000' else (0, (4, 3)))
    ax0.set_title('What the cameras see: passive sag')
    ax0.set_xlabel('time (s)'); ax0.set_ylabel('tip vertical displacement (mm)')
    ax0.set_ylim(-39, 3)
    ax0.text(.5, .04, 'three stems, ρ = 500 / 1000 / 2000 kg/m³, equal E/ρ: the three curves coincide', transform=ax0.transAxes, ha='center', fontsize=8, color=INK2)
    band = ours_passive_band()
    ax1.fill_between(t, band[:, 0]*1e3, band[:, 2]*1e3, color=OURS, alpha=.12, lw=0)
    peaks = {}
    for w in W:
        tr, _ = truth_stations('D2', w); z = tr[:, 2, 2]*1e3; peaks[w] = z.max()
        ax1.plot(t, z, color=DENS[w], lw=1.8); label_end(ax1, t[-1], z[-1], f'ρ={RHO[w]}', INK2)
    g = np.load(T/'gic/forecast-D0.npz')['station_centroid']; gz = (g[:, 2, 2]-g[0, 2, 2])*1e3
    ax1.plot(t, gz, color=GIC, lw=2, ls=(0, (5, 2)))
    ax1.plot(t, band[:, 1]*1e3, color=OURS, lw=1.2, ls=(0, (1.5, 1.5)))
    mid = (peaks['W500']+peaks['W2000'])/2
    ax1.annotate('', xy=(.69, peaks['W500']), xytext=(.69, peaks['W2000']), arrowprops=dict(arrowstyle='<->', color=INK, lw=1))
    ax1.text(.40, 37, f'peaks {peaks["W500"]-peaks["W2000"]:.0f} mm apart: any single prediction\nmisses one stem by ≥ {(peaks["W500"]-peaks["W2000"])/2:.1f} mm',
             ha='center', va='center', fontsize=8, color=INK, bbox=dict(fc='white', ec='none', alpha=.85, pad=1.5))
    ax1.text(.35, 17.5, 'both centre on ρ = 1000', fontsize=7.5, color=INK2, ha='center')
    ax1.text(.02, .97, 'GIC (one answer, fixed density)', transform=ax1.transAxes, color=GIC, fontsize=8, va='top', weight='bold')
    ax1.text(.02, .89, 'ours: passive-only posterior 5–95% (risk 0.86)', transform=ax1.transAxes, color=OURS, fontsize=8, va='top', weight='bold')
    ax1.set_title('What a robot needs: response to a 20 N push')
    ax1.set_xlabel('time (s)'); ax1.set_ylabel('lateral displacement at 0.75 L (mm)'); ax1.set_xlim(0, .83)


# ---------- Figure 2: rendering is not physics ----------
def fig_rendering(ax0, ax1):
    names = [('GIC deformation network (no physics)', 'GIC, no physics', GIC, '-'),
             ('GIC native physics forecast', 'GIC, physics', GIC, (0, (5, 2))),
             ('PAC-NeRF native forecast', 'PAC-NeRF, physics', PAC, '-')]
    for k, lab, c, ls in names:
        a, b = TOR[k]['fit_frames_0_7_mean_db'], TOR[k]['future_8_13_mean_db']
        ax0.plot([0, 1], [a, b], color=c, lw=2, ls=ls, marker='o', ms=6, mec='white', mew=1.5)
        ax0.annotate(f'{lab}  {b:.1f}', (1, b), xytext=(6, 6 if lab == 'GIC, physics' else (-6 if 'PAC' in lab else 0)),
                     textcoords='offset points', fontsize=8, color=INK2, va='center')
        ax0.annotate(f'{a:.1f}', (0, a), xytext=(-6, 5 if 'PAC' in lab else (-5 if lab == 'GIC, physics' else 0)), textcoords='offset points', fontsize=8, color=INK2, ha='right', va='center')
    ax0.axhline(TOR['GIC deformation network (no physics)']['frozen_frame7_future_mean_db'], color=MUTED, lw=1, ls=':')
    ax0.text(1.06, 10.1, 'holding the last\nframe still', color=MUTED, fontsize=8, ha='left')
    ax0.set_xticks([0, 1]); ax0.set_xticklabels(['fitted frames\n(held-out camera)', 'future frames\n(held-out camera)'])
    ax0.set_xlim(-.35, 1.9); ax0.set_ylabel('foreground PSNR (dB)')
    ax0.set_title('Torus: best visual fit, worst forecast')
    # stem: PSNR vs 3D error, held-out push D2b, same appearance for all motion sources
    pts = []
    for r in S2['rendering']:
        if r['episode'] == 'D2b' and r['appearance'] == 'GIC D0' and r['method'] != 'GIC oracle parameters':
            m = r['method']
            if m == 'truth motion':
                pmm = 0.0
            elif m == 'GIC forecast':
                pmm = row(S2['physics'], episode='D2b', world=r['world'], method='GIC', evidence='D0 only')['rms_mm']
            else:
                s = 'RGB-D' if 'rgbd' in m else 'tracks'; ev = 'D0 + D1 poke' if m.endswith('D0pD1') else 'D0 only'
                pmm = row(S2['physics'], episode='D2b', world=r['world'], method=f'ours {s}', evidence=ev)['rms_mm']
                if s == 'tracks':
                    continue
            pts.append((pmm, r['fg_psnr_db'], m, r['world']))
    style = {'truth motion': (INK, 'D', 'true motion'), 'GIC forecast': (GIC, 'o', 'GIC, passive fit'),
             'ours rgbd D0': (OURS, 'o', 'ours, passive only'), 'ours rgbd D0pD1': (OURS, 's', 'ours, after poke')}
    for m, (c, mk, lab) in style.items():
        xs = [p for p in pts if p[2] == m]
        ax1.scatter([p[0] for p in xs], [p[1] for p in xs], s=46, c='white' if m == 'ours rgbd D0' else c,
                    edgecolors=c, linewidths=1.8, marker=mk, label=lab, zorder=3)
    x = np.array([p[0] for p in pts]); y = np.array([p[1] for p in pts])
    ax1.set_xlabel('3D error of the predicted motion (mm RMS)'); ax1.set_ylabel('held-out camera PSNR (dB)')
    ax1.set_title('Stem: pixels compress millimetres')
    ax1.annotate('14 mm miss → −5 dB', (14.3, 16.34), xytext=(10.2, 19.3), fontsize=8, color=INK2, arrowprops=dict(arrowstyle='-', color=MUTED, lw=.8))
    ax1.annotate('3 mm miss → −1 dB', (3.08, 20.53), xytext=(4.5, 21.3), fontsize=8, color=INK2, arrowprops=dict(arrowstyle='-', color=MUTED, lw=.8))
    ax1.legend(loc='upper right', fontsize=8, handletextpad=.3)
    ax1.set_xlim(-.6, 15.5)


# ---------- Figure 3: one physical observation ----------
def fig_poke(ax):
    rows = []
    for ep, S, ev_g, lab_ep in (('D2', S1['rows'], 'after D1 poke refit', 'push A'), ('D2b', S2['physics'], 'after D1 poke refit', 'push B')):
        for w in W:
            gk = dict(world=w, method='GIC') | ({'episode': 'D2b'} if ep == 'D2b' else {})
            g0 = row(S, evidence='D0 only', **gk)['rms_mm']; g1 = row(S, evidence=ev_g, **gk)['rms_mm']
            if ep == 'D2':
                o0 = row(S, world=w, method='ours posterior mean', evidence='D0 only'); o1 = row(S, world=w, method='ours posterior mean', evidence='D0 + D1 poke')
            else:
                o0 = row(S, episode='D2b', world=w, method='ours RGB-D', evidence='D0 only'); o1 = row(S, episode='D2b', world=w, method='ours RGB-D', evidence='D0 + D1 poke')
            rows.append((f'{lab_ep}\nρ={RHO[w]}', g0, g1, o0['rms_mm'], o1['rms_mm'], o0['predicted_risk_rms_gt_3mm'], o1['predicted_risk_rms_gt_3mm']))
    y = np.arange(len(rows))[::-1]*1.0
    ax.axvspan(0, 3, color='#2a78d6', alpha=.05, lw=0); ax.axvline(3, color=MUTED, lw=1, ls=':')
    ax.text(3.15, y[-1]-.62, '3 mm tolerance', color=MUTED, fontsize=8, va='center')
    for yi, (lab, g0, g1, o0, o1, r0, r1) in zip(y, rows):
        for (a, b, c, dy) in ((g0, g1, GIC, .17), (o0, o1, OURS, -.17)):
            ax.annotate('', xy=(b, yi+dy), xytext=(a, yi+dy), arrowprops=dict(arrowstyle='-|>', color=c, lw=1.6, mutation_scale=9, shrinkA=3, shrinkB=3))
            ax.scatter([a], [yi+dy], s=26, facecolor='white', edgecolor=c, lw=1.5, zorder=3)
            ax.scatter([b], [yi+dy], s=26, color=c, zorder=3)
        ax.text(15.6, yi-.17, f'risk {r0:.2f} → {r1:.2f}', color=OURS, fontsize=7.5, va='center')
    ax.set_yticks(y); ax.set_yticklabels([r[0] for r in rows], fontsize=8)
    ax.set_xlim(0, 19.5); ax.set_xlabel('3D error on the held-out push (mm RMS)   ○ passive video only   ● after one 40 N poke')
    ax.set_title('One calibrated poke: ours recovers everywhere; GIC only where its density guess was right')
    from matplotlib.lines import Line2D
    ax.legend(handles=[Line2D([], [], color=GIC, lw=1.6, label='GIC (native refit, density fixed)'),
                       Line2D([], [], color=OURS, lw=1.6, label='ours (posterior, RGB-D for push B)')], loc='upper center', bbox_to_anchor=(.5, -.13), ncol=2, fontsize=8.5)
    ax.set_ylim(y[-1]-.9, y[0]+.6)
    ax.grid(axis='y', visible=False)


def save(fig, name):
    fig.savefig(OUT/f'{name}.png', dpi=200, bbox_inches='tight'); fig.savefig(OUT/f'{name}.pdf', bbox_inches='tight'); plt.close(fig)
    print(OUT/f'{name}.png')


fig, ax = plt.subplots(1, 2, figsize=(10, 3.6), gridspec_kw=dict(wspace=.28)); fig_identical(*ax); save(fig, 'fig1-identical-evidence')
fig, ax = plt.subplots(1, 2, figsize=(10, 3.6), gridspec_kw=dict(wspace=.3, width_ratios=[1, 1.15])); fig_rendering(*ax); save(fig, 'fig2-rendering-is-not-physics')
fig, ax = plt.subplots(figsize=(8.2, 5.4)); fig_poke(ax); save(fig, 'fig3-one-poke')
fig = plt.figure(figsize=(11, 11.2)); gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 1.45], hspace=.42, wspace=.28)
fig_identical(fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]))
fig_rendering(fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1]))
fig_poke(fig.add_subplot(gs[2, :]))
for a, l in zip(fig.axes, 'abcde'):
    a.text(-.08 if l != 'e' else -.04, 1.06, l, transform=a.transAxes, weight='bold', fontsize=12, color=INK)
save(fig, 'fig-key-rendering-vs-physics')
