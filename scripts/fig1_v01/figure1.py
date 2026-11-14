"""Figure 1 (benchmark): three held-out interactions, five methods, before/after one calibration interaction.
Every prediction is read through the sealed manifests (fig1_score.verify checks each sha256 first).

fig1-benchmark : one row per scenario. Left: reference and each method AFTER calibration at the same instant (the frame of
                 peak true displacement, middle world), particles coloured by displacement error against the reference
                 (error = |u_method(p) - u_true(p)|, u_true sampled at the nearest reference node to the particle's rest
                 position, so it measures motion, not reconstruction size); grey underlay = reference shape.
                 Right: error over the whole held-out trajectory (RMS over the scored stations/probes, pooled over the
                 three worlds), after calibration.
fig1-scores    : scored error per method before -> after calibration (mean over worlds, worlds as small dots).
Writes output/paper-figures/fig1-{benchmark,scores}.{png,pdf}."""
import json, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from scipy.spatial import cKDTree

HERE = Path(__file__).resolve().parent; R = HERE.parents[1]
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(R/'scripts/stem_trial_v01'))
import fig1_score as FS
import block as B, fem
from score import truth_stations
from ours import station_matrix

OUT = R/'output/paper-figures'; OUT.mkdir(parents=True, exist_ok=True)
ST, BK, EXT = FS.ST, FS.BK, FS.EXT
SCORES = json.loads((FS.REP/'fig1-scores.json').read_text())['rows']
FPS = 48.0
# categorical slots 1-6 of the validated reference palette, fixed order (ours always slot 1)
COL = {'ours': '#2a78d6', 'GIC': '#eb6834', 'PAC-NeRF': '#1baf7a', 'M-PhyGs': '#eda100', 'PhysTwin': '#e87ba4', 'FEM control': '#008300'}
LABEL = {'ours': 'ours (RGB)', 'GIC': 'GIC', 'PAC-NeRF': 'PAC-NeRF', 'M-PhyGs': 'M-PhyGs', 'PhysTwin': 'PhysTwin',
         'FEM control': 'fitted spatially varying FEM (control)'}
METHODS = ['GIC', 'PAC-NeRF', 'M-PhyGs', 'PhysTwin', 'ours']
INK, INK2, MUTED, GRID, UNDER = '#1f1e1c', '#52514e', '#8a8984', '#e4e3df', '#dcdbd6'
ERR_CMAP = LinearSegmentedColormap.from_list('err', ['#d8d3f3', '#9085e9', '#6250d6', '#3a2c9c', '#1d1456'])  # one hue (violet), light -> dark
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 8, 'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.edgecolor': MUTED, 'axes.labelcolor': INK2, 'xtick.color': INK2, 'ytick.color': INK2, 'text.color': INK,
                     'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': .6, 'axes.axisbelow': True, 'legend.frameon': False,
                     'axes.titlesize': 8.5, 'axes.titlelocation': 'left'})

SCEN = dict(
    S1=dict(title='S1  Stem: new push (20 N, unseen direction)', worlds=FS.SW, show='W1000', ep='D2c', plane=(0, 1), metric='station'),
    S2=dict(title='S2  Block: off-centre press over a hidden stiff region', worlds=FS.BW, show='Bcentre', ep='D2', plane=(0, 1), metric='probe'),
    S3=dict(title='S3  Stem: striker impact and rebound', worlds=FS.SW, show='W1000', ep='D3', plane=(0, 2), metric='station'))
SEALED = {s: FS.verify(s) for s in SCEN}  # asserts every sealed hash


def need(scen, p):
    q = SEALED[scen](p); assert q is not None, f'not sealed: {p}'; return q


# ---------------------------------------------------------------- inputs (all via the sealed manifests)
def truth(scen, w):
    if scen == 'S2':
        z = np.load(BK/f'truth/D2-{w}.npz'); m = B.mesh(tuple(z['cells']))
        tgt = np.einsum('pn,fnj->fpj', m.interpolation(B.PROBES).toarray(), z['U'])
        return z['X'], z['U'], tgt, None
    z = np.load(ST/f"truth/{SCEN[scen]['ep']}-{w}.npz")
    if scen == 'S1':
        tgt = truth_stations('D2c', w)[0]
    else:
        tgt = np.einsum('sn,fnj->fsj', station_matrix(fem.Mesh(tuple(z['cells']))), z['U'])
    return z['X'], z['U'], tgt, (z['sphere_c'] if 'sphere_c' in z.files else None)


def particle_file(scen, meth, w, stage):  # stage D0 / D1
    ep = SCEN[scen]['ep']
    if meth in ('GIC', 'PAC-NeRF'):
        d = {'GIC': 'gic', 'PAC-NeRF': 'pac'}[meth]
        if scen == 'S2':
            return BK/d/f'forecast-D2-{stage}-{w}.npz'
        return ST/d/(f'forecast-{ep}-D0.npz' if stage == 'D0' else f'forecast-{ep}-D1-{w}.npz')
    return EXT/{'M-PhyGs': 'mphygs', 'PhysTwin': 'phystwin'}[meth]/f'forecast-{ep}-{stage}-{w}.npz'


def shape(scen, meth, w):
    """(rest (N,3), positions (F,N,3), striker centre (F,3) or None) after calibration."""
    if meth == 'ours':
        if scen == 'S1':
            z = np.load(need(scen, ST/f'ours-fields/round3-rgb-{w}-D2c.npz')); return z['X'], z['X'][None]+z['D0pD1'], None
        if scen == 'S2':
            z = np.load(need(scen, BK/f'ours/field-rgb-{w}.npz')); return z['X'], z['X'][None]+z['D0pD1'], None
        z = np.load(need(scen, ST/f'ours/striker/field-rgb-{w}-D0pD1.npz')); return z['X'], z['X'][None]+z['U'], z['sphere_c']
    z = np.load(need(scen, particle_file(scen, meth, w, 'D1'))); xyz = z['xyz']
    sc = None
    if scen == 'S3' and 'striker' in z.files and len(z['striker']):
        h = z['striker']; t = np.arange(len(xyz))/FPS
        sc = h[np.abs(h[:, 0][None]-t[:, None]).argmin(1), 1:4]
    return xyz[0], xyz, sc


def trajectory(scen, meth, w, stage):
    """predicted (F,K,3) displacement of the scored stations/probes, exactly as fig1_score computes it."""
    if meth == 'ours':
        if scen == 'S1':
            return np.load(need(scen, ST/f'ours/round3-rgb-{w}.npz'))[f"pred_D2c_{'D0' if stage == 'D0' else 'D0+D1'}"]
        if scen == 'S2':
            return np.load(need(scen, BK/f'ours/pred-rgb-{w}.npz'))[f"pred_D2_{'D0' if stage == 'D0' else 'D0+D1'}"]
        return np.load(need(scen, ST/f"ours/striker/rgb-{w}-{'D0' if stage == 'D0' else 'D0pD1'}.npz"))['station_disp']
    if meth == 'FEM control':
        return np.load(need(scen, BK/f'control/pred-fem-{w}.npz'))[f"pred_D2_{'D0' if stage == 'D0' else 'D0+D1'}"]
    p = need(scen, particle_file(scen, meth, w, stage))
    return FS.block_probe_disp_particles(p, B.PROBES) if scen == 'S2' else FS.particle_station_disp(p)


# ---------------------------------------------------------------- figure 1: snapshots + error over time
def motion_error(rest, pos, k, X, U):
    idx = cKDTree(X).query(rest)[1]
    return np.linalg.norm((pos[k]-rest)-U[k][idx], axis=1)*1e3


def snapshot_row(fig, gs_row, scen):
    cfg = SCEN[scen]; w = cfg['show']; a, b = cfg['plane']
    X, U, _, sphere = truth(scen, w)
    # instant: peak true displacement; for the striker, peak of the first swing after impact (t <= 0.3 s)
    pk = np.linalg.norm(U, axis=2).max(1)
    k = int(pk.argmax()) if sphere is None else int(pk[:int(.3*FPS)+1].argmax())
    if sphere is not None and np.linalg.norm(sphere[k]-(X+U[k])[np.linalg.norm(X+U[k]-sphere[k], axis=1).argmin()]) > .1:
        sphere = None  # striker has left the scene
    shapes = {m: shape(scen, m, w) for m in METHODS}
    errs = {m: motion_error(r, p, k, X, U) for m, (r, p, _) in shapes.items()}
    vmax = float(np.ceil(np.percentile(np.concatenate(list(errs.values())), 95)))
    norm = Normalize(0, vmax)
    ref = X+U[k]
    pts = ref[:, [a, b]] if sphere is None else np.vstack([ref[:, [a, b]], sphere[k][[a, b]]+.03, sphere[k][[a, b]]-.03])
    lo = pts.min(0); hi = pts.max(0); pad = .06*(hi-lo).max()
    rng = np.random.default_rng(0); axes = []
    for j, m in enumerate(['reference']+METHODS):
        ax = fig.add_subplot(gs_row[j//3, j % 3]); axes.append(ax)
        ax.scatter(ref[:, a], ref[:, b], s=.6, c=UNDER, lw=0, rasterized=True)
        if sphere is not None:
            ax.add_patch(plt.Circle((sphere[k][a], sphere[k][b]), 0.03, fill=False, ec=MUTED, lw=.8, ls='--'))
        if m != 'reference':
            r, p, sc = shapes[m]; sel = rng.permutation(len(r))[:5000]; e = errs[m][sel]; o = np.argsort(e)
            ax.scatter(p[k][sel][o, a], p[k][sel][o, b], s=.9, c=e[o], cmap=ERR_CMAP, norm=norm, lw=0, rasterized=True)
            if sc is not None:
                ax.add_patch(plt.Circle((sc[k][a], sc[k][b]), 0.03, fill=False, ec=INK, lw=.9))
            med = float(np.median(errs[m]))
            ax.text(.99, -.02, f'median {med:.1f} mm', transform=ax.transAxes, fontsize=6.8, color=INK2, ha='right', va='top')
        ax.set_title(LABEL.get(m, 'reference (truth)'), fontsize=7.8, color=INK, loc='left', pad=4)
        ax.set_xlim(lo[0]-pad, hi[0]+pad); ax.set_ylim(lo[1]-pad, hi[1]+pad); ax.set_aspect('equal')
        ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
        for s in ax.spines.values():
            s.set_visible(False)
        if m != 'reference':
            ax.plot([0, 1], [1.0, 1.0], transform=ax.transAxes, color=COL[m], lw=2.4, solid_capstyle='butt', clip_on=False)
    axes[0].text(0, 1.38, cfg['title'], transform=axes[0].transAxes, fontsize=9.5, fontweight='bold', color=INK, ha='left')
    axes[0].text(.99, -.02, f"world {w}, t = {k/FPS:.2f} s, {'x-y side view' if b == 1 else 'x-z top view'}",
                 transform=axes[0].transAxes, fontsize=6.8, color=INK2, ha='right', va='top')
    return norm, axes


def error_over_time(ax, scen, legend=False):
    cfg = SCEN[scen]
    meths = METHODS+(['FEM control'] if scen == 'S2' else [])
    t_end = 0
    for m in meths:
        se = []
        for w in cfg['worlds']:
            tgt = truth(scen, w)[2]; pr = trajectory(scen, m, w, 'D1')
            se.append(((pr-tgt)**2).sum(-1).mean(-1))  # per-frame mean squared station/probe error
        e = np.sqrt(np.mean(se, 0))*1e3; t = np.arange(len(e))/FPS; t_end = t[-1]
        ax.plot(t, np.maximum(e, 1e-2), color=COL[m], lw=2.4 if m == 'ours' else 1.5, ls='--' if m == 'FEM control' else '-',
                label=LABEL[m], zorder=3 if m == 'ours' else 2, solid_capstyle='round')
    ax.set_yscale('log'); ax.set_xlim(0, t_end); ax.set_ylim(.05, 40)
    ax.set_xlabel('time (s)'); ax.set_ylabel('error after calibration (mm)')
    ax.axhline(3, color=MUTED, lw=.8, ls=':'); ax.text(.01, 3.25, '3 mm', color=MUTED, fontsize=6.3)
    ax.set_title('error over the whole held-out trajectory', fontsize=7.8)


def figure_benchmark():
    fig = plt.figure(figsize=(13.5, 10.5))
    outer = fig.add_gridspec(3, 5, width_ratios=[6, .22, .09, .5, 2.3], hspace=.62, wspace=.05)
    for i, scen in enumerate(SCEN):
        gs = outer[i, 0].subgridspec(2, 3, wspace=.06, hspace=.55)
        norm, axes = snapshot_row(fig, gs, scen)
        cax = fig.add_subplot(outer[i, 2].subgridspec(3, 1, height_ratios=[.15, .7, .15])[1])
        cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=ERR_CMAP), cax=cax)
        cax.set_title('motion\nerror\n(mm)', fontsize=6.5, color=INK2, loc='center', pad=4); cb.ax.tick_params(labelsize=6.5); cb.outline.set_visible(False)
        ax = fig.add_subplot(outer[i, 4]); error_over_time(ax, scen)
        if i == 0:
            hs, ls = ax.get_legend_handles_labels()
    hs.append(plt.Line2D([], [], color=COL['FEM control'], ls='--', lw=1.5)); ls.append(LABEL['FEM control']+', S2 only')
    fig.legend(hs, ls, loc='upper center', ncol=6, bbox_to_anchor=(.5, .955), fontsize=8)
    fig.text(.13, .02, 'All methods receive the same RGB footage and known commands; one calibration interaction; predictions sealed before the reference '
             'existed.\nSnapshots: after calibration, at the frame of peak true displacement. S3: peak of the first swing after impact (the striker has already rebounded out of view). Errors over time are pooled over the 3 worlds.',
             fontsize=7, color=INK2)
    for ext in ('png', 'pdf'):
        fig.savefig(OUT/f'fig1-benchmark.{ext}', dpi=220, bbox_inches='tight')
    plt.close(fig)


# ---------------------------------------------------------------- figure 2: scored error before -> after
def score_of(scen, meth, w, ev, key='rms_mm'):
    name = {'ours': 'ours (RGB)', 'FEM control': 'fitted spatially varying FEM (control)'}.get(meth, meth)
    r = [x for x in SCORES if x['scenario'] == scen and x['world'] == w and x['method'] == name and x['evidence'] == ev]
    return r[0][key] if r else np.nan


def figure_scores():
    panels = [('S1', 'rms_mm', 'S1 stem push: station error (mm)'), ('S2', 'rms_mm', 'S2 block press: surface-probe error (mm)'),
              ('S3', 'rms_mm', 'S3 striker: station error (mm)'), ('S3', 'impulse_err_Ns', 'S3 striker: impulse error (N·s)')]
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.4), gridspec_kw=dict(wspace=.85))
    for ax, (scen, key, title) in zip(axes, panels):
        meths = METHODS+(['FEM control'] if scen == 'S2' else [])
        for i, m in enumerate(meths[::-1]):
            b = [score_of(scen, m, w, 'passive only', key) for w in SCEN[scen]['worlds']]
            a = [score_of(scen, m, w, 'after poke', key) for w in SCEN[scen]['worlds']]
            mb, ma = np.nanmean(b), np.nanmean(a)
            ax.annotate('', xy=(ma, i), xytext=(mb, i), arrowprops=dict(arrowstyle='-|>', color=COL[m], lw=1.4, mutation_scale=8))
            ax.scatter([mb], [i], s=34, facecolor='white', edgecolor=COL[m], lw=1.6, zorder=3)
            ax.scatter([ma], [i], s=40, color=COL[m], zorder=4)
            ax.scatter(a, [i-.22]*len(a), s=9, color=COL[m], alpha=.55, lw=0, zorder=2)
            ax.text(1.02, i, f'{ma:.2f}' if key == 'rms_mm' else f'{ma:.3f}', transform=ax.get_yaxis_transform(), fontsize=6.8, color=INK2, va='center')
        ax.set_yticks(range(len(meths))); ax.set_yticklabels([LABEL[m].replace('fitted spatially varying FEM (control)', 'FEM control') for m in meths[::-1]])
        ax.set_xscale('log'); ax.set_title(title, fontsize=8); ax.grid(axis='y', visible=False)
        ax.set_ylim(-.6, len(meths)-.4)
    axes[0].scatter([], [], s=34, facecolor='white', edgecolor=MUTED, lw=1.6, label='initial footage only (mean of 3 worlds)')
    axes[0].scatter([], [], s=40, color=MUTED, label='after one calibration interaction (mean)')
    axes[0].scatter([], [], s=9, color=MUTED, alpha=.55, label='after calibration, each world')
    axes[0].legend(loc='lower left', bbox_to_anchor=(0, 1.12), ncol=3, fontsize=7)
    for ext in ('png', 'pdf'):
        fig.savefig(OUT/f'fig1-scores.{ext}', dpi=220, bbox_inches='tight')
    plt.close(fig)


if __name__ == '__main__':
    figure_scores(); figure_benchmark()
    print(OUT/'fig1-benchmark.png', OUT/'fig1-scores.png')
