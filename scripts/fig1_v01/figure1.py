"""Figure 1 (benchmark): three scenarios x {GIC, PAC-NeRF, ours}, before/after one calibration interaction.
Reads reports/fig1-v01/fig1-scores.json. Writes output/paper-figures/fig1-benchmark.{png,pdf}."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

R = Path(__file__).resolve().parents[2]
S = json.loads((R/'reports/fig1-v01/fig1-scores.json').read_text())['rows']
OUT = R/'output/paper-figures'
OURS, GIC, PAC = '#2a78d6', '#eb6834', '#1baf7a'
INK2, MUTED, GRID = '#52514e', '#8a8984', '#e4e3df'
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.edgecolor': MUTED, 'axes.labelcolor': INK2, 'xtick.color': INK2, 'ytick.color': INK2,
                     'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': .6, 'axes.axisbelow': True, 'legend.frameon': False,
                     'axes.titlesize': 10, 'axes.titleweight': 'bold', 'axes.titlelocation': 'left'})
METHODS = [('GIC', GIC), ('PAC-NeRF', PAC), ('ours (RGB)', OURS)]
SCEN = [('S1', 'Stem: new push (20 N, unseen direction)', ['W500', 'W1000', 'W2000'], ['ρ=500', 'ρ=1000', 'ρ=2000'], 'rms_mm', 'station error (mm)'),
        ('S2', 'Block: off-centre press, hidden stiff region', ['Bleft', 'Bcentre', 'Bright'], ['stiff left', 'stiff centre', 'stiff right'], 'rms_mm', 'surface error (mm)'),
        ('S3', 'Stem: striker impact and rebound', ['W500', 'W1000', 'W2000'], ['ρ=500', 'ρ=1000', 'ρ=2000'], 'impulse_err_Ns', 'impulse error (N·s)')]


def val(scen, w, m, ev, key):
    r = [x for x in S if x['scenario'] == scen and x['world'] == w and x['method'] == m and x['evidence'] == ev and key in x]
    return r[0][key] if r else np.nan


fig, axes = plt.subplots(3, 1, figsize=(9.5, 10.5), gridspec_kw=dict(hspace=.55))
for ax, (scen, title, worlds, wl, key, yl) in zip(axes, SCEN):
    x = np.arange(len(worlds)); bw = .26
    for j, (m, c) in enumerate(METHODS):
        b = [val(scen, w, m, 'passive only', key) for w in worlds]; a = [val(scen, w, m, 'after poke', key) for w in worlds]
        xs = x+(j-1)*bw
        ax.bar(xs, a, bw*.9, color=c, label=f'{m}, after calibration')
        ax.scatter(xs, b, marker='_', s=260, color=c, linewidths=2.2, zorder=3, label=f'{m}, before' if scen == 'S1' else None)
        for xi, bi, ai in zip(xs, b, a):
            if np.isfinite(bi) and np.isfinite(ai):
                ax.plot([xi, xi], [ai, bi], color=c, lw=.8, ls=':', zorder=2)
    if key == 'rms_mm':
        ax.axhline(3, color=MUTED, lw=1, ls=':'); ax.text(x[-1]+.45, 3, '3 mm', color=MUTED, fontsize=8, va='center')
    means = {m: np.nanmean([val(scen, w, m, 'after poke', key) for w in worlds]) for m, _ in METHODS}
    ax.set_title(f'{title}   mean after calibration: ' + ', '.join(f'{m.split(" ")[0]} {v:.2f}' for m, v in means.items()), fontsize=9.5)
    ax.set_xticks(x); ax.set_xticklabels(wl); ax.set_ylabel(yl)
axes[0].legend(ncol=3, fontsize=8, loc='upper center', bbox_to_anchor=(.5, 1.42))
fig.text(.5, .015, 'bars: error after one calibration interaction;  dashes: error from the initial footage alone.  Same RGB frames and known commands for every method.',
         ha='center', fontsize=8.5, color=INK2)
fig.savefig(OUT/'fig1-benchmark.png', dpi=180, bbox_inches='tight'); fig.savefig(OUT/'fig1-benchmark.pdf', bbox_inches='tight')
print(OUT/'fig1-benchmark.png')
