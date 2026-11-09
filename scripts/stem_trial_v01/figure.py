"""Figure from scored series: lateral (z) displacement of the 0.75 L station during D2, per world."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

R = Path(__file__).resolve().parents[2]
S = dict(np.load(R/'runs/stem-trial-v01/scored-series.npz'))
scores = json.loads((R/'reports/stem-trial-v01/scores.json').read_text())
out = R/'output/stem-trial-v01'; out.mkdir(parents=True, exist_ok=True)
t = np.arange(37)/48
lines = [('GIC|D0 only', 'GIC, passive fit only', '#c0392b', '--'),
         ('GIC|after D1 poke refit', 'GIC, refit after poke', '#c0392b', '-'),
         ('ours posterior mean|D0 only', 'Ours, passive only (posterior mean)', '#2471a3', '--'),
         ('ours posterior mean|D0 + D1 poke', 'Ours, after poke', '#2471a3', '-'),
         ('GIC simulator, true E/nu/rho (oracle control)|D0 reconstruction', 'GIC simulator, true parameters', '#7f8c8d', ':')]
fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), sharey=True)
for ax, w in zip(axes, ['W500', 'W1000', 'W2000']):
    ax.plot(t, S[f'truth-{w}'][:, 2, 2]*1e3, color='k', lw=2.4, label='Held-out truth')
    for key, lab, c, ls in lines:
        k = f'{key}|{w}'
        if k in S:
            ax.plot(t, S[k][:, 2, 2]*1e3, color=c, ls=ls, lw=1.6, label=lab)
    rho = {'W500': 500, 'W1000': 1000, 'W2000': 2000}[w]
    ax.set_title(f'density {rho} kg/m³ (E/ρ identical)', fontsize=10)
    ax.set_xlabel('time (s)'); ax.grid(alpha=.3)
axes[0].set_ylabel('lateral displacement at 0.75 L (mm)')
axes[0].legend(fontsize=7.5, loc='upper left')
fig.suptitle('Held-out 20 N push at 0.75 L: one passive video cannot fix the force scale; one calibrated poke can', fontsize=11)
fig.tight_layout(); fig.savefig(out/'d2-lateral-075L.png', dpi=160); fig.savefig(out/'d2-lateral-075L.pdf')
print(out/'d2-lateral-075L.png')
