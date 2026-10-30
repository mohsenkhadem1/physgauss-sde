"""Plot every completed primary pair; report absent pairs and reference flags."""
from pathlib import Path
import hashlib
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT/'reports/coupled-surface-v13'


def main():
    paths = [REPORT/name for name in ('protocol.json', 'reserved-results.json', 'ablation-results.json')]
    protocol, result, supplements = [json.loads(path.read_text()) for path in paths]
    scenes = protocol['primary_forecast_scenarios']
    expected = {(obj['id'], scene) for obj in protocol['objects'] if obj['split']=='reserved' for scene in scenes}
    rows=[row for row in result['cases'] if row['scenario'] in scenes]
    primary = {(row['object'], row['scenario']): row for row in rows}
    if len(primary)!=len(rows):raise ValueError('Duplicate primary plot records')
    rows=[row for row in supplements['cases'] if row['split']=='reserved']
    supplemental = {(row['object'], row['scenario']): row for row in rows}
    if len(supplemental)!=len(rows):raise ValueError('Duplicate supplemental plot records')
    if set(primary) != expected:
        raise ValueError('Primary plot requires all fixed cohort records, including failures')
    colors = dict(zip(scenes, ['#247c91', '#b96a2d', '#775a9e', '#588333']))
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 5.4))
    panels = []
    for ax, method, label in zip(axes, ['rigid', 'nominal'], ['Rigid contact', 'Fixed nominal FEM']):
        points, missing = [], []
        for key in sorted(expected):
            row = primary[key]
            ours = row.get('results', {}).get('fem', {})
            other = (row if method=='rigid' else supplemental.get(key, {})).get('results', {}).get(method, {})
            if not ours.get('complete') or not other.get('complete'):
                missing.append(dict(object=key[0], scenario=key[1]))
                continue
            if not np.isfinite([ours['coordinate_rmse_m'],other['coordinate_rmse_m']]).all():raise ValueError('Nonfinite completed plot score')
            point = dict(object=key[0], scenario=key[1], fem_mm=ours['coordinate_rmse_m']*1000,
                         comparator_mm=other['coordinate_rmse_m']*1000,
                         joint_reference_qualified=bool(row.get('joint_reference_qualified')))
            points.append(point)
            ax.scatter(point['comparator_mm'], point['fem_mm'], s=54, linewidths=1.4,
                       edgecolors=colors[key[1]], facecolors=colors[key[1]] if point['joint_reference_qualified'] else 'none')
        limit = max([1.] + [max(p['comparator_mm'], p['fem_mm'])*1.08 for p in points])
        ax.plot([0, limit], [0, limit], color='#8c949c', linestyle='--', linewidth=1)
        ax.set(xlim=(0, limit), ylim=(0, limit), xlabel=label+' world RMSE (mm)',
               ylabel='Inferred FEM world RMSE (mm)', title=f'{label}: {len(points)}/{len(expected)} paired cases')
        ax.set_aspect('equal', adjustable='box')
        ax.grid(alpha=.18)
        for scene in scenes:
            ax.scatter([], [], color=colors[scene], label=scene.replace('_', ' '), s=35)
        ax.legend(loc='upper left', fontsize=8, frameon=True)
        panels.append(dict(comparator=method, expected=len(expected), paired=len(points), points=points, missing_pairs=missing))
    fig.suptitle('Reserved primary cohort: below the diagonal favors inferred FEM', fontsize=13, y=.98)
    fig.text(.5, .035, 'Fixed membership; eight objects share repeated interactions. Hollow markers: reference check failed.\n'
             'Failed predictions cannot be plotted; paired coverage and missing cases are retained in the data.',
             ha='center', va='bottom', fontsize=9, color='#42515d')
    fig.tight_layout(rect=(0, .11, 1, .93))
    for suffix in ('png', 'svg'):
        fig.savefig(REPORT/f'primary-paired-comparison.{suffix}', dpi=180, facecolor='white')
    plt.close(fig)
    metadata = dict(scope='Fixed primary cohort only; no per-case quality filtering; stress results remain separate.',
        source_sha256={str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths},
        plotter_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), panels=panels)
    (REPORT/'primary-paired-comparison-data.json').write_text(json.dumps(metadata, indent=2)+'\n')


if __name__ == '__main__':
    main()
