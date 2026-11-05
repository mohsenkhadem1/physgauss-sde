"""Render the fixed primary campaign comparisons; never read raw targets."""
import hashlib
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path('reports/gaussian-boundary-flow-v02')


def load(name):
    path = ROOT / name
    return json.loads(path.read_text()), hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    soft, sh = load('softvt-validation/verified-summary.json')
    phys, ph = load('phystwin-all-eight-comparison.json')
    isaac, ih = load('isaac-recorded-evaluation/summary.json')
    soft_names = list(soft['methods'])
    common = [r for r in soft['cases'] if all(r['methods'][m]['complete'] for m in soft_names)]
    assert len(soft['cases']) == 12 and len(common) == 11
    soft_rows = [dict(method=m, position_mm=float(np.mean([r['methods'][m]['coordinate_rmse_mm'] for r in common])),
                      shape_mm=float(np.mean([r['methods'][m]['aligned_shape_rmse_mm'] for r in common])),
                      complete=soft['methods'][m]['complete_cases'], expected=12) for m in soft_names]
    phys_rows = []
    for name, row in phys['aggregates'].items():
        assert row['complete_cases'] == row['expected_cases'] == 8
        phys_rows.append(dict(method=name, position_mm=row['conditional_mean_coordinate_rmse_mm'],
                              shape_mm=row['conditional_mean_aligned_shape_rmse_mm'], complete=8, expected=8))
    isaac_rows = []
    for name in ('flow', 'mdn', '25d', 'chart', 'ridge'):
        row = isaac['methods'][name]; metrics = row['episode_macro']
        assert row['complete'] == row['expected'] == 76
        isaac_rows.append(dict(method=name, position_mm=1000*metrics['coordinate_rmse_m'],
                               shape_mm=1000*metrics['rigid_aligned_shape_rmse_m'], complete=76, expected=76,
                               trajectory_energy_score_mm=1000*metrics['trajectory_energy_score_m'],
                               coordinate90_coverage=None if name == 'ridge' else metrics['coordinate90_coverage'],
                               active_parameter90_coverage=None if name == 'ridge' else metrics['active_parameter90_coverage']))
    panels = [dict(name='SoftVT: spatial Gaussian boundary + FEM', rows=soft_rows,
                   sample='11 common successful cases of 12; baselines complete on all 12'),
              dict(name='PhysTwin: spatial Gaussian boundary + FEM', rows=phys_rows,
                   sample='8 real-video cases; retrospective comparison'),
              dict(name='Isaac: global-code inverse + learned dynamics', rows=isaac_rows,
                   sample='76 previously exposed simulation cases; distinct model representation')]
    result = dict(scope='Frozen primary campaign results, before any later checkpoint-selection check', panels=panels,
                  source_sha256={'softvt-validation/verified-summary.json': sh,
                                 'phystwin-all-eight-comparison.json': ph,
                                 'isaac-recorded-evaluation/summary.json': ih},
                  softvt_excluded_from_matched_means=[r['case'] for r in soft['cases'] if r not in common],
                  limitations=['SoftVT means condition on 11 successful cases; the solver-bank failure is not imputed as zero.',
                               'PhysTwin controls added to the original four cases are retrospective.',
                               'Isaac compares a restricted global code, not the spatial Gaussian boundary-field model.',
                               'Archived Isaac models used different training objectives; the new mixture is the matched-supervision control.',
                               'Motion and shape agreement do not establish force/material recovery or certified physical realism.',
                               'Ridge is deterministic; its degenerate sample intervals are not treated as uncertainty estimates.'])
    (ROOT/'campaign-primary-results.json').write_text(json.dumps(result, indent=2)+'\n')
    labels = {'flow_raw':'Flow raw', 'mixture_raw':'Mixture raw', 'flow_tracking':'Flow rebased',
              'mixture_tracking':'Mixture rebased', 'flow_pose_contact':'Flow + contact',
              'mixture_pose_contact':'Mixture + contact', 'nearest_bank_raw':'Nearest raw',
              'nearest_bank_pose_contact':'Nearest + contact', 'prefix_persistence':'Persistence',
              'prefix_velocity':'Velocity', 'rigid_contact':'Rigid + contact',
              'flow_contact':'Flow + contact', 'mixture_contact':'Mixture + contact',
              'nearest_raw':'Nearest raw', 'nearest_contact':'Nearest + contact',
              'persistence':'Persistence', 'velocity':'Velocity', 'flow':'Flow (new)',
              'mdn':'Mixture (new)', '25d':'Archived 25D', 'chart':'Archived chart', 'ridge':'Ridge'}
    plt.rcParams.update({'font.size':10, 'svg.fonttype':'none', 'svg.hashsalt':'gbf-primary-v01'})
    fig, axes = plt.subplots(3, 2, figsize=(13, 15), gridspec_kw={'height_ratios':[11,9,5]})
    for i, panel in enumerate(panels):
        for j, (metric, title) in enumerate([('position_mm','Position error (mm)'),('shape_mm','Rigid-aligned shape error (mm)')]):
            ax = axes[i,j]; rows = panel['rows']; y = np.arange(len(rows)); values = [r[metric] for r in rows]
            colors = ['#1769aa' if r['method'].startswith('flow') else '#d87528' if ('mixture' in r['method'] or r['method']=='mdn') else '#626e78' for r in rows]
            ax.scatter(values, y, c=colors, s=48, zorder=3)
            ax.set_yticks(y, [labels[r['method']] for r in rows] if j == 0 else ['']*len(rows))
            ax.set_ylim(len(rows)-.4, -.7); ax.set_xlim(0, max(values)*1.22)
            ax.grid(axis='x', color='#dce1e6', linewidth=.7); ax.set_axisbelow(True)
            for yy, value, color in zip(y, values, colors):
                ax.annotate(f'{value:.3f}' if i==2 else f'{value:.2f}', (value,yy), xytext=(6,0),
                            textcoords='offset points', va='center', fontsize=9, color=color)
            for side in ('top','right','left'): ax.spines[side].set_visible(False)
            ax.tick_params(axis='y', length=0); ax.set_xlabel(title+' — lower is better')
            if j == 0: ax.set_title(panel['name']+'\n'+panel['sample'], loc='left', fontsize=11, pad=16)
    fig.suptitle('Primary benchmark results: gains and limits', fontsize=18, x=.05, ha='left')
    fig.text(.05,.015, 'Blue: flow candidates. Orange: mixture candidates. Gray: fixed controls. Different model/input routes across datasets.\n'
             'All declared methods are shown. Means are descriptive; failures and uncertainty details remain in the linked source reports.', fontsize=10)
    fig.tight_layout(rect=(0,.055,1,.96), h_pad=3, w_pad=3)
    fig.savefig(ROOT/'campaign-primary-results.svg', metadata={'Date':None})
    fig.savefig(ROOT/'campaign-primary-results.png', dpi=150)
    plt.close(fig)
    lines = ['PRIMARY CAMPAIGN RESULTS', '', 'These are the original frozen comparisons, not a replacement by later development candidates.', '']
    for panel in panels:
        lines += [panel['name'], panel['sample'], 'Method                         Position mm    Shape mm']
        for row in panel['rows']:
            lines.append(f"{labels[row['method']]:30s} {row['position_mm']:10.3f} {row['shape_mm']:11.3f}")
        lines.append('')
    lines += ['Interpretation', 'SoftVT: flow modestly improves same-contact retrieval, but the rigid control is stronger on the matched successful cases.',
              'PhysTwin: deformation improves the rigid control, but flow does not beat nearest retrieval; velocity is stronger for position.',
              'Isaac: the teacher-validation gain did not transfer to recorded motion; archived 25D remains stronger than the new flow.',
              'No global model replacement is justified. Existing best models and all original reports remain preserved.', '', 'Limits'] + result['limitations']
    (ROOT/'campaign-primary-results.txt').write_text('\n'.join(lines)+'\n')


if __name__ == '__main__': main()
