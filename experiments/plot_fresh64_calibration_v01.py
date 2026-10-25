"""Render complete, independently audited scalar aggregates only.

No object selection, raw observations, posterior replay, or random draws.
"""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def render(summary_path, audit_path, output):
    summary_path, audit_path, output = map(Path, (summary_path, audit_path, output))
    s = json.loads(summary_path.read_text())
    a = json.loads(audit_path.read_text())
    if (a.get('integrity_passed') is not True or a.get('comparison_blocked') is not False
            or a.get('summary', {}).get('sha256') != digest(summary_path)
            or s.get('completed') is not True or s.get('comparison_blocked') is not False
            or s.get('objects') != 64 or s.get('qualified_posteriors') != 896
            or a.get('protocol', {}).get('sha256') != s.get('protocol_SHA256')):
        raise ValueError('Complete 64/896 results with exact independent audit required')
    output.mkdir(parents=True, exist_ok=False)
    phase_labels = ['Before probe', 'After normal probe', 'After axial probe']
    colors = {'linear': '#9b5b45', 'nonlinear': '#197c78'}
    fig, axes = plt.subplots(1, 2, figsize=(11, 5), layout='constrained', sharey=True)
    for parameter, ax in enumerate(axes):
        for model, offset in [('linear', -.07), ('nonlinear', .07)]:
            rows = s['primary_calibration'][model]['entries'][parameter::2]
            if len(rows) != 3 or any(r['object_count'] != 64 or r['unresolved_coverage'] for r in rows):
                raise ValueError('Exactly three complete phases per parameter')
            means = np.array([r['coverage'] for r in rows])
            intervals = np.array([r['intervals']['simultaneous_Bonferroni_Clopper_Pearson'] for r in rows])
            ax.errorbar(np.arange(3)+offset, 100*means,
                        yerr=100*np.stack((means-intervals[:, 0], intervals[:, 1]-means)),
                        fmt='o', capsize=4, color=colors[model], label=model.capitalize()+' model')
        ax.axhline(90, color='#333333', linestyle='--', linewidth=1)
        ax.set_xticks(range(3), phase_labels, rotation=12)
        ax.set_ylim(0, 103)
        ax.set_title(['Material stiffness E', 'Fixture stiffness K'][parameter])
        ax.grid(axis='y', alpha=.15)
        ax.spines[['top', 'right']].set_visible(False)
    axes[0].set_ylabel('Objects inside their 90% posterior interval (%)')
    axes[1].legend(loc='lower left', fontsize=9)
    fig.suptitle('Do the uncertainty intervals contain the true stiffness?', fontsize=14)
    fig.supxlabel('64 independent objects; reset 0 only. Whiskers: simultaneous 95% intervals across six entries.\n'
                  'Nonlinear is the primary family; linear is a comparator. Agreement with 90% is not proof of calibration.', fontsize=9)
    fig.savefig(output/'stiffness-coverage.png', dpi=180)
    plt.close(fig)

    bootstrap = s['paired_metrics']['shared_bootstrap']
    if bootstrap['object_clusters'] != 64 or bootstrap['resamples'] != 10000:
        raise ValueError('Predeclared shared object bootstrap required')
    metrics = {r['label']: r for r in bootstrap['entries']}
    fig, axes = plt.subplots(1, 2, figsize=(11, 5), layout='constrained')
    phases = ['prefix', 'normal-r0', 'axial-r0']
    metric_suffixes = ['prefix_RMS_m', 'normal_reset0_RMS_m', 'axial_reset0_RMS_m']
    for model, offset in [('linear', -.07), ('nonlinear', .07)]:
        rows = [metrics[model+'_'+suffix] for suffix in metric_suffixes]
        means = np.array([r['mean'] for r in rows])*1e6
        intervals = np.array([r['percentile_95_interval'] for r in rows])*1e6
        for x, mean, (lo, hi) in zip(np.arange(3)+offset, means, intervals):
            axes[0].plot([x, x], [lo, hi], color=colors[model])
        axes[0].plot(np.arange(3)+offset, means, 'o-', color=colors[model], label=model.capitalize()+' model')
    rows = [metrics['nonlinear_minus_linear_'+phase+'_RMS_m'] for phase in phases]
    for x, row in enumerate(rows):
        lo, hi = np.array(row['percentile_95_interval'])*1e6
        axes[1].plot([x, x], [lo, hi], color=colors['nonlinear'], linewidth=2)
        axes[1].plot(x, row['mean']*1e6, 'o', color=colors['nonlinear'])
    axes[1].axhline(0, color='#333333', linestyle='--', linewidth=1)
    for ax in axes:
        ax.set_xticks(range(3), phase_labels, rotation=12)
        ax.grid(axis='y', alpha=.15)
        ax.spines[['top', 'right']].set_visible(False)
    axes[0].set_ylabel('Mean held-load displacement RMS error (µm)')
    axes[0].set_ylim(bottom=0)
    axes[0].set_title('Four loads withheld from inference')
    axes[0].legend(fontsize=9)
    axes[1].set_ylabel('Nonlinear minus linear error (µm)')
    axes[1].set_title('Paired difference; negative favors nonlinear')
    fig.suptitle('Prediction on held-out forces', fontsize=14)
    fig.supxlabel('64 independent objects; reset 0 only; 95% exploratory paired object-bootstrap intervals.\n'
                  'Static, known support geometry. These fixed probes do not demonstrate an adaptive policy advantage.', fontsize=9)
    fig.savefig(output/'held-load-prediction.png', dpi=180)
    plt.close(fig)
    result = dict(summary_SHA256=digest(summary_path),audit_SHA256=digest(audit_path),
                  protocol_SHA256=s['protocol_SHA256'],raw_data_read=False,
                  figures={p.name:digest(p) for p in output.glob('*.png')},
                  no_population_calibration_pass_rule=True,no_active_policy_claim=True)
    (output/'provenance.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--summary',required=True)
    parser.add_argument('--audit',required=True)
    parser.add_argument('--output',required=True)
    args = parser.parse_args()
    print(json.dumps(render(args.summary,args.audit,args.output)))
