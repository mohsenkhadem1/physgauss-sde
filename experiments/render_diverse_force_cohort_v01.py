"""Plot terminal scalar cohort diagnostics. No inference or simulation calls."""
from pathlib import Path
import json
import hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT/'runs/gaussian-campaign-v01/diverse-force-cohort-v01'
OUT = ROOT/'runs/operator-dev/diverse-force-cohort-media-v01'


def render():
    summary = json.loads((RUN/'summary.json').read_text())
    rows = [json.loads((RUN/f'object-{i:02d}/summary.json').read_text()) for i in range(12)]
    if not summary['completed'] or len(rows) != 12 or not all(r['completed'] for r in rows):
        raise ValueError('All twelve terminal objects required, including failed checks')
    OUT.mkdir(parents=True, exist_ok=False)
    x = np.arange(1, 13)
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), layout='constrained')
    for name, label, color in [('nominal_prior_center', 'Nominal material', '#8b93a0'),
                                ('physical_mean_plugin', 'Mean physical parameters', '#b86a35'),
                                ('posterior_mean', 'Posterior forecast', '#207b78')]:
        axes[0, 0].plot(x, [r['baseline_RMSE_m'][name]*1e6 for r in rows], 'o-',
                        color=color, label=label, markersize=4)
    axes[0, 0].set_title('Before probing: unseen-force forecast error')
    axes[0, 0].set_ylabel('RMSE (µm)')
    axes[0, 0].legend(fontsize=8)
    policies = [('target_variance', 'Target uncertainty', '#207b78'),
                ('fixed_first', 'Fixed poke', '#b86a35'),
                ('local_Fisher', 'Local Fisher', '#725ca1'),
                ('uniform_random_expected', 'Uniform random expectation', '#8b93a0')]
    for name, label, color in policies:
        axes[0, 1].plot(x, [np.mean(r['policy_RMSE_m'][name])*1e6 for r in rows], 'o-',
                        color=color, label=label, markersize=4, alpha=.85)
    axes[0, 1].set_title('After one poke: mean error over 16 noise trials')
    axes[0, 1].set_ylabel('RMSE (µm)')
    axes[0, 1].legend(fontsize=8)
    for shift, (name, label, color) in zip([-.15, 0., .15], policies[:3]):
        axes[1, 0].scatter(x+shift, [r['choices'][name] for r in rows], color=color,
                           label=label, s=35, alpha=.8)
    axes[1, 0].set(yticks=range(6), ylim=(-.5, 5.5), ylabel='Chosen poke index',
                   title='Do different objects lead to different choices?')
    axes[1, 0].legend(fontsize=8)
    checks = [('parameter_precision_passed', 'Parameters'),
              ('function_precision_passed', 'Fixture functions'),
              ('predictive_precision_passed', 'Prediction precision'),
              ('acquisition_stable', 'Action stability'),
              ('reference', 'Independent references'), ('passed', 'All checks')]
    matrix = np.array([[bool(all(c['passed'] for c in r['SMC_comparisons']+r['SMC_pairwise'])
                                  and r['logZ_range'] <= .05) if key == 'reference' else bool(r[key])
                        for r in rows] for key, _ in checks], dtype=int)
    axes[1, 1].imshow(matrix, aspect='auto', interpolation='nearest',
                       cmap=ListedColormap(['#c85d58', '#388b72']), vmin=0, vmax=1)
    axes[1, 1].set(xticks=range(12), xticklabels=x,
                   yticks=range(len(checks)), yticklabels=[label for _, label in checks],
                   xlabel='Object', title='Numerical checks: green pass, red fail')
    for ax in axes.flat[:3]:
        ax.set(xticks=x, xlabel='Object', xlim=(.5, 12.5))
        ax.grid(alpha=.15)
    fig.suptitle(f'Twelve preselected finite-model objects · {summary["numerically_passed_objects"]}/12 pass all numerical checks\n'
                 'Three force programs at three future frames; compliant support center varies along x only', fontsize=12)
    fig.savefig(OUT/'cohort-results.png', dpi=150)
    plt.close(fig)
    proof = dict(input_summary_SHA256=hashlib.sha256((RUN/'summary.json').read_bytes()).hexdigest(),
                 object_summaries_SHA256={str(i):hashlib.sha256((RUN/f'object-{i:02d}/summary.json').read_bytes()).hexdigest()
                                          for i in range(12)},
                 all_objects_retained=True, no_new_inference=True, raw_media_local=True,
                 metric='135 selected future coordinates, distinct from the earlier1620 metric',
                 population_calibration_claim=False)
    (OUT/'render-provenance.json').write_text(json.dumps(proof, indent=2)+'\n')
    print(str((OUT/'cohort-results.png').relative_to(ROOT)))


if __name__ == '__main__':
    render()
