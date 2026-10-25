"""Local scientific figure after the frozen balanced screen's terminal audit.

No simulation, policy selection, random draws or uploads. All 54 support pairs
remain in the display; their differences are deterministic grid diagnostics.
"""
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
LOCAL = ROOT / 'runs/operator-dev/fixture-balanced-decision-screen-v01'
MEDIA = ROOT / 'runs/operator-dev/fixture-balanced-development-media-v01'


def render():
    from experiments import run_fixture_balanced_decision_screen_v01 as run
    review = run.read(LOCAL / 'independent-terminal-review.json')
    summary = review['summary']
    if not (summary['integrity_passed'] and summary['execution_clean']
            and summary['completed_case_records'] == 36):
        raise ValueError('Complete independently audited screen before plotting')
    for key in ('protocol', 'release', 'execution', 'root_closure'):
        if run.record(ROOT / review[key]['path']) != review[key]:
            raise ValueError('Exact audited artifact bytes')
    protocol = run.validate(review['protocol']['sha256'])
    builder = run.read(LOCAL / 'builder-result.json')
    run.verify_records(builder)
    with np.load(ROOT / builder['held_response_record']['path'], allow_pickle=False) as z:
        response = dict(zip(z['case_index'].tolist(), z['responses']))
    prefix = {}
    for item in builder['prefixes']:
        with np.load(ROOT / item['record']['path'], allow_pickle=False) as z:
            prefix[item['index']] = z['observed_prefix']
    x, y = [], []
    cases = protocol['worklist']
    for i, a in enumerate(cases):
        for b in cases[i + 1:]:
            if (a['E_Pa'], a['K_N_m']) != (b['E_Pa'], b['K_N_m']):
                continue
            x.append(np.sqrt(np.mean((prefix[a['index']] - prefix[b['index']]) ** 2)) * 1e6)
            y.append(np.sqrt(np.mean((response[a['index']] - response[b['index']]) ** 2)) * 1e6)
    if len(x) != 54:
        raise ValueError('Every fixed material/support pair retained')
    old = run.read(ROOT / 'runs/operator-dev/fixture-Gaussian-linear-screen-v01/held-target-energy-concept-compact.json')
    old_share = old['by_mesh']['6']['down_share']
    shares = np.array([[old_share, 1 - old_share],
                       builder['closed36_target_separation']['energy_share']]) * 100

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), layout='constrained')
    for i, color in enumerate(('#376ab2', '#d98118')):
        axes[0].bar(np.arange(2), shares[:, i], bottom=shares[:, :i].sum(axis=1), color=color)
        for j in range(2):
            axes[0].text(j, shares[j, :i].sum() + shares[j, i] / 2,
                         f'{shares[j, i]:.1f}%', ha='center', va='center', color='white')
    axes[0].set(xticks=[0, 1], xticklabels=['Earlier: down / axial', 'New: two oblique loads'],
                ylim=(0, 100), ylabel='Share of squared support-sensitive displacement (%)',
                title='Measured contribution of each prediction load')
    axes[0].text(.5, -.2, 'Equal weighting of nine material/support-stiffness grid points.\nThis is a signal diagnostic, not posterior risk.',
                 transform=axes[0].transAxes, ha='center', fontsize=9)
    axes[1].scatter(x, y, s=30, color='#376ab2', alpha=.7)
    axes[1].axvline(20, color='#657889', linestyle='--', label='Single-channel noise SD: 20 µm')
    axes[1].set(xlim=(0, None), ylim=(0, None), xlabel='Initial observation difference (RMS, µm)',
                ylabel='Future response difference (RMS, µm)', title='Same material; different hidden support\nAll 54 pairs retained')
    axes[1].grid(alpha=.15)
    axes[1].legend(fontsize=8, frameon=False)
    fig.suptitle('Balanced-load development task: support sensitivity, not active-policy advantage')
    MEDIA.mkdir(parents=True, exist_ok=True)
    path = MEDIA / 'future-load-support-sensitivity.png'
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return str(path)


if __name__ == '__main__':
    print(render())
