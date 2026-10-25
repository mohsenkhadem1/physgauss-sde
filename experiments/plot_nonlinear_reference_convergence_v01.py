"""Plot published scalar controls; observations and posterior arrays stay local."""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def render(output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    rows, sources = [], []
    for version in (2, 3):
        source = ROOT / f'reports/surface-nonlinear-posterior-cost-v0{version}/summary.json'
        document = json.loads(source.read_text())
        checks = document['numerical_results']['9']
        checks = checks['all_checks' if version == 2 else 'all_engine_checks']
        adaptive = checks['original_adaptive_checks']
        witness = checks['strict_absolute_checks']['final_axis_witnesses']['inner']
        errors = witness['actual_absolute_errors_m']
        rows.append([adaptive['maximum_CDF_error'] / .005,
                     adaptive['log_evidence_difference'] / .01,
                     max(errors.values()) / 2e-6])
        sources.append(dict(path=str(source.relative_to(ROOT)),
                            sha256=hashlib.sha256(source.read_bytes()).hexdigest()))
    values = np.asarray(rows)
    if not np.isfinite(values).all() or np.any(values <= 0):
        raise ValueError('Positive finite measured errors required')
    fig, ax = plt.subplots(figsize=(10, 5.4), layout='constrained')
    x = np.arange(3)
    for i, (label, color) in enumerate((('Gaussian-only integration (v02)', '#b44a37'),
                                       ('Gaussian + prior mixture (v03)', '#197c78'))):
        bars = ax.bar(x + (i - .5) * .34, values[i], .32, label=label, color=color)
        for bar, value in zip(bars, values[i]):
            ax.annotate(f'{value:.3g}× limit', (bar.get_x() + bar.get_width()/2, value),
                        xytext=(0, 5), textcoords='offset points', ha='center', fontsize=9)
    ax.axhline(1, color='#333333', linestyle='--', linewidth=1.2, label='Predeclared limit')
    ax.set_yscale('log')
    ax.set_ylim(.0001, 250)
    ax.set_ylabel('Error / its numerical limit (log scale; lower is better)')
    ax.set_xticks(x, ['Independent CDF\ndiscrepancy', 'Independent log-evidence\ndiscrepancy',
                     'Prediction mean/SD\nfinal inner refinement'])
    ax.set_title('Same observation, unchanged physical model and accuracy limits', fontsize=13, pad=14)
    ax.grid(axis='y', alpha=.15)
    ax.set_axisbelow(True)
    ax.spines[['top', 'right']].set_visible(False)
    ax.legend(loc='lower left', fontsize=9)
    fig.supxlabel('One synthetic numerical control, nine-point atlas; population calibration is still pending.',
                  fontsize=10)
    figure = output / 'numerical-reference-repair.png'
    fig.savefig(figure, dpi=180)
    plt.close(fig)
    metadata = dict(sources=sources,normalized_errors=values.tolist(),
                    column_order=['independent_CDF','independent_log_evidence','inner_prediction'],
                    figure_SHA256=hashlib.sha256(figure.read_bytes()).hexdigest(),
                    population_calibration_claim=False,raw_data_read=False)
    (output/'provenance.json').write_text(json.dumps(metadata,indent=2)+'\n')
    return str(figure)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    print(render(parser.parse_args().output))
