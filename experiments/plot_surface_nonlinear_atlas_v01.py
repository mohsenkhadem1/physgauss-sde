"""Plot aggregate finite atlas checks; raw data and media stay local."""
import hashlib
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]


def main():
    source = ROOT/'runs/operator-dev/surface-full-atlas-feasibility-v01/summary-LOCAL.json'
    payload = source.read_bytes()
    result = json.loads(payload)
    if not result['completed'] or not result['qualified'] or result['physical_passed'] != 1130:
        raise ValueError('Complete qualified full catalog required')
    levels = [3, 5, 9]
    response = [max(c['levels'][i]['maximum_coordinate_error_m']
                    for c in result['perforce_validation'].values())*1e6 for i in range(3)]
    single = [max(r['maximum_near_tested_peak_loglikelihood_error']
                  for c in result['perforce_validation'].values()
                  for r in c['levels'][i]['likelihood_cases']) for i in range(3)]
    joint = [max(r['maximum_near_joint_tested_peak_loglikelihood_error']
                 for r in result['joint_validation_levels'][i]['cases']) for i in range(3)]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.7))
    for ax, values, threshold, title, label in zip(axes,
            [response, single, joint], [.1, .01, .01],
            ['Displacement', 'Single observation: 24 channels', 'Prefix + poke: 48 channels'],
            ['Worst coordinate error (µm)', 'Worst log-likelihood error', 'Worst log-likelihood error']):
        ax.plot(levels, values, color='#185e82', marker='o', linewidth=2)
        ax.axhline(threshold, color='#a34332', linestyle='--', label='Declared limit')
        for n, value in zip(levels, values):
            ax.annotate(f'{value:.3g}', (n, value), xytext=(0, 7),
                        textcoords='offset points', ha='center', fontsize=9)
        ax.set_yscale('log');ax.set_xticks(levels)
        ax.set_xlabel('Grid points per parameter axis')
        ax.set_ylabel(label);ax.set_title(title, fontsize=11)
        ax.grid(alpha=.2);ax.legend(fontsize=8, loc='lower left')
        low, high = ax.get_ylim();ax.set_ylim(low/2, high*2)
    fig.suptitle('Nonlinear correction: reserved-point numerical validation', fontsize=14)
    fig.text(.5, .02,
             'Worst across 10 forces and 32 reserved parameter points; 72 joint likelihood cases.\n'
             'Likelihood errors use points within 25 log units of the tested reference peak.\n'
             'Same n4 nonlinear equations as reference. This does not measure mesh error or uncertainty calibration.',
             ha='center', fontsize=9)
    fig.tight_layout(rect=(0, .16, 1, .94))
    output = ROOT/'runs/operator-dev/surface-full-atlas-media-v02'
    output.mkdir(exist_ok=False)
    fig.savefig(output/'atlas-validation.png', dpi=170)
    (output/'provenance.json').write_text(json.dumps(dict(
        source_SHA256=hashlib.sha256(payload).hexdigest(),
        plot_source_SHA256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        media_LOCAL_only=True), indent=2)+'\n')
    plt.close(fig)


if __name__ == '__main__':
    main()
