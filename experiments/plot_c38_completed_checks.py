"""Plot frozen completed checks; no simulation, training or live-result reads."""
from pathlib import Path
import hashlib
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]


def main():
    output = ROOT / 'runs/operator-dev/c38-completed-figure-v01'
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('Preserve completed plot output')
    output.mkdir(parents=True, exist_ok=True)
    inventories = ['runs/campaign-c38g-paths.json', 'runs/campaign-c38h-paths.json',
                   'runs/operator-dev/p2-torch-warp-hvp-benchmark-v02/publication-inventory.json']
    allowed = {}
    for name in inventories:
        for row in json.loads((ROOT / name).read_text())['files']:
            if row['path'] in allowed and allowed[row['path']] != row['sha256']:
                raise ValueError('Conflicting frozen inputs')
            allowed[row['path']] = row['sha256']
    inputs = []

    def read(name):
        data = (ROOT / name).read_bytes()
        sha = hashlib.sha256(data).hexdigest()
        if sha != allowed[name]:
            raise ValueError('Changed frozen plot input')
        inputs.append(dict(path=name, sha256=sha))
        return json.loads(data)

    uniform = read('runs/operator-dev/c38-static-prequalification-v01/completed-summary.json')
    graded = read('runs/operator-dev/c38-static-graded-refinement-v03/completed-summary.json')
    contact = read('runs/operator-dev/p2-warp-surface-q16-v01/result-summary.json')
    speed = read('runs/operator-dev/p2-torch-warp-hvp-benchmark-v02/result-summary.json')
    plt.rcParams.update({'font.size': 11, 'axes.spines.top': False,
                         'axes.spines.right': False, 'savefig.dpi': 170})
    fig = plt.figure(figsize=(14, 8))
    grid = fig.add_gridspec(2, 2, width_ratios=[1.5, 1], hspace=.65, wspace=.48)
    ax = fig.add_subplot(grid[:, 0])
    rows = uniform['rows']
    labels = []
    y = np.arange(len(rows))
    for row in rows:
        c = row['case']
        labels.append(f"{c['young_Pa']/1000:.1f} kPa, " +
                      ('ends' if c['support'] == 'both_ends' else 'bottom') +
                      f", {c['Fz_N']:+.0f} N")
    values = [r['conditional_actual_h_indicators_m'][-1] * 1e6 for r in rows]
    ax.barh(y, values, height=.64, color=['#87a69b' if v <= 20 else '#cc8b79' for v in values],
            label='Original uniform mesh pair')
    for row in graded['rows']:
        index = next(i for i, old in enumerate(rows) if old['case']['id'] == row['case']['id'])
        value = row['conditional_displacement_indicators_m'][-1] * 1e6
        ax.scatter(value, index, s=75, color='#195ca4', marker='D', zorder=4,
                   label='New graded 32 to 40' if row is graded['rows'][0] else None)
        ax.text(value + 1, index - .32, f'{value:.1f}', color='#195ca4', fontsize=9)
    ax.axvline(20, color='#333333', linestyle='--', linewidth=1, label='20 µm limit')
    ax.set(yticks=y, yticklabels=labels, xlabel='Conditional displacement indicator (µm)',
           title='Static displacement: 7 of 14 original cases fail')
    ax.invert_yaxis()
    ax.legend(fontsize=9, loc='lower right')
    ax.text(0, -.13, 'Two refined cases pass. Remaining family checks are pending.\n'
            'Indicators assume order ≥ 1; peak stress remains unqualified.',
            transform=ax.transAxes, fontsize=10, va='top')

    ax = fig.add_subplot(grid[0, 1])
    pair = contact['pair']
    ratios = [pair['maximum_allnode_position_difference_m'] / 20e-6,
              pair['maximum_fixedquery_position_difference_m'] / 20e-6,
              pair['maximum_allnode_velocity_difference_m_s'] / .0001]
    ax.bar(['Node position', 'Query position', 'Velocity'], ratios, color='#bb5e49')
    ax.set_yscale('log')
    ax.axhline(1, color='#333333', linestyle='--', linewidth=1)
    for i, value in enumerate(ratios):
        ax.text(i, value * 1.25, f'{value:.1f}×', ha='center', fontsize=10)
    ax.set(ylabel='Difference / unchanged limit', ylim=(.5, 12000),
           title='Contact Q8 to Q16: accuracy fails')
    ax.tick_params(axis='x', labelsize=9)
    ax.text(0, -.25, 'Conservation passes, but does not establish field accuracy.',
            transform=ax.transAxes, fontsize=9)

    ax = fig.add_subplot(grid[1, 1])
    seconds = [r['linear_s'] for r in speed['cases']]
    ax.barh(['Original Torch', 'New Warp'], seconds, color=['#8b929a', '#195ca4'])
    for i, value in enumerate(seconds):
        ax.text(value + 2, i, f'{value:.2f} s', va='center', fontsize=10)
    ax.set(xlim=(0, 190), xlabel='Inclusive CG solve time (seconds)',
           title='Same linear problem: corrections agree')
    ax.invert_yaxis()
    ax.text(0, -.35, 'One paired 165,888-tet control; JIT shown separately in report.\n'
            'No full nonlinear or trajectory speed claim.',
            transform=ax.transAxes, fontsize=9)
    fig.suptitle('C38 completed checks — mechanics and numerical solver evidence', fontsize=15, y=.99)
    fig.subplots_adjust(left=.20, right=.96, bottom=.17, top=.91)
    target = output / 'completed-checks.png'
    fig.savefig(target, facecolor='white')
    plt.close(fig)
    receipt = dict(inputs=inputs, source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                   image_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
                   image_local_only=True, live_results_read=False, campaign_complete=False)
    (output / 'receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(dict(image=str(target.relative_to(ROOT)), inputs=len(inputs))))


if __name__ == '__main__':
    main()
