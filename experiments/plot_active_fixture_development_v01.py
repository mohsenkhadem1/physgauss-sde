"""Local scientific illustration from CLOSED, independently audited development.

No simulation, inference, selection, random draws or media uploads.
"""
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
LOCAL = ROOT / 'runs/operator-dev/active-fixture-development-media-v01'


def render():
    from experiments import audit_fixture_gaussian_linear_screen_v01 as audit
    _, _, result = audit.metadata_barrier()
    review = json.loads((audit.stage.LOCAL / 'independent-review.json').read_text())
    if review.get('audit_passed') is not True or review['source_result'] != audit.stage.record(audit.stage.LOCAL / 'result-LOCAL.json'):
        raise ValueError('Completed independent audit before illustration')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    fig = plt.figure(figsize=(12, 5), layout='constrained')
    ax = fig.add_subplot(121, projection='3d')
    vertices = np.array([[x, y, z] for x in (0, .36) for y in (0, .08) for z in (0, .08)])
    sides = [[0, 1, 3, 2], [4, 5, 7, 6], [0, 1, 5, 4], [2, 3, 7, 6], [0, 2, 6, 4], [1, 3, 7, 5]]
    ax.add_collection3d(Poly3DCollection([vertices[s] for s in sides], facecolor='#b5c2ce', edgecolor='#657889', alpha=.12))
    ax.add_collection3d(Poly3DCollection([vertices[s] for s in sides[:2]], facecolor='#376ab2', edgecolor='#376ab2', alpha=.4))
    supports = np.array([s['center_m'] for s in audit.stage.mechanics.bank.LAYOUTS])
    ax.scatter(*supports.T, color='#d98118', marker='s', s=65, label='One of four support locations')
    for x, y, _ in supports:
        for sign in (-1, 1):
            d = np.array([0, sign, -1]) / np.sqrt(2)
            p = np.array([x, y, .08]) - .035 * d
            ax.quiver(*p, *(.035*d), color='#984b9b', arrow_length_ratio=.25, linewidth=1.5)
    markers = audit.stage.mechanics.bank.original.PREFIX_POINTS
    ax.scatter(*markers.T, color='#171d24', s=14, label='Four initial measurements')
    ax.text(.003, .04, .105, 'Fixed end', color='#376ab2', fontsize=9)
    ax.text(.315, .04, .105, 'Fixed end', color='#376ab2', fontsize=9)
    ax.set(xlim=(-.015, .375), ylim=(-.015, .095), zlim=(-.01, .135), xlabel='x (m)', ylabel='y (m)', zlabel='z (m)')
    ax.set_box_aspect((3.6, .8, 1.45))
    ax.set_xticks([0, .12, .24, .36])
    ax.set_yticks([0, .04, .08])
    ax.set_zticks([0, .04, .08, .12])
    ax.view_init(elev=23, azim=-59)
    ax.set_title('3D block: uncertain material and support\nFour poke locations, two force directions each', fontsize=11)
    ax.legend(loc='lower left', fontsize=8, frameon=False)
    chart = fig.add_subplot(122)
    rows = [s for s in result['layout_separation'] if s['n'] == 6]
    x = np.array([s['prefix4_RMS_m'] for s in rows]) * 1e6
    y = np.array([s['held48_RMS_m'] for s in rows]) * 1e6
    chart.axvspan(0, 20, color='#eef2f6', zorder=0)
    chart.scatter(x, y, c='#b45f16', s=30, alpha=.7, edgecolor='white', linewidth=.4)
    chart.axvline(20, color='#596f85', linestyle='--', label='Single-channel noise SD: 20 µm')
    chart.set(xlabel='Initial-measurement difference (RMS, µm)', ylabel='Later held-load difference (RMS, µm)', title='Changing the hidden support can matter\n54 pairs on a fixed development grid')
    chart.set_xlim(left=0)
    chart.set_ylim(bottom=0)
    chart.grid(alpha=.15)
    chart.legend(fontsize=8, frameon=False)
    fig.suptitle('Scenario illustration and linear development diagnostic — no active-policy result', fontsize=12)
    LOCAL.mkdir(parents=True, exist_ok=True)
    path = LOCAL / 'scenario-and-support-effect.png'
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return str(path)


if __name__ == '__main__':
    print(render())
