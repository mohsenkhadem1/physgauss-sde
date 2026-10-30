"""Draw the actual position-based identification and prediction workflow."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT = Path(__file__).resolve().parents[2]


def main():
    fig, ax = plt.subplots(figsize=(13, 6.3))
    fig.patch.set_facecolor('white')
    ax.set(xlim=(0, 13), ylim=(0, 6.3))
    ax.axis('off')

    def box(x, y, w, h, title, body, color='#edf5f5'):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.03,rounding_size=0.12',
                     facecolor=color, edgecolor='#66828b', linewidth=1.1))
        ax.text(x+.18, y+h-.25, title, ha='left', va='top', fontsize=12, weight='bold', color='#173c50')
        ax.text(x+.18, y+h-.73, body, ha='left', va='top', fontsize=10.2, linespacing=1.5, color='#253f49')

    def arrow(a, b, dashed=False):
        ax.add_patch(FancyArrowPatch(a, b, arrowstyle='-|>', mutation_scale=14,
                     linewidth=1.3, linestyle='--' if dashed else '-', color='#55727d'))

    ax.text(.2, 6.03, 'Predict motion from positions and a coupled physical scene',
            fontsize=18, weight='bold', color='#173c50')
    box(.2, 2.5, 2.75, 2.25, '1  Observe motion',
        'Surface positions during\nsqueeze, shear and release\n\nKnown tool poses and\ninitial object geometry')
    box(3.4, 2.5, 2.75, 2.25, '2  Fit parameters',
        'Simulate a fixed grid of\nstiffness and friction values\n\nSelect the smallest context\nposition error')
    box(6.6, 2.5, 3.05, 2.25, '3  Simulate contact',
        'Deformable: nonlinear FEM\nRigid: pose and momentum\nContact: two-way forces\n\nGravity + commanded motion')
    box(10.1, 2.5, 2.7, 2.25, '4  Predict motion',
        'Object positions\nShape changes\nFree rigid-body motion\n\nFor a new interaction')
    arrow((2.97, 3.65), (3.36, 3.65))
    arrow((6.17, 3.65), (6.56, 3.65))
    arrow((9.67, 3.65), (10.06, 3.65))
    box(6.6, .8, 3.05, 1.2, 'Known future inputs',
        'Tool poses and geometry\nGravity and initial state', '#f7f1e7')
    arrow((8.12, 2.02), (8.12, 2.46))
    box(10.1, .8, 2.7, 1.2, 'Evaluation only',
        'Reserved future positions\nReference quality checks', '#f0f0f3')
    arrow((11.45, 2.46), (11.45, 2.04), dashed=True)
    ax.text(.23, 1.6, 'Forces are computed by the model.\nFuture target positions are used only for scoring.\nGrid point estimates are not calibrated uncertainty.',
            fontsize=10.5, linespacing=1.7, color='#435b65', va='top')
    ax.text(.23, .25, 'Controlled simulation study: known geometry and gravity; unknown stiffness and friction. '
            'No image-to-geometry or full PhysTwin pipeline is implied.', fontsize=9, color='#526570')
    directory = ROOT/'reports/coupled-surface-v13'
    for suffix in ('png', 'svg'):
        fig.savefig(directory/f'method-overview.{suffix}', dpi=180, bbox_inches='tight', facecolor='white')
    plt.close(fig)


if __name__ == '__main__':
    main()
