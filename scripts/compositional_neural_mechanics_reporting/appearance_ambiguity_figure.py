"""Reproduce and plot the zero-load appearance/reaction counterexample.

Display crops are identical; metrics still use full calibrated camera images.
"""
import os
for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[name] = '1'
os.environ['MPLBACKEND'] = 'Agg'
import copy
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt
from scripts.compositional_neural_mechanics.scenes import make_model
from scripts.compositional_neural_mechanics.visual_metric_bridge import render_state, calibrated_cameras

ROOT = Path(__file__).resolve().parents[2]
R = ROOT/'reports/compositional-neural-mechanics-v16'


def main():
    torch.set_num_threads(1)
    source = R/'full-continuous-inference-pilot-world0.json'
    scene = copy.deepcopy(json.loads(source.read_text())['scene'])
    scene['gravity'] = [0., 0., 0.]
    model = make_model(scene, n=2)
    pose = np.zeros((2, 6))
    pose[1, 0] = .004
    model.set_controls(pose)
    state = model.solve(force_tolerance_N=1e-10)
    u = state['displacement_local_m']
    scaled = copy.deepcopy(scene)
    scaled['bulk_latent'][0] += np.log(2)
    for descriptor in scaled['interface_latents']:
        descriptor[0] += np.log(2)
    other = make_model(scaled, n=2)
    other.set_controls(pose)
    residual = float(np.linalg.norm(other.evaluate(u, False)['residual']))
    if residual > 1e-8:
        raise ValueError('Scaled equilibrium does not pass residual gate')
    cameras = calibrated_cameras(model.geometry)
    views = [render_state(m, u, cameras) for m in (model, other)]
    for name in cameras:
        for channel in ('rgb', 'depth'):
            if not np.array_equal(views[0][name][channel], views[1][name][channel]):
                raise ValueError('Appearance ambiguity diagnostic failed')
    reactions = [m.reaction_wrenches(state)['forces_N'] for m in (model, other)]
    if not np.allclose(reactions[1], 2*reactions[0], rtol=1e-12, atol=1e-12):
        raise ValueError('Exact reaction scaling failed')
    camera = 'oblique'
    depth = views[0][camera]['depth']
    # Renderer uses non-finite depth for background; bound the visible object.
    ys, xs = np.nonzero(np.isfinite(depth))
    box = (max(0, ys.min()-12), min(depth.shape[0], ys.max()+13),
           max(0, xs.min()-12), min(depth.shape[1], xs.max()+13))
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.3), gridspec_kw={'width_ratios':[1, 1, 1.1]})
    for index in (0, 1):
        axes[index].imshow(views[index][camera]['rgb'][box[0]:box[1], box[2]:box[3]])
        axes[index].set_title(('Original stiffness', 'Material + supports: 2x stiffness')[index], fontsize=10)
        axes[index].axis('off')
    norms = [float(np.linalg.norm(f)) for f in reactions]
    axes[2].bar(['Original', '2x stiffness'], norms, color=['#377e99', '#cc7143'], width=.55)
    axes[2].set_ylabel('Norm of tool reaction vector (N)')
    axes[2].set_ylim(0, max(norms)*1.25)
    for index, value in enumerate(norms):
        axes[2].text(index, value+.025, f'{value:.3f}', ha='center', fontsize=10)
    axes[2].spines[['top', 'right']].set_visible(False)
    fig.suptitle('Identical appearance does not imply the same physical response', fontsize=13)
    fig.text(.5, .025, 'Zero gravity; prescribed positions; same equilibrium. All three full RGB/depth views are identical.',
             ha='center', fontsize=9)
    fig.tight_layout(rect=(0, .08, 1, .9))
    output = R/'appearance-physical-ambiguity.png'
    fig.savefig(output, dpi=180)
    plt.close(fig)
    metadata = dict(scope=__doc__, source=str(source.relative_to(ROOT)),
                    source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                    renderer_source_sha256=hashlib.sha256((ROOT/'scripts/compositional_neural_mechanics/visual_metric_bridge.py').read_bytes()).hexdigest(),
                    script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    stiffness_multiplier=2., gravity=[0, 0, 0], scaled_residual_norm_N=residual,
                    reaction_norms_N=norms, images_bit_identical=True, camera=camera,
                    common_display_crop_yxyx=list(map(int, box)),
                    image_sha256=hashlib.sha256(output.read_bytes()).hexdigest())
    output.with_suffix('.json').write_text(json.dumps(metadata, indent=2)+'\n')


if __name__ == '__main__':
    main()
