"""Local scientific plots/video of the frozen manufactured force challenge.

No training, inference or action selection. Media and physical arrays stay local.
The video shows future forecasts before the extra poke, with explicit 10x
displacement magnification and slow playback. Truth uses the independent solve.
"""
from pathlib import Path
import hashlib
import json
import mmap
import subprocess

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
import imageio_ffmpeg

from experiments import campaign_compliant_force_challenge_v01 as helper

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'runs/gaussian-campaign-v01/compliant-force-challenge-v01'
OUT = ROOT / 'runs/operator-dev/compliant-force-challenge-media-v01'


def predictive_mean():
    a = np.load(RUN / 'MH-target.npy', mmap_mode='r', allow_pickle=False)
    if a.shape != (512000, 1620):
        raise ValueError('Complete frozen target layout required')
    total = np.zeros(1620)
    for first in range(0, len(a), 1024):
        total += np.asarray(a[first:first+1024]).sum(0)
        a._mmap.madvise(mmap.MADV_DONTNEED)
    return (total / len(a)).reshape(3, 36, 5, 3)


def render():
    summary = json.loads((RUN / 'summary.json').read_text())
    if not summary['completed']:
        raise ValueError('Terminal challenge required')
    OUT.mkdir(parents=True, exist_ok=False)
    mean = predictive_mean()
    programs = helper.force_programs()
    truth = np.stack([helper.independent_numpy_forward(helper.original.TRUTH, force)[25:61]
                      for force in programs.values()])
    np.savez_compressed(OUT / 'video-fields-local.npz', prediction_m=mean, actual_m=truth)
    labels = {'nominal_prior_center': 'Nominal material', 'posterior_MAP': 'Best-fit parameters',
              'posterior_mean_physical_plugin': 'Mean parameters',
              'posterior_predictive_mean': 'Full posterior forecast',
              'hold_posterior_mean_A_frame24': 'Hold estimated last state',
              'zero_displacement': 'No displacement'}
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), layout='constrained')
    values = summary['prediction_RMSE_m']
    keys = list(labels)
    axes[0].barh([labels[k] for k in keys], [values[k]*1e6 for k in keys],
                 color=['#2c9686' if k == 'posterior_predictive_mean' else '#6885ac' for k in keys])
    axes[0].set_xscale('log')
    axes[0].set_xlabel('Future displacement RMSE (µm, logarithmic scale)')
    axes[0].set_title('Three unseen force programs, before poking')
    axes[0].invert_yaxis()
    names = list(summary['policy_mean_RMSE_m']) + ['uniform_random_expected']
    policy_labels = {'target_variance': 'Target uncertainty', 'fixed_first': 'Fixed action',
                     'seeded_random': 'Seeded random action', 'local_Fisher': 'Local Fisher',
                     'parameter_information': 'Parameter uncertainty',
                     'uniform_random_expected': 'Uniform random expectation'}
    post = summary['policy_mean_RMSE_m']
    post = {**post, 'uniform_random_expected': summary['uniform_random_expected_mean_RMSE_m']}
    axes[1].barh([policy_labels[k] for k in names], [post[k]*1e6 for k in names],
                 color=['#2c9686' if k == 'target_variance' else '#6885ac' for k in names])
    axes[1].axvline(values['posterior_predictive_mean']*1e6, color='#d17335', linestyle='--',
                    label='Before poke')
    axes[1].set_xlabel('Mean future displacement RMSE (µm)')
    axes[1].set_title('After one poke, 16 paired noise trials')
    axes[1].invert_yaxis()
    axes[1].legend(fontsize=8)
    status = 'PASS' if summary['reference_agreement_passed'] else 'FAIL: exploratory values only'
    fig.suptitle('One manufactured object · exact finite model · no population calibration\n'
                 f'Numerical reference agreement: {status}', fontsize=12)
    fig.savefig(OUT / 'prediction-and-probing.png', dpi=150)
    plt.close(fig)

    X, _, faces = helper.original.tiny_geometry()
    edges = sorted({tuple(sorted((int(face[i]), int(face[(i+1) % 3]))))
                    for face in faces for i in range(3)})
    fig = plt.figure(figsize=(12, 4.4), dpi=100)
    axes = [fig.add_subplot(1, 3, k+1, projection='3d') for k in range(3)]
    gain = 10.
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    width, height = fig.canvas.get_width_height()
    video = OUT / 'actual-versus-predicted-future.mp4'
    command = [ffmpeg, '-v', 'error', '-y', '-f', 'rawvideo', '-vcodec', 'rawvideo',
               '-pix_fmt', 'rgb24', '-s', f'{width}x{height}', '-r', '12', '-i', '-',
               '-an', '-vcodec', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '20', str(video)]
    process = subprocess.Popen(command, stdin=subprocess.PIPE)
    try:
        for frame in range(36):
            for k, (name, ax) in enumerate(zip(programs, axes)):
                ax.clear()
                actual = X + gain*truth[k, frame]
                predicted = X + gain*mean[k, frame]
                ax.add_collection3d(Poly3DCollection(actual[faces], facecolor='#e69a58',
                                                     edgecolor='#a25c27', alpha=.25, linewidth=.7))
                for a, b in edges:
                    ax.plot(*predicted[[a, b]].T, color='#1763ad', linewidth=1.7)
                ax.set(xlim=(-.02, .14), ylim=(-.02, .14), zlim=(-.02, .14),
                       xlabel='x (m)', ylabel='y (m)', zlabel='z (m)')
                ax.set_box_aspect((1, 1, 1))
                ax.view_init(elev=22, azim=25+35*frame/35)
                ax.set_title(name.replace('_', ' ').title(), fontsize=11)
                ax.tick_params(labelsize=7)
            fig.suptitle(f'Actual: orange surface | Forecast: blue edges | t={(25+frame)/120:.3f} s\n'
                         'Future only, before extra poke · displacement ×10 · slow playback\n'
                         f'Finite linear toy model · numerical reference agreement: {status}',
                         fontsize=11)
            fig.subplots_adjust(left=.01, right=.98, bottom=.08, top=.78, wspace=.05)
            fig.canvas.draw()
            rgb = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
            process.stdin.write(rgb.tobytes())
        process.stdin.close()
        if process.wait(timeout=60) != 0:
            raise RuntimeError('Video encoder failed')
    except BaseException:
        process.kill()
        process.wait()
        raise
    finally:
        plt.close(fig)
    proof = dict(input_summary_sha256=hashlib.sha256((RUN/'summary.json').read_bytes()).hexdigest(),
                 forecasts='Full frozen posterior mean, before the extra poke', frames=[25, 60],
                 actual='Independent original-assembly 30x30 NumPy backward-Euler solve',
                 displacement_magnification=gain, playback_fps=12, simulation_hz=120,
                 raw_fields_and_media_local=True, no_new_inference=True,
                 numerical_gates_passed=summary['reference_agreement_passed'])
    (OUT/'render-provenance.json').write_text(json.dumps(proof, indent=2)+'\n')
    print(json.dumps(dict(video=str(video.relative_to(ROOT)), chart=str((OUT/'prediction-and-probing.png').relative_to(ROOT)))))


if __name__ == '__main__':
    render()
