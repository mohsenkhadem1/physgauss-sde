"""Report negative recovery pilots and the measured force-linearity mismatch."""
import argparse
import hashlib
import html
import json
from pathlib import Path
import shutil
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def table(headers, rows):
    return '<table><tr>'+''.join('<th>'+html.escape(str(v))+'</th>' for v in headers)+'</tr>'+''.join('<tr>'+''.join('<td>'+html.escape(str(v))+'</td>' for v in row)+'</tr>' for row in rows)+'</table>'


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=Path('runs/gaussian-campaign-v01'))
    p.add_argument('--linearity', type=Path, required=True)
    p.add_argument('--output', type=Path, default=Path('reports/gaussian-recovery-first-trials-v01'))
    a = p.parse_args()
    fit_path, audit_path = a.root/'pilot-fit-transfer.json', a.root/'superposition-audit.json'
    fit, audit, linearity = [json.loads(path.read_text()) for path in [fit_path, audit_path, a.linearity]]
    a.output.mkdir(parents=True, exist_ok=False)
    names = ['original_passive', 'original_controlled', 'same_loss_final', 'small_response_loss_final', 'balanced_small_response_loss_final']
    labels = ['Original\npassive', 'Original\ncontrolled', 'Same loss\nfinal', 'Smaller floor\nfinal', 'Balanced\nfinal']
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), layout='constrained')
    for ax, split, title in zip(axes, ['training', 'validation'], ['Fit: 48 TRAIN controls / 12 layouts', 'Transfer: 16 VALIDATION controls / 4 layouts']):
        values = [1000*fit['models'][n][split]['summary']['mean_coordinate_rmse_m'] for n in names]
        bars = ax.bar(labels, values, color=['#8995a4', '#67768a', '#348bb6', '#61a5c8', '#ce7a39'])
        ax.bar_label(bars, fmt='%.3f', padding=3, fontsize=8)
        static = 1000*fit['models'][names[0]][split]['summary']['mean_static_coordinate_rmse_m']
        ax.axhline(static, linestyle='--', color='#49515b', label=f'Static: {static:.3f}mm')
        ax.set(title=title, ylabel='Mean coordinate RMSE (mm)', ylim=(0, .98))
        ax.legend(fontsize=8)
    fig.savefig(a.output/'fit-versus-transfer.png', dpi=160)
    fig.savefig(a.output/'fit-versus-transfer.pdf')
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), layout='constrained')
    rows = audit['rows']
    axes[0].bar([r['episode_id'] for r in rows], [100*r['best_joint_residual_fraction'] for r in rows], color='#ce7a39')
    axes[0].set(title='Best unrestricted force-linear fit', ylabel='Residual / static pooled RMS (%)', ylim=(0, 55))
    axes[0].tick_params(axis='x', rotation=55)
    # Case t0001 is declared first in the zero-offset diagnostic. Never select
    # a representative case according to the size of its newly observed gain.
    curves_path = a.linearity.parent/'t0001-linearity-curves.npz'
    with np.load(curves_path) as raw:
        for key, label, color in [('diagonal_rms_motion_mm', 'Actual diagonal', '#348bb6'),
            ('zero_corrected_combination_rms_motion_mm', 'Axis combination, zero corrected', '#ce7a39'),
            ('zero_force_rms_motion_mm', 'Measured zero-force baseline', '#8995a4')]:
            axes[1].plot(raw['times_s'], raw[key], label=label, color=color)
    axes[1].set(title='TRAIN t0001: unchanged physics, new zero control', xlabel='Time (s)', ylabel='Whole-body coordinate RMS motion (mm)')
    axes[1].legend(fontsize=8)
    fig.savefig(a.output/'force-linearity.png', dpi=160)
    fig.savefig(a.output/'force-linearity.pdf')
    plt.close(fig)
    metric_rows = []
    for name, model in fit['models'].items():
        metric_rows.append([name, f"{1000*model['training']['summary']['mean_coordinate_rmse_m']:.3f}",
            f"{1000*model['validation']['summary']['mean_coordinate_rmse_m']:.3f}",
            f"{model['validation']['summary']['median_amplitude_ratio_eligible']:.3f}"])
    zero_rows = [[r['episode_id'], r['boundary_type'], f"{r['zero_force_coordinate_rms_from_rest_m']*1000:.6f}",
        f"{r['diagonal_zero_corrected_combination_rmse_m']*1000:.3f}",
        f"{100*r['best_affine_residual_fraction']:.1f}%", str(r['old_vs_new_axis_bitwise_equal'])] for r in linearity['rows']]
    report = '''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Recovery campaign: first trials and force-linearity audit</title><style>body{font:17px/1.5 system-ui;max-width:1120px;margin:36px auto;padding:0 20px;color:#1b2c3b}img{width:100%;height:auto}table{font-size:13px;border-collapse:collapse;width:100%;display:block;overflow:auto}td,th{padding:9px;border-bottom:1px solid #d5dce2;text-align:left}.callout{background:#edf4f8;padding:16px;border-left:5px solid #348bb6}a{color:#176184}</style>
<h1>Recovery campaign: better fitting is not yet better prediction</h1>
<p>Follow-up to the <a href="../gaussian-forward-inverse-diagnosis-v01/index.html">forward/inverse diagnosis</a>. This report contains development evidence, not a new sealed comparison. The campaign remains active.</p>
<div class="callout"><strong>Three training changes failed to produce a useful transfer gain.</strong><p>Reducing the small-response loss floor and balancing support types improves the fit on training controls. It does not fix the prediction on different validation layouts. A separate TRAIN-only experiment also shows that exact force superposition is incompatible with these simulator recordings, even after measuring the zero-force response.</p></div>
<h2>Loss and sampling trials</h2>
<p>All three trials use the unchanged Green architecture, the same passive initializer, 152 passive TRAIN episodes and 48 candidate TRAIN controls. Each runs 12,000 steps with matched data budgets. The only ablations are the control loss normalization floor (0.4 versus 0.04 mm) and balanced support-type sampling. Four old validation layouts select checkpoints; all best checkpoints occur at the first gradient step. Final checkpoints are reported separately to distinguish capacity to fit from transfer. Independent diagonal responses never fit these weights.</p>
<img src="fit-versus-transfer.png" alt="Improved training fits fail to transfer to different validation layouts">
'''+table(['Model/checkpoint', 'TRAIN error(mm)', 'VALIDATION error(mm)', 'Median VALIDATION amplitude ratio*'], metric_rows)+'''
<p>*Amplitude ratios use only controls whose actual coordinate RMS motion is at least 0.05 mm. They do not measure direction or phase. The balanced final model fits TRAIN with 0.153 mm error versus 0.803 mm static, but validation remains 0.834 mm versus 0.597 mm static. The best selected balanced checkpoint is 0.782 mm, still worse than static. These are four fixed validation layouts; control frames are not independent population samples.</p>
<h2>Check the exact force-linearity assumption</h2>
<p>On nine TRAIN layouts without prescribed support motion, use the three actual axis-push recordings to predict the independently recorded diagonal response. The logged forces obey the expected 0.5x + 0.5y + 0.25z combination. The mean trajectory error of that combination is 0.238 mm, versus 0.229 mm for static. Even allowing a completely unrestricted independent linear response at every time, node and coordinate leaves a mean 39.3% residual relative to each layout's pooled static RMS.</p>
<p>This initial fit assumes zero displacement at zero input, as the current Green model does. That assumption itself must be checked; the result alone is not proof of nonlinear real-world elasticity.</p>
<img src="force-linearity.png" alt="Force-linear fit residuals and an independently verified zero-force experiment">
<h2>Measure zero force with the same simulator</h2>
<p>A transparent wrapper replaces only the public control menu of the unchanged verified Isaac worker. It measures three axis pulses, zero force, the diagonal and a reset replay, on four TRAIN layouts covering all support types. Physics, hardware, timestep, capture synchronization and acceptance tolerances are unchanged. Only fully independently validated cases appear below.</p>
'''+table(['TRAIN layout', 'Support', 'Zero-force RMS from rest(mm)', 'Zero-corrected diagonal error(mm)', 'Best affine residual', 'Old axis replay bitwise equal'], zero_rows)+'''
<p>In t0001 the zero-force baseline is only about 4 nanometres RMS from rest; the original +x response reproduces bit for bit. Subtracting this measured baseline still leaves 0.177 mm diagonal error and a 33.4% pooled affine-fit residual. Constant drift and changed generation conditions do not explain that case. Numerical solver/discretization effects and finite-deformation physics remain possible causes; continuum accuracy is not established. For moving supports, zero force still permits the prescribed support motion and must not be interpreted as zero input.</p>
<h2>What is running next</h2>
<p>The next matched from-scratch comparison uses the original Green map, the existing nonlinear coordinate operator, and the same nonlinear operator referenced to zero force and zero support-motion amplitude. The last variant adds no parameters; it retains nonlinear force dependence while enforcing exact rest at zero input. These candidates are not declared successful. The four old validation layouts are development only. Newly captured control layouts and 32 newly reserved confirmation layouts remain separate roles.</p>
<p>The additional data loader admits only declared TRAIN and checkpoint-selection controls. It rejects calibration-only cases, reserved control-response gate cases and all TEST cases before opening their private responses. Validation passive trajectories were already used in model selection in earlier phases; the new control gate is a held-out response task, not wholly unseen physical layouts. The 32 fresh confirmation layouts are the final independent comparison.</p>
<p>No uncertainty-guided probe comparison has been rerun. Known-force prediction must pass the campaign gate first.</p>
<h2>Evidence and reproduction</h2>
<p><a href="pilot-fit-transfer.json">All fit/transfer metrics and checkpoint hashes</a> · <a href="superposition-audit.json">Nine-layout TRAIN audit</a> · <a href="zero-force-analysis.json">Verified zero-force analysis and input hashes</a> · <a href="tests.txt">17 phase 2 checks</a> · <a href="manifest.json">Manifest</a> · <a href="../../configs/gaussian_campaign_nonlinear_v01.json">Nonlinear phase protocol</a> · <a href="../../docs/GAUSSIAN_RECOVERY_CAMPAIGN.md">Campaign</a></p>
<p>Commands: <code>experiments/evaluate_campaign_pilots.py</code>, <code>experiments/audit_campaign_superposition.py</code>, <code>experiments/run_isaac_linearity_diagnostic.py</code>, and <code>experiments/analyze_isaac_linearity.py --episodes t0001 t0000 t0002 t0003 --output NEW.json</code>. Existing dataset/models are in the published artifact index; campaign raw data/model publication follows completed snapshots.</p></html>'''
    (a.output/'index.html').write_text(report)
    for source, name in [(fit_path, 'pilot-fit-transfer.json'), (audit_path, 'superposition-audit.json'),
                         (a.linearity, 'zero-force-analysis.json'), (a.root/'phase2-tests.txt', 'tests.txt')]:
        shutil.copy2(source, a.output/name)
    shutil.copy2(curves_path, a.output/curves_path.name)
    provenance = a.output/'provenance'
    provenance.mkdir()
    for name in ['report_campaign_first_trials.py', 'evaluate_campaign_pilots.py', 'audit_campaign_superposition.py',
                 'analyze_isaac_linearity.py', 'isaac_probe_linearity_diagnostic.py', 'run_isaac_linearity_diagnostic.py']:
        shutil.copy2(Path(__file__).with_name(name), provenance/name)
    manifest = {'campaign_sha256': sha('configs/gaussian_campaign_v01.json'), 'weights_changed_by_reporting': False,
        'files': {str(path.relative_to(a.output)): sha(path) for path in sorted(a.output.rglob('*')) if path.is_file()},
        'qualification': 'Development evidence; no fresh confirmation or reserved gate labels used here'}
    (a.output/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(json.dumps({'report': str(a.output), 'verified_zero_cases': len(linearity['rows'])}))


if __name__ == '__main__':
    main()
