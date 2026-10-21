"""Publish a separate explanatory diagnosis; leave sealed results untouched."""
import argparse
import html
import json
import shutil
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from diagnose_gaussian_forward import sha
from gaussian_study_statistics import paired_layout_bootstrap


def table(headers, rows):
    return '<div class="scroll"><table><tr>'+''.join('<th>'+html.escape(str(x))+'</th>' for x in headers)+'</tr>'+''.join('<tr>'+''.join('<td>'+html.escape(str(x))+'</td>' for x in row)+'</tr>' for row in rows)+'</table></div>'


def save(fig, output, name):
    fig.savefig(output/(name+'.png'), dpi=160, bbox_inches='tight')
    fig.savefig(output/(name+'.pdf'), bbox_inches='tight')
    plt.close(fig)


def run(oracle_path, reference_path, output):
    output.mkdir(parents=True, exist_ok=False)
    oracle, reference = [json.loads(p.read_text()) for p in [oracle_path, reference_path]]
    if oracle['protocol_sha256'] != reference['protocol_sha256']:
        raise ValueError('Different diagnostic protocols')
    if len(oracle['passive']) != 76 or len(oracle['controls']) != 40 or len(reference['episodes']) != 96:
        raise ValueError('Incomplete diagnosis')
    primary = [r for r in oracle['passive'] if r['dataset'] == 'primary_dataset']
    controlled = [r for r in oracle['controls'] if r['action_id'] == 'evaluation']
    mean = lambda values: float(np.mean(list(values)))
    summary = {'protocol_sha256': oracle['protocol_sha256'],
        'counts': {'fresh_layouts': 64, 'secondary_layouts': 4, 'secondary_episodes': 12,
                   'controlled_layouts': 8, 'controlled_responses': 40, 'reference_layouts': 4},
        'passive': {}, 'control': {}, 'reference': {}, 'comparisons': oracle['comparisons']}
    for name, metric, learned_key, oldmetric in [
        ('missing_interval', 'gap', 'learned_rgbd_full', 'gap_coordinate_rmse_m'),
        ('future', 'late', 'learned_rgbd_prefix', 'late_2to3_coordinate_rmse_m')]:
        summary['passive'][name] = {
            'static_mm': 1000*mean(r['oracle'][metric+'_static_rmse_m'] for r in primary),
            'inferred_rgbd_mm': 1000*mean(r[learned_key][oldmetric] for r in primary),
            'true_parameter_forward_mm': 1000*mean(r['oracle'][metric+'_coordinate_rmse_m'] for r in primary)}
    summary['control'] = {
        'static_mm': 1000*mean(r['models']['controlled']['all_static_rmse_m'] for r in controlled),
        'passive_inference_mm': 1000*mean(r['learned_control_policies']['passive']['withheld_control_coordinate_rmse_m'] for r in controlled),
        'after_risk_poke_mm': 1000*mean(r['learned_control_policies']['predictive_risk']['withheld_control_coordinate_rmse_m'] for r in controlled),
        'true_parameters_controlled_forward_mm': 1000*mean(r['models']['controlled']['all_coordinate_rmse_m'] for r in controlled),
        'true_parameters_original_forward_mm': 1000*mean(r['models']['passive']['all_coordinate_rmse_m'] for r in controlled)}
    summary['controlled_vs_original_true_parameter'] = paired_layout_bootstrap(
        [r['models']['controlled']['all_coordinate_rmse_m'] for r in controlled],
        [r['models']['passive']['all_coordinate_rmse_m'] for r in controlled])
    for mode in ['prefix', 'full']:
        summary['reference'][mode] = {}
        for noise in oracle['protocol']['noise_std_mm']:
            rows = [r for r in reference['episodes'] if r['mode'] == mode and r['noise_std_mm'] == noise]
            checks = [r for r in reference['synthetic_sensor_checks'] if r['mode'] == mode and r['noise_std_mm'] == noise]
            summary['reference'][mode][str(noise)] = {
                'simulator_mean_true_probability': mean(r['simulator_reference']['true_probability'] for r in rows),
                'neural_mean_true_probability': mean(r['neural_forward_reference']['true_probability'] for r in rows),
                'simulator_mean_credible_set_size': mean(r['simulator_reference']['credible90_size'] for r in rows),
                'neural_mean_credible_set_size': mean(r['neural_forward_reference']['credible90_size'] for r in rows),
                'mean_total_variation_simulator_neural': mean(r['total_variation_simulator_vs_neural'] for r in rows),
                'synthetic_sensor': {key: {metric: mean(c[key][metric] for c in checks) for metric in
                    ['map_correct', 'credible90_contains_truth', 'credible90_size', 'true_probability']}
                    for key in ['simulator', 'neural_forward']}}
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), layout='constrained')
    for ax, task, title in zip(axes[:2], ['missing_interval', 'future'], ['Missing interval: 64 layouts', 'Future: inputs through 1 s']):
        values = summary['passive'][task]
        bars = ax.bar(['Static', 'Inferred\nparameters', 'True\nparameters'], list(values.values()), color=['#929aa5', '#348bb6', '#ce7a39'])
        ax.bar_label(bars, fmt='%.3f', padding=3)
        ax.set(title=title, ylabel='Mean coordinate RMSE (mm)', ylim=(0, 1.02))
    ax = axes[2]
    ratios = [r['models']['controlled']['all_predicted_rms_motion_m']/r['models']['controlled']['all_actual_rms_motion_m'] for r in controlled]
    ax.bar([r['episode_id'] for r in controlled], ratios, color='#ce7a39')
    ax.axhline(1, color='#4d6170', linestyle='--', label='Matched RMS motion')
    ax.set(title='Known-parameter diagonal poke', ylabel='Predicted / actual RMS motion', ylim=(0, 1.13))
    ax.tick_params(axis='x', rotation=60)
    ax.legend(fontsize=8)
    save(fig, output, 'forward-check')
    fig, axes = plt.subplots(2, 3, figsize=(11, 10), layout='constrained')
    for mode_index, mode in enumerate(['prefix', 'full']):
        rows = [r for r in reference['episodes'] if r['mode'] == mode and r['noise_std_mm'] == .5]
        for column, (key, title) in enumerate([
            ('simulator_reference', 'Numerical simulator posterior'),
            ('neural_forward_reference', 'Same grid, neural forward'),
            ('flow_density_slice', 'Amortized density slice*')]):
            matrix = np.array([r[key]['normalized_scores' if key == 'flow_density_slice' else 'probabilities'] for r in rows])
            ax = axes[mode_index, column]
            ax.imshow(matrix, vmin=0, vmax=1, cmap='Blues', aspect='auto')
            for i in range(len(rows)):
                for j in range(3):
                    ax.text(j, i, f'{matrix[i,j]:.2f}', ha='center', va='center', fontsize=8,
                            color='white' if matrix[i,j] > .6 else '#18242f')
                ax.plot(rows[i]['true_index'], i, marker='s', markersize=26, markerfacecolor='none', markeredgecolor='#d78032', markeredgewidth=1.5)
            ax.set_xticks(range(3), ['1.5 kPa', '5 kPa', '20 kPa'])
            ax.set_yticks(range(12), [r['episode_id']+' ('+r['group_id']+')' for r in rows] if column == 0 else ['']*12)
            ax.set(title=title, xlabel='Candidate stiffness')
            if column == 0:
                ax.set_ylabel(('Inputs through 1 s' if mode == 'prefix' else 'All ten input snapshots')+'\nTest episode / fixed layout')
    fig.suptitle('Known nuisance parameters; 0.5 mm noise scale. Orange box = true stiffness.\n*Flow slice retains its learned prior; not an equal-prior marginal posterior.', fontsize=11)
    save(fig, output, 'reference-posteriors')
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout='constrained', sharey=True)
    noise_values = oracle['protocol']['noise_std_mm']
    for ax, mode in zip(axes, ['prefix', 'full']):
        for key, label, color in [('simulator', 'Simulator reference', '#348bb6'), ('neural_forward', 'Neural forward substituted', '#ce7a39')]:
            coverage = [100*summary['reference'][mode][str(n)]['synthetic_sensor'][key]['credible90_contains_truth'] for n in noise_values]
            ax.plot(noise_values, coverage, 'o-', color=color, label=label)
        ax.axhline(90, linestyle='--', color='#777777', linewidth=1)
        ax.set(xlabel='Declared noise scale (mm)', ylabel='90% credible-set coverage (%)', ylim=(60, 102),
               title='Inputs through 1 s' if mode == 'prefix' else 'All ten input snapshots')
        ax.legend(fontsize=8)
    fig.suptitle('256 synthetic sensor draws per fixed layout, four layouts; uniform three-point prior.\nCoverage under the stipulated sensor model; these are not new independent physical objects.', fontsize=10)
    save(fig, output, 'sensor-sensitivity')
    passive_table = table(['Task / 64 layouts', 'Static', 'Inferred RGB-D', 'True parameters'], [
        [name.replace('_', ' '), *[f'{v:.3f} mm' for v in row.values()]] for name, row in summary['passive'].items()])
    control_table = table(['Independent diagonal response / 8 objects', 'Mean coordinate error'], [[key.replace('_', ' '), f'{value:.3f} mm'] for key, value in summary['control'].items()])
    response_table = table(['Object', 'Support', 'Actual RMS motion', 'Predicted RMS motion', 'Amplitude ratio'], [
        [r['episode_id'], r['boundary_type'], f"{r['models']['controlled']['all_actual_rms_motion_m']*1000:.3f} mm",
         f"{r['models']['controlled']['all_predicted_rms_motion_m']*1000:.3f} mm", f'{ratio:.3f}'] for r, ratio in zip(controlled, ratios)])
    ref_table = table(['Inputs', 'Noise scale', 'True probability: simulator / neural', 'Synthetic 90% coverage: simulator / neural'], [
        [mode, str(n)+' mm', f"{v['simulator_mean_true_probability']:.3f} / {v['neural_mean_true_probability']:.3f}",
         f"{100*v['synthetic_sensor']['simulator']['credible90_contains_truth']:.1f}% / {100*v['synthetic_sensor']['neural_forward']['credible90_contains_truth']:.1f}%"]
        for mode in ['prefix', 'full'] for n in noise_values for v in [summary['reference'][mode][str(n)]]])
    body = '''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Forward model and inference diagnosis</title><style>body{font:17px/1.55 system-ui,sans-serif;max-width:1150px;margin:40px auto;padding:0 22px;color:#1b2c3b}h1,h2{line-height:1.2}img{width:100%;height:auto;border:1px solid #d5dce2}table{border-collapse:collapse;width:100%;font-size:14px}td,th{padding:9px;border-bottom:1px solid #d5dce2;text-align:left}.scroll{overflow:auto}.callout{background:#edf4f8;border-left:5px solid #348bb6;padding:16px}code{font-size:14px}pre{overflow:auto;background:#f3f5f7;padding:14px}a{color:#176184}</style>
<h1>Why the model becomes confident without becoming correct</h1>
<p>Controlled diagnosis of the existing frozen models. No architecture, checkpoint or action policy was fitted or selected. The <a href="../gaussian-sealed-study-v01/index.html">original sealed report</a> remains unchanged. This is explanatory reuse of already opened TEST data, not a new confirmatory benchmark.</p>
<div class="callout"><strong>The forward predictor itself is a major limitation, and some observations genuinely cannot distinguish stiffness.</strong><p>Giving it the true material, holding points and forces helps fill missing motion, but does not make the response accurate. A small numerical simulator reference separates one clearly distinguishable case from another that remains ambiguous. We have not isolated the full inverse-model error or established calibrated active probing.</p></div>
<h2>1. Give the predictor the correct physical answer</h2>
<p>We supplied all true physical coefficients to the frozen forward predictor: material properties, support and load locations, support/load categories, force vectors and prescribed support motion. All 64 fresh layouts and all 12 secondary test episodes were evaluated. The table shows the fresh layouts. “True parameters” is a privileged diagnostic, not a deployable image predictor.</p>'''+passive_table+'''
<p>The missing-interval improvement from inferred to true parameters is 14.3%, with a paired 64-layout bootstrap interval of 4.8–22.2%. However, the true-parameter future error is essentially the same as the inferred forecast. A true-parameter query is not a lower bound on a biased surrogate: wrong parameters can compensate for wrong mechanics. Forward and inverse errors cannot be subtracted as independent error components.</p>
<img src="forward-check.png" alt="True parameters help interpolation but not future prediction; known-force amplitudes are often too small">
<h2>2. Test held-out forces on the same objects</h2>
<p>All four candidate pokes and the independent diagonal evaluation control were checked on each of eight test objects (40 saved responses). The original supports, moving-support targets and both hardware handles were retained. Every old unknown force was replaced with the declared known command. These oracle queries never modify the encoder or its saved posterior.</p>'''+control_table+'''
<p>The controlled teacher has higher true-parameter error than the original passive forward model on all five control families on average. This is a diagnostic comparison of frozen checkpoints, not selection of a replacement model. Its independent diagonal response remains substantially wrong even with the exact physical answer supplied.</p>'''+response_table+'''
<p>For six of eight diagonal responses, the controlled teacher produces less than 16% of the actual RMS motion. The two moving-support objects have larger responses; their directions and time courses can still be wrong. RMS amplitude is descriptive and does not measure phase, direction or maximum displacement.</p>
<h2>3. Enumerate a small numerical reference posterior</h2>
<p>Four earlier held-out layouts each have exact Isaac recordings at 1.5, 5 and 20 kPa. All other material properties, supports, forces, hardware and camera paths are identical within each triple and checked before comparison. We supplied those nuisance values and put equal prior mass on the three stiffnesses. The reference posterior is an exact normalized finite sum over these three simulated possibilities.</p>
<p>The sensor uses 12 pixels chosen from known rest geometry at each observed timestamp. There is independent, frame-shared and sequence-shared range noise, plus explicit visibility errors and an outlier branch. All four noise scales were specified before running this diagnosis; none was fitted to TEST. Prefix and full-context results are separate. Predictions from the neural model are ray-cast at exactly the same pixels, and use the same prior and likelihood as the simulator bank.</p>
<img src="reference-posteriors.png" alt="Three-stiffness posterior matrices comparing simulator, neural forward and learned density slice">
<p><strong>Genuine ambiguity:</strong> in g06, the largest recorded depth difference between stiffness candidates at the selected input pixels is only 0.0102 mm. With a 0.5 mm noise scale the numerical posterior remains about one-third for each stiffness, even with all snapshots. In g12, 5 and 20 kPa also remain difficult to distinguish.</p>
<p><strong>Forward-model error:</strong> in g03 the simulator reference cleanly separates the three stiffnesses. With inputs through 1 s, for the true 20 kPa case, replacing the simulator with the neural forward model instead favours 5 kPa. This comparison holds the prior, candidate set, nuisance parameters, observations and likelihood fixed; only the forward response changes.</p>
<p><strong>Amortized inference:</strong> the third panel evaluates the learned 25D density at the same true-nuisance hypotheses and normalizes those scores with equal log-coordinate cell widths. Its learned prior and nuisance conditioning differ from the uniform-grid problem; it is a density slice, not a full marginal or an equal-prior Bayes reference. Its disagreements are useful warnings, but cannot alone quantify pure inference error. A full numerical posterior with matching prior and marginalized nuisance variables remains uncomputed.</p>
<h2>4. Check how forward error changes uncertainty</h2>
<p>The noiseless simulator-bank self-match is an upper bound, not evidence of universal identification. We therefore generated 256 noisy observations per layout from the declared simulator-template sensor model with a uniform stiffness prior, then inferred with either the simulator templates or the neural responses. These repeated sensor draws concern four fixed layouts; they are not thousands of independent objects. Discrete 90% credible sets can contain more than 90% mass, explaining conservative reference coverage.</p>'''+ref_table+'''
<img src="sensor-sensitivity.png" alt="Replacing true simulator responses with neural forward responses lowers credible-set coverage under the same sensor model">
<p>At 0.5 mm and full context, synthetic credible-set coverage drops from 99.6% with simulator templates to 79.0% when neural responses are substituted. This is a controlled demonstration of forward misspecification causing bad uncertainty, not a new physical coverage guarantee. The conclusion depends on noise scale; the full sweep is shown rather than selecting the most favourable setting.</p>
<h2>What this changes about the next experiment</h2>
<p>First establish accurate known-force responses, including amplitude, phase, direction, support motion and stiffness sensitivity on reserved layouts. Check control finetuning against the original model rather than assuming it helps. Then use a matching-prior numerical posterior with nuisance integration to measure amortization error. Only after predictive uncertainty passes reserved calibration checks should it choose views or pokes. This diagnosis does not solve any of those steps.</p>
<h2>Reproduction and evidence</h2><p><a href="oracle-results.json">All oracle case results, paired bootstrap comparisons and input hashes</a> · <a href="reference-results.json">All grid posteriors, synthetic checks and input hashes</a> · <a href="summary.json">Summary</a> · <a href="evidence-manifest.json">Evidence manifest</a> · <a href="diagnosis-tests.txt">Six numerical checks</a> · <a href="regression-tests.txt">20 targeted checks including the six new tests</a></p>
<pre>OPENBLAS_NUM_THREADS=2 .venv/bin/python experiments/diagnose_gaussian_forward.py --output runs/new-oracle --device cuda:0
OPENBLAS_NUM_THREADS=2 .venv/bin/python experiments/diagnose_gaussian_reference.py --output runs/new-reference --device cuda:1
OPENBLAS_NUM_THREADS=2 .venv/bin/python experiments/report_gaussian_diagnosis.py --oracle runs/new-oracle/results.json --reference runs/new-reference/results.json --output reports/new-diagnosis
OPENBLAS_NUM_THREADS=2 .venv/bin/python -m unittest discover -s tests -p test_gaussian_diagnosis.py</pre>
<p>The <a href="../../examples/gaussian-neural-boundary/artifacts.json">existing artifact index</a> supplies the frozen models, original simulations and sensor data. Only CPU/GPU evaluation was run for this diagnosis; no new Isaac simulations were needed. Bird remains stopped.</p></html>'''
    (output/'index.html').write_text(body)
    (output/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    shutil.copy2(oracle_path, output/'oracle-results.json')
    shutil.copy2(reference_path, output/'reference-results.json')
    for p in reference_path.parent.glob('*-sensor-templates.npz'):
        shutil.copy2(p, output/p.name)
    shutil.copy2('runs/gaussian-diagnosis-tests.txt', output/'diagnosis-tests.txt')
    shutil.copy2('runs/gaussian-diagnosis-regression.txt', output/'regression-tests.txt')
    source_dir = output/'provenance'
    source_dir.mkdir()
    sources = ['diagnose_gaussian_forward.py', 'diagnose_gaussian_reference.py', 'report_gaussian_diagnosis.py',
        'censored_range_observation.py', 'gaussian_acquisition_sensor.py', 'validate_isaac_dataset.py',
        'gaussian_operator.py', 'evaluate_gaussian_inverse.py', 'train_gaussian_inverse.py',
        'train_gaussian_forward.py', 'gaussian_observation.py', 'gaussian_observed_features.py',
        'gaussian_video_encoder.py', 'gaussian_study_statistics.py']
    for name in sources:
        shutil.copy2(Path(__file__).with_name(name), source_dir/name)
    shutil.copy2('configs/gaussian_diagnosis_v01.json', output/'protocol.json')
    shutil.copy2('tests/test_gaussian_diagnosis.py', source_dir/'test_gaussian_diagnosis.py')
    manifest = {'source_commit_before_diagnosis': oracle['protocol']['source_commit'],
        'protocol_sha256': oracle['protocol_sha256'], 'no_checkpoint_updates': True,
        'files': {str(p.relative_to(output)): sha(p) for p in sorted(output.rglob('*')) if p.is_file()},
        'qualification': 'Separate exploratory diagnosis. Every saved template and numerical result is included; original sealed report is unchanged.'}
    (output/'evidence-manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--oracle', type=Path, default=Path('runs/gaussian-diagnosis-oracle-v03/results.json'))
    parser.add_argument('--reference', type=Path, default=Path('runs/gaussian-diagnosis-reference-v01/results.json'))
    parser.add_argument('--output', type=Path, default=Path('reports/gaussian-forward-inverse-diagnosis-v01'))
    args = parser.parse_args()
    run(args.oracle, args.reference, args.output)
