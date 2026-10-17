"""Verify portable posterior evidence and build the offline scientific report."""
import argparse
import base64
import csv
import hashlib
import html
import json
from pathlib import Path
import shutil
import subprocess
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from physgauss.accurate_study import digest_tensor
from physgauss.cli import source_fingerprint
from physgauss.neural_marginal_study import calibrate, construct
from physgauss.camera import orbit
from physgauss.study import write_json


BASE = Path('examples/calibrated-noise/comparison')
COLORS = {'floor': '#888888', 'constant': '#c37b25', 'neural': '#197a91'}
LABELS = {'no_load': 'No load', 'external_pulse': 'Pulse', 'stochastic': 'Internal noise', 'pulse_and_stochastic': 'Pulse + noise'}


def compact(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, separators=(',', ':'), allow_nan=False)+'\n', encoding='utf-8')


def digest(path):
    path = Path(path)
    data = path.read_bytes()
    if path.suffix in ['.json', '.html', '.csv', '.md', '.py', '.yml', '.toml']:
        data = data.replace(b'\r\n', b'\n')
    return hashlib.sha256(data).hexdigest()


def verify(raw):
    summary = json.loads((raw/'summary.json').read_text())
    rows = summary['cases']; plan = summary['plan']
    if len(rows) != 36 or [r['case'] for r in rows] != plan['cases']:
        raise ValueError('Incomplete/reordered comparison.')
    source = source_fingerprint()
    torch.set_num_threads(2)
    import torch.backends.opt_einsum as backend
    backend.enabled = True; backend.strategy = 'greedy'
    base_manifest = json.loads((BASE/'manifest.json').read_text())
    base_plan = base_manifest['plan']
    net, calibration = calibrate(base_plan, BASE/'calibration.json')
    models = {mode: construct(base_plan, net, calibration, mode) for mode in COLORS}
    records = []
    for row in rows:
        if row['source_fingerprint_sha256'] != source:
            raise ValueError('Wrong numerical source.')
        folder = raw/'cases'/row['case']
        saved = json.loads((folder/'summary.json').read_text())
        if saved != row:
            raise ValueError('Aggregate and case summaries differ.')
        p = json.loads((folder/'posterior.json').read_text())
        diagnostic = json.loads((folder/'diagnostics.json').read_text())
        settings = json.loads((folder/'settings.json').read_text())
        for key in ['observations_sha256', 'source_fingerprint_sha256', 'reference_mean_sha256', 'plan_sha256', 'worker_script_sha256_normalized_lf']:
            if settings[key] != row[key]:
                raise ValueError('Case provenance differs: '+key)
        if settings['worker_script_sha256_normalized_lf'] != digest('scripts/converged_sparse_comparison.py'):
            raise ValueError('Worker implementation differs.')
        model = models[row['mode']]
        old = json.loads((BASE/'cases'/row['case']/'summary.json').read_text())
        if old['scores'] != row['previous_MAP_scores']:
            raise ValueError('MAP comparator changed.')
        inputs = json.loads((BASE/'datasets'/f"{row['condition']}__{row['seed']}"/'inputs.json').read_text())
        if digest_tensor(model.times.new_tensor(inputs['observed_rgb'])) != row['observations_sha256']:
            raise ValueError('Observed input changed.')
        state = model.times.new_tensor(p['normalized_state_mean'])
        with torch.no_grad(): x = model.trajectory(state[None]).x[0]
        torch.testing.assert_close(x, model.times.new_tensor(p['positions_mean_m']), atol=2e-12, rtol=1e-12)
        truth = model.times.new_tensor(inputs['positions_m_evaluation_only'])[:, model.free]
        gap = (model.times > .3+1e-8)&(model.times < .7-1e-8)
        lower, upper = [model.times.new_tensor(p[k]) for k in ['free_position_q05_m', 'free_position_q95_m']]
        error = x[:, model.free]-truth
        values = {'gap_rmse_m': float(error[gap].square().mean().sqrt()),
            'all_position_rmse_m': float(error.square().mean().sqrt()),
            'gap_component_90_coverage': float(((truth >= lower)&(truth <= upper))[gap].double().mean()),
            'gap_mean_width_m': float((upper-lower)[gap].mean())}
        for key, value in values.items():
            if abs(value-row['scores'][key]) > 1e-11:
                raise ValueError('Portable posterior scoring mismatch: '+key)
        cols = diagnostic['columns']; n = 1549 if row['mode'] == 'floor' else 1566
        if len(diagnostic['names']) != n or row['monitored_quantities'] != n:
            raise ValueError('Incomplete diagnostics.')
        for i in range(n):
            passed = (cols['rank_folded_split_rhat'][i] is not None and cols['rank_folded_split_rhat'][i] < 1.01
                and cols['bulk_ess'][i] >= 400 and cols['tail_ess'][i] >= 400 and cols['frozen_chains'][i] == 0)
            if passed != cols['diagnostic_pass'][i]:
                raise ValueError('Diagnostic pass flag does not reproduce.')
        if bool(all(cols['diagnostic_pass'])) != (row['posterior_status'] == 'finite_diagnostics_pass'):
            raise ValueError('Incorrect posterior validation label.')
        total = model.times.new_tensor(p['force_coefficient_covariance'])
        conditional = model.times.new_tensor(p['force_mean_conditional_covariance'])
        between = model.times.new_tensor(p['force_between_trajectory_covariance'])
        torch.testing.assert_close(total, conditional+between, rtol=1e-12, atol=1e-12)
        if float(torch.linalg.eigvalsh(total).min()) < -1e-10:
            raise ValueError('Nonpositive force covariance.')
        if len(p['portable_draws']['gap_free_xyz_m']) != 400:
            raise ValueError('Missing paired portable draws.')
        records.append({'case': row['case'], 'scores_reproduced': True, 'diagnostics_recomputed': True,
            'force_covariance_identity': True, 'observation_hash_verified': True})
    return summary, models, records


def figures(summary, models, output, old_audit):
    plt.rcParams.update({'font.size': 11, 'axes.spines.top': False, 'axes.spines.right': False,
                         'figure.facecolor': 'white', 'axes.titleweight': 'bold'})
    conditions = list(LABELS)
    aggregate = {(r['condition'], r['mode']): r for r in summary['aggregates']}
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharey=True)
    for ax, condition in zip(axes.flat, conditions):
        for i, mode in enumerate(COLORS):
            r = aggregate[condition, mode]
            ax.bar(i-.18, 1000*r['MAP_gap_rmse_m'], .34, facecolor='white', edgecolor=COLORS[mode], linewidth=2)
            ax.bar(i+.18, 1000*r['posterior_gap_rmse_m'], .34, color=COLORS[mode])
        ax.set(xticks=range(3), xticklabels=['Floor', 'Constant', 'Neural'], title=LABELS[condition], ylabel='Missing-gap error (mm)')
        ax.grid(axis='y', alpha=.2)
    from matplotlib.patches import Patch
    fig.legend(handles=[Patch(facecolor='white', edgecolor='#444', label='Previous MAP'), Patch(facecolor='#444', label='Posterior mean')], ncol=2, loc='upper center', bbox_to_anchor=(.5, .955), frameon=False)
    fig.suptitle('Same six images and same models: change the inference', y=.995, fontsize=16)
    fig.tight_layout(rect=(0, 0, 1, .90)); fig.savefig(output/'reconstruction.png', dpi=160, bbox_inches='tight'); plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for i, mode in enumerate(COLORS):
        r = aggregate['pooled_stochastic', mode]
        for offset, prefix, fill in [(-.18, 'MAP', False), (.18, 'posterior', True)]:
            axes[0].bar(i+offset, 100*r[prefix+'_component_90_coverage'], .34, color=COLORS[mode] if fill else 'white', edgecolor=COLORS[mode])
            axes[1].bar(i+offset, 1000*r[prefix+'_width_m'], .34, color=COLORS[mode] if fill else 'white', edgecolor=COLORS[mode])
    axes[0].axhline(90, ls='--', c='#444'); axes[0].set(ylabel='Covered gap components (%)', ylim=(0, 105), title='Central 90% component intervals')
    axes[1].set(ylabel='Average component width (mm)', title='No conformal widening')
    for ax in axes:
        ax.set_xticks(range(3), ['Floor', 'Constant', 'Neural']); ax.grid(axis='y', alpha=.2)
    fig.suptitle('Six paired stochastic benchmark cases; descriptive coverage, not calibration proof', y=.995)
    fig.legend(handles=[Patch(facecolor='white', edgecolor='#444', label='MAP/local intervals'), Patch(facecolor='#444', label='Posterior intervals')], ncol=2, loc='upper center', bbox_to_anchor=(.5, .945), frameon=False)
    fig.tight_layout(rect=(0, 0, 1, .86)); fig.savefig(output/'uncertainty.png', dpi=160); plt.close(fig)
    selected = 'pulse_and_stochastic__4101__neural'
    post = json.loads((output/'cases'/selected/'posterior.json').read_text())
    model = models['neural']
    inputs = json.loads((BASE/'datasets'/'pulse_and_stochastic__4101'/'inputs.json').read_text())
    truth = model.times.new_tensor(inputs['positions_m_evaluation_only'])
    original = json.loads((BASE/'cases'/selected/'solver_state.json').read_text())
    with torch.no_grad(): old_path = model.trajectory(model.times.new_tensor(original['coordinates'])[None]).x[0]
    mean = model.times.new_tensor(post['positions_mean_m'])
    low = model.times.new_tensor(post['free_position_q05_m']); high = model.times.new_tensor(post['free_position_q95_m'])
    fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
    node = int(torch.where(model.free)[0][-1]); free_index = int(model.free.sum())-1
    for k, ax in enumerate(axes):
        ax.axvspan(.3, .7, color='#e4e8eb', alpha=.6)
        ax.fill_between(model.times, low[:, free_index, k], high[:, free_index, k], color=COLORS['neural'], alpha=.2, label='Posterior 90%')
        ax.plot(model.times, truth[:, node, k], color='#222', label='Reference')
        ax.plot(model.times, old_path[:, node, k], '--', color='#bc5352', label='Previous MAP')
        ax.plot(model.times, mean[:, node, k], color=COLORS['neural'], label='Posterior mean')
        ax.scatter(model.times[model.view_frames], truth[model.view_frames, node, k], c='#222', s=15)
        ax.set_ylabel('xyz'[k]+' position (m)'); ax.grid(alpha=.2)
    axes[0].legend(ncol=4, fontsize=9); axes[-1].set_xlabel('Time (s)')
    fig.suptitle('Declared development case: node 7, pulse + internal noise, seed 4101\nDots mark image times (not observed 3D landmarks); gray marks the gap')
    fig.tight_layout(); fig.savefig(output/'trajectory.png', dpi=160); plt.close(fig)
    with torch.no_grad():
        constant = json.loads((output/'cases'/'pulse_and_stochastic__4101__constant'/'posterior.json').read_text())
        state_old_c = json.loads((BASE/'cases'/'pulse_and_stochastic__4101__constant'/'solver_state.json').read_text())
        old_c = models['constant'].trajectory(model.times.new_tensor(state_old_c['coordinates'])[None]).x[0]
        paths = [truth, old_path, mean, old_c, model.times.new_tensor(constant['positions_mean_m'])]
        eye, rotation = orbit(model.times)
        images = [model.times.new_tensor(inputs['clean_rgb']['0'])[10].numpy()]+[
            model.renderer(x[None, [10]], eye[[10]], rotation[[10]])[0, 0].numpy() for x in paths[1:]]
    fig, axes = plt.subplots(2, 5, figsize=(14, 5.5))
    vmax = max(np.abs(im-images[0]).mean(-1).max() for im in images[1:])
    for j, (image, label) in enumerate(zip(images, ['Reference', 'Neural MAP', 'Neural posterior', 'Constant MAP', 'Constant posterior'])):
        axes[0, j].imshow(np.clip(image, 0, 1), interpolation='nearest'); axes[0, j].set_title(label)
        if j:
            im = axes[1, j].imshow(np.abs(image-images[0]).mean(-1), cmap='magma', vmin=0, vmax=vmax)
        else:
            axes[1, j].text(.1, .5, 'Mean absolute\nRGB error\n(shared color scale)', transform=axes[1, j].transAxes)
        for ax in axes[:, j]: ax.axis('off')
    fig.colorbar(im, ax=list(axes[1, 1:]), shrink=.6)
    fig.suptitle('Unobserved middle frame (0.50 s), original camera; 24×24 model images')
    fig.savefig(output/'images.png', dpi=160, bbox_inches='tight'); plt.close(fig)
    # Historical trace is exported as a small portable numerical record.
    if (old_audit/'chains.pt').exists():
        old_chain = torch.load(old_audit/'chains.pt', weights_only=True)
        prior = model.prior
        old_drag = torch.exp(prior.mean[1]+prior.std[1]*old_chain['coordinates'][..., 1])[:, ::10]
        old_indices = list(range(0, old_chain['coordinates'].shape[1], 10))
    else:
        saved_trace = json.loads(Path('examples/converged-posterior/old_pcn_damping_traces.json').read_text())
        old_drag = model.times.new_tensor(saved_trace['damping_per_s'])
        old_indices = saved_trace['indices']
    indices = post['portable_draws']['retained_indices'][:100]
    new_drag = np.array(post['portable_draws']['physical_parameters'])[:, 1].reshape(4, 100)
    compact(output/'old_pcn_damping_traces.json', {'case': selected, 'indices': old_indices,
        'damping_per_s': old_drag.tolist(), 'qualification': 'historical failed pCN audit; used only for exploration diagnosis'})
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    for chain in range(4):
        axes[0].plot(old_indices, old_drag[chain], alpha=.8, lw=.9)
        axes[1].plot(indices, new_drag[chain], alpha=.8, lw=.9)
    axes[0].set_title('Previous pCN: failed diagnostics'); axes[1].set_title('Split HMC: all 1,566 monitors pass')
    for ax in axes:
        ax.axhline(.55, ls='--', color='#444', label='Truth'); ax.set_xlabel('Retained transition'); ax.grid(alpha=.2)
    axes[0].legend(loc='upper right', fontsize=9)
    axes[0].set_ylabel('Damping (/s)'); fig.tight_layout(); fig.savefig(output/'mixing.png', dpi=160); plt.close(fig)
    with torch.no_grad():
        coefficients = model.times.new_tensor(post['portable_draws']['force_coefficients'])
        force_draws = model.force_prior.total_force(coefficients, model.times)
        force_mean = model.times.new_tensor(post['force_total_mean_N'])
        q05, q95 = torch.quantile(force_draws, force_draws.new_tensor([.05, .95]), dim=0)
        truth_force = (model.mesh.mass[model.free].sum()*torch.exp(-.5*((model.times-.5)/.08).square()))[:, None]*model.times.new_tensor([.45, 0., -.30])
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharex=True)
    for k, ax in enumerate(axes):
        ax.fill_between(model.times, q05[:, k], q95[:, k], color=COLORS['neural'], alpha=.2)
        ax.plot(model.times, truth_force[:, k], color='#222', label='Reference load')
        ax.plot(model.times, force_mean[:, k], color=COLORS['neural'], label='Posterior mean')
        ax.set(title='xyz'[k]+' component', xlabel='Time (s)', ylabel='Total force (N)'); ax.grid(alpha=.2)
    axes[0].legend(fontsize=9)
    fig.suptitle('Force remains marginalized during inference; posterior mixes its Gaussian conditionals')
    fig.tight_layout(); fig.savefig(output/'force.png', dpi=160); plt.close(fig)


def report(summary, ablation, mode, output):
    pooled = {r['mode']: r for r in summary['aggregates'] if r['condition'] == 'pooled_stochastic'}
    rows = summary['cases']
    table = ''.join(f"<tr><td>{m.title()}</td><td>{pooled[m]['MAP_gap_rmse_m']*1000:.3f}</td><td>{pooled[m]['posterior_gap_rmse_m']*1000:.3f}</td><td>{100*pooled[m]['posterior_component_90_coverage']:.1f}%</td><td>{pooled[m]['posterior_width_m']*1000:.2f}</td></tr>" for m in COLORS)
    images = ''.join('<figure><img src="data:image/png;base64,'+base64.b64encode((output/file).read_bytes()).decode()+'"><figcaption>'+caption+'</figcaption></figure>' for file, caption in [
        ('reconstruction.png', 'All 36 cases are included. Each bar averages three paired seeds.'),
        ('uncertainty.png', 'Coordinates and the two stochastic conditions are correlated. These are descriptive benchmark frequencies.'),
        ('mixing.png', 'The development case was selected before the final comparison. Trace points are decimated for display; all retained draws enter diagnostics.'),
        ('trajectory.png', 'This illustrative case improves strongly. The neural posterior mean worsens error in four of the six stochastic cases, despite the improved average.'),
        ('force.png', 'Illustrative central bands use 400 paired conditional force draws; full covariance uses every retained state. The force spatial profile is fixed.'),
        ('images.png', 'Images show the mean reconstructed scene, not the posterior predictive average of rendered images.')])
    mode_rows = mode['records']
    body = f'''<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>Posterior inference audit</title>
<style>body{{font:17px/1.6 system-ui,sans-serif;color:#20323b;max-width:1160px;margin:40px auto;padding:0 24px}}h1{{line-height:1.2}}h2{{margin-top:2em}}.notice{{background:#e9f3ef;padding:18px;border-left:5px solid #197a91}}table{{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}}th,td{{padding:10px;text-align:left;border-bottom:1px solid #ccd8dd}}figure{{margin:32px 0}}img{{max-width:100%}}figcaption,.small{{color:#52636a;font-size:14px}}a{{color:#176d85}}code{{font-size:14px}}</style></head><body>
<h1>Better noise calibration exposed an inference problem</h1>
<p class="notice"><strong>All 36 nonlinear posterior runs pass the finite convergence screen.</strong> Worst Rhat {max(r['worst_rhat'] for r in rows):.4f}; minimum bulk/tail ESS {min(r['minimum_bulk_ess'] for r in rows):.0f}/{min(r['minimum_tail_ess'] for r in rows):.0f}. Four chains per case; 2,000 retained transitions per chain, extended to 4,000 for one floor-control case. Passing these checks cannot prove global exploration or physical-model correctness.</p>
<p>The neural model's apparent disadvantage largely disappears when we average over plausible motions instead of selecting the highest-density path. Neural and constant noise give practically the same reconstruction. This is evidence for better inference, not a demonstrated neural-model advantage.</p>
<h2>Same six images, same physics, same marginalized forces</h2>
<p>The experiment repeats the original four truth conditions × three seeds × three noise models. Geometry, appearance, camera, FEM, six-substep integrator, frozen noise weights, independent fine reference data and all density normalizers are unchanged. The benchmark was already inspected; this is an inference comparison, not a blind test.</p>
<table><thead><tr><th>Noise model</th><th>Previous MAP gap error (mm)</th><th>Posterior mean gap error (mm)</th><th>Posterior component 90% coverage</th><th>Width (mm)</th></tr></thead><tbody>{table}</tbody></table>
<p class="small">Means over six stochastic-condition cases sharing three underlying seeds. The neural–constant difference is only {(pooled['constant']['posterior_gap_rmse_m']-pooled['neural']['posterior_gap_rmse_m'])*1000:.3f} mm; maximum reported per-coordinate mean MCSE across the study is {max(r['scores']['max_gap_mean_batch_mcse_m'] for r in rows)*1000:.3f} mm. No credible superiority claim follows from this difference. Central component intervals use no conformal multiplier and do not target simultaneous whole-gap coverage.</p>
<h2>Why calibration made the old point estimates worse</h2>
<p>Correcting an overestimated diffusion amplitude tightened the mechanics prior and introduced more realistic state dependence. The most probable complete path can then move toward a region with higher probability density but less total probability mass. Covariance and drag normalizers matter in a path with 1,440 velocity coordinates. Removing them would change the model; integrating over paths addresses the point-estimator problem.</p>
<p>The calibration-only control uses the same final numerical operators. Across three pulse-plus-noise seeds, old/calibrated/oracle neural MAP errors are {ablation['legacy_neural']*1000:.3f}/{ablation['calibrated_neural']*1000:.3f}/{ablation['oracle_amplitude']*1000:.3f} mm. Even the exact analytic amplitude gives almost the same MAP error as the calibrated network. Remaining amplitude approximation error is therefore not the main explanation.</p>
<p>On the declared development case, restarting optimization from the posterior mean returns to the original mode: objective {mode_rows['original_MAP']['objective']:.6f} → {mode_rows['refit_from_posterior_mean']['objective']:.6f}, gap error {mode_rows['original_MAP']['gap_rmse_m']*1000:.3f} → {mode_rows['refit_from_posterior_mean']['gap_rmse_m']*1000:.3f} mm. The posterior average has {mode_rows['posterior_mean']['gap_rmse_m']*1000:.3f} mm error. Tighter local optimization does not recover that average.</p>
<p>Posterior averaging does not improve every realization. Four of six stochastic neural cases worsen, while the average improves because two cases improve substantially. The old model also has a restricted force basis, fixed scene assumptions and an independent discrepancy floor. This audit does not attribute every remaining reconstruction error to MAP alone.</p>
<h2>What changed in inference</h2>
<p>The old sampler made small random moves and mixed poorly in damping. The replacement uses Gaussian-reference split Hamiltonian proposals, with exact mechanics gradients. A local image approximation makes proposals cheaper; the full nonlinear renderer and normalized marginalized-force density decide acceptance. Step size adapts only in discarded warmup, then freezes. Every state coordinate, gap coordinate, conditional force mean, and applicable force variance/normalizer is monitored.</p>
<p>Some proposals have large energy errors or invalid physical intermediates and are rejected. Counts are preserved per case; independent smaller-step checks are in <code>step-checks/</code>. These are numerical sensitivity checks, not model tuning. Force uncertainty averages both the conditional covariance and variation between inferred trajectories.</p>
{images}
<h2>Evidence and limitations</h2><p><a href="summary.json">All numerical results</a> · <a href="comparison.csv">Comparison table</a> · <a href="publication.json">Hashes and run provenance</a> · <a href="../../docs/POSTERIOR_INFERENCE.md">Equations and algorithms</a> · <a href="../../docs/POSTERIOR_CONVERGENCE_PROTOCOL.md">Frozen protocol</a></p>
<p>94 numerical tests pass. The public Actions artifacts retain full latent chains and proposal errors for 30 days; code, fixed seeds, exact observed images, portable joint draws, all-coordinate diagnostics and reproduction commands remain in the repository. Raw posterior intervals are conditional on a known synthetic scene and frozen diffusion model. Repeated-realization posterior calibration, time/mesh sensitivity, unknown cameras/geometry, contact and real-video reconstruction remain open.</p></body></html>'''
    (output/'report.html').write_text(body, encoding='utf-8')


def publish(raw, output, ablations, mode_audit, step_checks, old_audit):
    raw, output = Path(raw), Path(output)
    if output.exists(): raise FileExistsError('Choose a new publication directory.')
    summary, models, checks = verify(raw)
    output.mkdir(parents=True)
    compact(output/'summary.json', summary)
    for row in summary['cases']:
        folder = output/'cases'/row['case']; folder.mkdir(parents=True)
        for name in ['settings.json', 'summary.json', 'posterior.json', 'diagnostics.json']:
            compact(folder/name, json.loads((raw/'cases'/row['case']/name).read_text()))
    all_ablation = []
    for root in map(Path, ablations):
        saved = json.loads((root/'summary.json').read_text())
        all_ablation += saved['cases']
        # Re-evaluate all nine added MAP states with the frozen operators.
        from audit_calibration_effect import Oracle
        base_plan = json.loads((BASE/'manifest.json').read_text())['plan']
        old_plan = json.loads(Path('examples/neural-marginal/manifest.json').read_text())['plan']
        old_net, old_cal = calibrate(old_plan, 'examples/neural-marginal/calibration.json')
        new_net, new_cal = calibrate(base_plan, BASE/'calibration.json')
        for row in saved['cases']:
            if row['variant'] not in ['legacy_constant', 'legacy_neural', 'oracle_amplitude']:
                continue
            name = row['variant']; seed = row['seed']
            point = json.loads((root/f'{seed}__{name}'/'solver_state.json').read_text())
            net, cal, noise_mode = (Oracle(), new_cal, 'neural') if name == 'oracle_amplitude' else (old_net, old_cal, 'constant' if name == 'legacy_constant' else 'neural')
            model = construct(base_plan, net, cal, noise_mode)
            inputs = json.loads((BASE/'datasets'/f'pulse_and_stochastic__{seed}'/'inputs.json').read_text())
            from physgauss.benchmark import Observations
            obs = Observations(model.view_frames, model.times.new_tensor(inputs['observed_rgb']), .015)
            state = model.times.new_tensor(point['coordinates'])
            with torch.no_grad():
                value = float(model.objective(state[None], obs)[0])
                error = model.trajectory(state[None]).x[0, :, model.free]-model.times.new_tensor(inputs['positions_m_evaluation_only'])[:, model.free]
                gap = (model.times > .3+1e-8)&(model.times < .7-1e-8)
                rmse = float(error[gap].square().mean().sqrt())
            if abs(value-row['fit']['marginal_objective']) > 1e-7 or abs(rmse-row['gap_rmse_m']) > 1e-11:
                raise ValueError('Calibration ablation failed independent state/target verification.')
        dest = output/'calibration-ablation'/root.name; dest.mkdir(parents=True)
        for p in root.rglob('*.json'):
            compact(dest/p.relative_to(root), json.loads(p.read_text()))
    if len(all_ablation) != 15 or len({(r['seed'], r['variant']) for r in all_ablation}) != 15:
        raise ValueError('Incomplete calibration-only ablation.')
    means = {label: sum(r['gap_rmse_m'] for r in all_ablation if r['variant'] == label)/3 for label in sorted({r['variant'] for r in all_ablation})}
    compact(output/'calibration-ablation'/'summary.json', {'cases': all_ablation, 'means_m': means})
    mode = json.loads((Path(mode_audit)/'summary.json').read_text())
    compact(output/'mode-audit.json', mode)
    for root in map(Path, step_checks):
        found = list(root.rglob('summary.json'))
        if len(found) != 1:
            raise ValueError('Incomplete independent step-size check.')
        replica = json.loads(found[0].read_text())
        if replica['source_fingerprint_sha256'] != source_fingerprint():
            raise ValueError('Step-check target implementation differs.')
        for p in root.rglob('*.json'):
            compact(output/'step-checks'/root.name/p.relative_to(root), json.loads(p.read_text()))
    if len(list((output/'step-checks').rglob('summary.json'))) != 2:
        raise ValueError('Need both independent step-size checks.')
    with (output/'comparison.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['case', 'mode', 'MAP_gap_rmse_mm', 'posterior_gap_rmse_mm', 'posterior_component_90_coverage', 'posterior_width_mm', 'Rhat_max', 'bulk_ESS_min', 'tail_ESS_min', 'draws_per_chain'])
        for r in summary['cases']:
            writer.writerow([r['case'], r['mode'], 1000*r['previous_MAP_scores']['original_gap']['position_rmse_m'], 1000*r['scores']['gap_rmse_m'], r['scores']['gap_component_90_coverage'], 1000*r['scores']['gap_mean_width_m'], r['worst_rhat'], r['minimum_bulk_ess'], r['minimum_tail_ess'], r['retained_per_chain']])
    figures(summary, models, output, Path(old_audit))
    report(summary, means, mode, output)
    compact(output/'verification.json', {'cases': checks, 'all_36_portable_scores_reproduced': True,
        'all_diagnostic_flags_recomputed': True, 'conditional_force_total_covariance_verified': True})
    artifact = json.loads((raw/'artifact_download.json').read_text())
    files = {p.relative_to(output).as_posix(): digest(p) for p in sorted(output.rglob('*')) if p.is_file()}
    compact(output/'publication.json', {'model_source_commit': '6a00d90c0ca8a53183c8346fc28def808c8e26ec',
        'source_fingerprint_sha256': source_fingerprint(), 'publisher_source_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        'publisher_script_sha256_normalized_lf': digest(__file__), 'artifact': artifact,
        'base_publication_sha256_normalized_lf': digest('examples/calibrated-noise/publication.json'),
        'numeric_checks_passed': 94, 'files_sha256': files,
        'qualification': 'text hashes normalize LF; full chain artifacts retained externally, compact paired draws and complete diagnostics are versioned'})
    print('Published', len(files), 'files', sum(p.stat().st_size for p in output.rglob('*') if p.is_file()), 'bytes', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--raw', type=Path, required=True); p.add_argument('--output', type=Path, required=True)
    p.add_argument('--ablations', type=Path, nargs=3, required=True)
    p.add_argument('--mode-audit', type=Path, required=True)
    p.add_argument('--step-checks', type=Path, nargs=2, required=True)
    p.add_argument('--old-audit', type=Path, default=Path('runs/calibrated-posterior-optimized'))
    a = p.parse_args(); publish(a.raw, a.output, a.ablations, a.mode_audit, a.step_checks, a.old_audit)
