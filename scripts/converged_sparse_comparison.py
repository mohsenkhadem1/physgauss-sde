"""Exact-target posterior means versus the published sparse-view MAP fits.

All physical models, frozen noise weights, images and references are preserved.
Only inference and the point estimator change. Truth is accessed by score(),
after chain diagnostics, and never by proposal construction or stopping rules.
"""
import argparse
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import time
import torch
from physgauss.accurate_study import digest_tensor
from physgauss.benchmark import Observations
from physgauss.camera import orbit
from physgauss.chain_diagnostics import diagnose_quantities
from physgauss.cli import source_fingerprint
from physgauss.likelihood_audit import _environment
from physgauss.neural_marginal_study import calibrate, construct
from physgauss.split_hmc import SplitHMCConfig, sample_split_hmc
from physgauss.study import write_json
from physgauss.white_posterior import WhitePosterior


BASE = Path('examples/calibrated-noise/comparison')


def setup(case, output):
    manifest = json.loads((BASE/'manifest.json').read_text())
    plan = manifest['plan']; torch.set_num_threads(plan['threads'])
    import torch.backends.opt_einsum as backend
    backend.enabled = True; backend.strategy = 'greedy'
    network, calibration = calibrate(plan, BASE/'calibration.json')
    folder = BASE/'cases'/case
    row = json.loads((folder/'summary.json').read_text())
    coordinates = json.loads((folder/'solver_state.json').read_text())
    model = construct(plan, network, calibration, row['mode'])
    # Read only observed pixels here. Evaluation states are used only by score.
    data_file = BASE/'datasets'/f"{row['condition']}__{row['seed']}"/'inputs.json'
    rgb = model.times.new_tensor(json.loads(data_file.read_text())['observed_rgb'])
    obs = Observations(model.view_frames, rgb, .015)
    if digest_tensor(rgb) != row['observations_sha256']:
        raise ValueError('Observed pixels differ from the matched comparison.')
    mean = model.times.new_tensor(coordinates['coordinates'])
    with torch.no_grad(): value = float(model.objective(mean[None], obs)[0])
    if abs(value-row['fit']['marginal_objective']) > 1e-7:
        raise ValueError('Legacy normalized target was not reproduced.')
    fit_file = output/'reference.pt'
    if fit_file.exists():
        fit = torch.load(fit_file, weights_only=True)
        torch.testing.assert_close(fit['mean'], mean, rtol=0, atol=0)
    else:
        precision, _ = model.fisher_precision(mean, obs)
        fit = {'mean': mean, 'precision_cholesky': torch.linalg.cholesky(precision)}
        torch.save(fit, fit_file)
    return model, obs, fit, row, data_file, manifest


@torch.no_grad()
def summarize_chains(model, target, chains):
    c, n, d = chains['white'].shape
    coordinates = target.decode(chains['white'].flatten(0, 1))
    gap = (model.times > .3+1e-8)&(model.times < .7-1e-8)
    path = model.trajectory(coordinates).x.reshape(c, n, 21, 8, 3)
    gap_path = path[:, :, gap][:, :, :, model.free].flatten(2)
    means, covariances, logdet_q, logdet_a, quadratic = [], [], [], [], []
    for z in coordinates.split(16):
        e = model.elimination_at(z)
        r = model.residual(z)
        m = e.conditional_mean(r)
        white = e.whiten_q(r)-torch.einsum('bsdk,bk->bsd', e.white_mapping, m)
        means.append(m); covariances.append(e.conditional_covariance)
        logdet_q.append(e.logdet_q); logdet_a.append(e.logdet-e.logdet_q)
        quadratic.append(white.square().sum((-1, -2))+m.square().sum(-1))
    means, covariances, logdet_q, logdet_a, quadratic = [torch.cat(v) for v in
        (means, covariances, logdet_q, logdet_a, quadratic)]
    log_target = chains['log_ratio']-target.center-.5*chains['white'].square().sum(-1)
    monitor = torch.cat((coordinates.reshape(c, n, d), gap_path, log_target[..., None],
        means.reshape(c, n, 15), quadratic.reshape(c, n, 1)), -1)
    names = ([f'state_{i}' for i in range(d)]+[f'gap_xyz_{i}' for i in range(84)]+['log_target']
        +[f'force_conditional_mean_{i}' for i in range(15)]+['profile_quadratic'])
    # Fixed Q/G implies exactly constant conditional force covariance and
    # normalizers in the floor control; these are known constants, not samples.
    if model.internal_config.mode != 'floor':
        monitor = torch.cat((monitor, covariances.diagonal(dim1=-2, dim2=-1).reshape(c, n, 15),
            torch.stack((logdet_q, logdet_a), -1).reshape(c, n, 2)), -1)
        names += [f'force_conditional_variance_{i}' for i in range(15)]+['logdet_Q', 'logdet_A']
    diagnostics = diagnose_quantities(monitor, names)
    physical = model.prior.mean+model.prior.std*coordinates[:, :8]
    physical[:, :2] = physical[:, :2].exp()
    conditional = covariances.mean(0); between = torch.cov(means.T)
    force_cov = conditional+between
    return {'coordinates': coordinates, 'path': path, 'gap_path': gap_path, 'physical': physical,
        'conditional_force_means': means, 'conditional_force_covariances': covariances,
        'force_mean': means.mean(0), 'force_covariance': force_cov,
        'mean_conditional_covariance': conditional, 'between_covariance': between,
        'log_target': log_target, 'diagnostics': diagnostics,
        'normalizers': torch.stack((logdet_q, logdet_a, quadratic), -1).reshape(c, n, 3)}


def compact_diagnostics(diagnostics):
    rows = diagnostics['quantities']
    return {k: v for k, v in diagnostics.items() if k != 'quantities'} | {
        'names': list(rows), 'columns': {key: [v[key] for v in rows.values()] for key in next(iter(rows.values()))}}


@torch.no_grad()
def score(model, target, chains, result, data_file, row, output):
    inputs = json.loads(data_file.read_text())
    truth = model.times.new_tensor(inputs['positions_m_evaluation_only'])
    gap = (model.times > .3+1e-8)&(model.times < .7-1e-8)
    pop = result['path'][:, :, :, model.free].flatten(0, 1)
    lower, upper = torch.quantile(pop, pop.new_tensor([.05, .95]), dim=0)
    mean = pop.mean(0)
    error = mean-truth[:, model.free]
    # Batch-mean MCSE is an additional numerical error indicator, not coverage.
    c, n = result['gap_path'].shape[:2]
    batch = max(20, int(n**.5)); blocks = n//batch
    batches = result['gap_path'][:, :blocks*batch].reshape(c, blocks, batch, 84).mean(2).flatten(0, 1)
    mcse = batches.std(0)/len(batches)**.5
    scores = {'gap_rmse_m': float(error[gap].square().mean().sqrt()),
        'all_position_rmse_m': float(error.square().mean().sqrt()),
        'gap_component_90_coverage': float(((truth[:, model.free] >= lower)&(truth[:, model.free] <= upper))[gap].double().mean()),
        'gap_mean_width_m': float((upper-lower)[gap].mean()),
        'max_gap_mean_batch_mcse_m': float(mcse.max()),
        'per_chain_gap_mean_rmse_m': [float((p.mean(0)-truth[:, model.free]).square()[gap].mean().sqrt()) for p in result['path'][:, :, :, model.free]],
        'qualification': 'posterior mean and central component intervals; finite-chain diagnostics required; no conformal scaling or repeated-case coverage guarantee'}
    rng = torch.Generator().manual_seed(chains['config']['seed']+61000)
    # Select equally spaced indices separately within each chain. Preserve all
    # joint parameter/path/force associations in the portable draws.
    indices = torch.linspace(0, n-1, 100).round().long()
    selected = (torch.arange(c)[:, None]*n+indices[None]).flatten()
    fm = result['conditional_force_means'][selected]
    fc = result['conditional_force_covariances'][selected]
    force_draws = fm+(torch.linalg.cholesky(fc)@torch.randn(*fm.shape, 1, dtype=fm.dtype, generator=rng)).squeeze(-1)
    total = model.force_prior.total_force(result['force_mean'][None], model.times)[0]
    pulse = row['condition'] in ['external_pulse', 'pulse_and_stochastic']
    strength = torch.exp(-.5*((model.times-.5)/.08).square()) if pulse else torch.zeros_like(model.times)
    true_force = strength[:, None]*model.mesh.mass[model.free].sum()*model.times.new_tensor([.45, 0., -.30])
    scores['force_total_mean_rmse_N'] = float((total-true_force).square().mean().sqrt())
    mean_x = result['path'].flatten(0, 1).mean(0)[None]
    for offset in [0, 200]:
        eye, rotation = orbit(model.times, offset_degrees=offset)
        rgb = torch.cat([model.renderer(mean_x[:, a:a+3], eye[a:a+3], rotation[a:a+3])[0] for a in range(0, 21, 3)])
        clean = model.times.new_tensor(inputs['clean_rgb'][str(offset)])
        scores[f'camera_{offset}_mean_scene_clean_psnr_db'] = float(-10*torch.log10((rgb-clean).square().mean()))
    write_json(output/'posterior.json', {'case': row['case'], 'positions_mean_m': result['path'].flatten(0, 1).mean(0).tolist(),
        'free_position_q05_m': lower.tolist(), 'free_position_q95_m': upper.tolist(),
        'normalized_state_mean': result['coordinates'].mean(0).tolist(),
        'parameter_names': ['young_Pa', 'drag_per_s', 'offset_x_m', 'offset_y_m', 'offset_z_m', 'velocity_x_m_s', 'velocity_y_m_s', 'velocity_z_m_s'],
        'parameter_mean': result['physical'].mean(0).tolist(), 'parameter_covariance': torch.cov(result['physical'].T).tolist(),
        'force_coefficient_mean': result['force_mean'].tolist(), 'force_coefficient_covariance': result['force_covariance'].tolist(),
        'force_mean_conditional_covariance': result['mean_conditional_covariance'].tolist(),
        'force_between_trajectory_covariance': result['between_covariance'].tolist(),
        'force_total_mean_N': total.tolist(),
        'portable_draws': {'chain_indices': torch.arange(c).repeat_interleave(100).tolist(), 'retained_indices': indices.repeat(c).tolist(),
            'gap_free_xyz_m': result['gap_path'].flatten(0, 1)[selected].tolist(),
            'physical_parameters': result['physical'][selected].tolist(), 'force_coefficients': force_draws.tolist()},
        'normalizer_means_per_chain': result['normalizers'].mean(1).tolist(),
        'log_target_traces': result['log_target'][:, indices].tolist(),
        'qualification': '400 decimated joint draws for portability only; all retained draws enter estimates, diagnostics and total force covariance'})
    return scores


def run(case, output, settings):
    output = Path(output)/'cases'/case; output.mkdir(parents=True, exist_ok=True)
    began = time.perf_counter()
    model, obs, fit, row, data_file, manifest = setup(case, output)
    sampler = SplitHMCConfig(**settings['sampler'])
    sampler = replace(sampler, seed=sampler.seed+settings['cases'].index(case))
    metadata = {'case': case, 'source_fingerprint_sha256': source_fingerprint(),
        'base_source_fingerprint_sha256': manifest['source_fingerprint_sha256'],
        'observations_sha256': digest_tensor(obs.rgb), 'reference_mean_sha256': digest_tensor(fit['mean']),
        'plan_sha256': hashlib.sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest(),
        'worker_script_sha256_normalized_lf': hashlib.sha256(Path(__file__).read_bytes().replace(b'\r\n', b'\n')).hexdigest(),
        'sampler': asdict(sampler), 'environment': _environment(),
        'proposal': 'fixed Fisher Gaussian coordinates; exact dynamics gradients and observed image Taylor Hessian; full nonlinear image target in MH acceptance',
        'target_reproduction_abs_error': abs(float(model.objective(fit['mean'][None], obs)[0])-row['fit']['marginal_objective'])}
    settings_file = output/'settings.json'
    if settings_file.exists() and json.loads(settings_file.read_text()) != metadata:
        raise ValueError('Source, environment, plan or input changed; choose a fresh output.')
    write_json(settings_file, metadata)
    target = WhitePosterior(model, obs, fit['mean'], fit['precision_cholesky'], image_taylor=True)
    checkpoint = output/'chains.pt'
    previous = torch.load(checkpoint, weights_only=True) if checkpoint.exists() else None
    history = []
    for budget in settings['draw_budgets']:
        if source_fingerprint() != metadata['source_fingerprint_sha256']:
            raise RuntimeError('Numerical source changed during sampling.')
        if previous is not None and previous['white'].shape[1] > budget:
            continue
        if previous is None:
            cfg = replace(sampler, draws=budget)
        else:
            more = budget-previous['white'].shape[1]
            cfg = replace(sampler, warmup=0, draws=more, step_size=previous['step_size'])
        if previous is None or cfg.draws:
            previous = sample_split_hmc(target, model.state_dimension, cfg, resume=previous,
                progress=lambda s: print(case, s, flush=True))
            torch.save(previous, checkpoint)
        result = summarize_chains(model, target, previous)
        diag = result['diagnostics']
        history.append({'retained_per_chain': budget, 'all_pass': diag['all_monitored_quantities_pass'], 'failures': len(diag['failed_quantities'])})
        write_json(output/'diagnostics.json', compact_diagnostics(diag))
        print(case, 'diagnostic budget', budget, 'failures', len(diag['failed_quantities']), flush=True)
        if diag['all_monitored_quantities_pass']:
            break
    scores = score(model, target, previous, result, data_file, row, output)
    q = result['diagnostics']['quantities']
    summary = {**metadata, 'condition': row['condition'], 'seed': row['seed'], 'mode': row['mode'],
        'scores': scores, 'previous_MAP_scores': row['scores'], 'posterior_status': 'finite_diagnostics_pass' if result['diagnostics']['all_monitored_quantities_pass'] else 'not_validated',
        'monitored_quantities': len(q), 'diagnostic_history': history, 'retained_per_chain': previous['white'].shape[1],
        'worst_rhat': max(v['rank_folded_split_rhat'] or float('inf') for v in q.values()),
        'minimum_bulk_ess': min(v['bulk_ess'] or 0 for v in q.values()), 'minimum_tail_ess': min(v['tail_ess'] or 0 for v in q.values()),
        'transition_counts': previous['counts'], 'frozen_step_size': previous['step_size'],
        'seconds': time.perf_counter()-began,
        'qualification': 'same known synthetic scene and six images; finite diagnostics do not prove global exploration or calibrated model uncertainty; all cases retained, including failures'}
    write_json(output/'summary.json', summary)


def merge(output, settings):
    output = Path(output)
    rows = [json.loads((output/'cases'/case/'summary.json').read_text()) for case in settings['cases']]
    for key in ['source_fingerprint_sha256', 'base_source_fingerprint_sha256', 'plan_sha256', 'worker_script_sha256_normalized_lf']:
        if len({r[key] for r in rows}) != 1:
            raise ValueError('Mixed numerical implementation or settings: '+key)
    if rows[0]['source_fingerprint_sha256'] != source_fingerprint():
        raise ValueError('Aggregate source differs from workers.')
    aggregates = []
    for condition in ['no_load', 'external_pulse', 'stochastic', 'pulse_and_stochastic', 'pooled_stochastic']:
        for mode in ['floor', 'constant', 'neural']:
            selected = [r for r in rows if r['mode'] == mode and (r['condition'] == condition or
                (condition == 'pooled_stochastic' and r['condition'] in ['stochastic', 'pulse_and_stochastic']))]
            aggregates.append({'condition': condition, 'mode': mode, 'cases': len(selected),
                'posterior_gap_rmse_m': sum(r['scores']['gap_rmse_m'] for r in selected)/len(selected),
                'MAP_gap_rmse_m': sum(r['previous_MAP_scores']['original_gap']['position_rmse_m'] for r in selected)/len(selected),
                'posterior_component_90_coverage': sum(r['scores']['gap_component_90_coverage'] for r in selected)/len(selected),
                'MAP_component_90_coverage': sum(r['previous_MAP_scores']['original_gap']['component_90_coverage'] for r in selected)/len(selected),
                'posterior_width_m': sum(r['scores']['gap_mean_width_m'] for r in selected)/len(selected),
                'MAP_width_m': sum(r['previous_MAP_scores']['original_gap']['component_90_mean_width_m'] for r in selected)/len(selected),
                'passed_cases': sum(r['posterior_status'] == 'finite_diagnostics_pass' for r in selected)})
    write_json(output/'summary.json', {'plan': settings, 'cases': rows, 'aggregates': aggregates,
        'all_cases_pass_finite_diagnostics': all(r['posterior_status'] == 'finite_diagnostics_pass' for r in rows),
        'qualification': '36 matched benchmark cases, four chains per case, paired three seeds; previously inspected data; no claim of frequentist or real-video calibration'})


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan', type=Path, default=Path('configs/converged_sparse_comparison.json'))
    p.add_argument('--case-index', type=int)
    p.add_argument('--merge', action='store_true')
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    settings = json.loads(args.plan.read_text())
    if args.merge:
        merge(args.output, settings)
    else:
        if args.case_index is None or not 0 <= args.case_index < len(settings['cases']):
            p.error('Choose a valid --case-index or --merge.')
        run(settings['cases'][args.case_index], args.output, settings)
