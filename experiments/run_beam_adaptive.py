"""Audit temporal misspecification and compare adaptive force-prior policies."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import gzip
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from beam_boundary import (candidates, response, force_covariance, observation_design,
    observation_matrix, mixture_cdf, mixture_quantile, uncertainty_metrics, simulation_audit)
from beam_active import (State, action_pool, action_models, update, select_action,
                         predictive, target_risk)
from beam_adaptive import AdaptiveForcePosterior, select_adaptive_action
from run_beam_active import save_gzip, bootstrap_difference

POLICIES = ['active', 'random', 'fixed']
METHODS = [f'{prior}/{policy}' for prior in ['fixed_2s', 'adaptive'] for policy in POLICIES]+['oracle_length/fixed']
COLORS = ['#007f86', '#ad5269', '#c28216']


def generate_data(config, spec, index):
    if 'source' in spec:
        old = json.loads(gzip.decompress(Path(spec['source']).read_bytes()))
        keys = ['labels', 'force_N', 'initial_design', 'initial_observations_pixels', 'actions',
                'potential_action_observations_pixels', 'reference_x_m',
                'reference_response_m_per_N', 'random_action_orders']
        data = {k: old[k] for k in keys}
        data['true_lengths_s'] = [spec['true_length_s']]*len(data['labels'])
        data['replay_source_sha256'] = hashlib.sha256(Path(spec['source']).read_bytes()).hexdigest()
        if 'case_limit' in config:
            for k in ['labels', 'force_N', 'initial_observations_pixels', 'potential_action_observations_pixels', 'random_action_orders', 'true_lengths_s']:
                data[k] = data[k][:config['case_limit']]
        return data
    n = min(spec['cases'], config.get('case_limit', spec['cases']))
    rng = np.random.default_rng(config['seed']+index*10000)
    bounds = candidates(); x = np.linspace(0, 1, 81); design = observation_design('sparse')
    labels = rng.integers(len(bounds), size=n)
    if isinstance(spec['true_length_s'], str):
        prob = np.array(config['length_prior_weights'], dtype=float); prob /= prob.sum()
        lengths = rng.choice(config['lengths_s'], size=n, p=prob)
    else:
        lengths = np.full(n, spec['true_length_s'])
    force = np.stack([np.linalg.cholesky(force_covariance(np.arange(1, 9), length=l))@rng.normal(size=8) for l in lengths])
    reference = np.stack([response(b, x, config['truth_elements']) for b in bounds])
    matrices = np.stack([observation_matrix(b, design, config['truth_elements']) for b in bounds])
    initial = np.einsum('nmt,nt->nm', matrices[labels], force)
    initial += rng.normal(scale=config['pixel_noise_std'], size=initial.shape)
    h, offset, _ = action_models(bounds, action_pool(), config['truth_elements'], config['pixel_noise_std'])
    potential = np.einsum('ant,nt->na', h[:, labels], force)+offset[:, labels].T
    potential += rng.normal(scale=config['pixel_noise_std'], size=potential.shape)
    rr = np.random.default_rng(config['seed']+index*10000+5000)
    return {'labels': labels.tolist(), 'force_N': force.tolist(), 'true_lengths_s': lengths.tolist(),
        'initial_design': design, 'initial_observations_pixels': initial.tolist(), 'actions': action_pool(),
        'potential_action_observations_pixels': potential.tolist(), 'reference_x_m': x.tolist(),
        'reference_response_m_per_N': reference.tolist(),
        'random_action_orders': [rr.permutation(len(action_pool())).tolist() for _ in range(n)]}


def setup(config, lengths):
    bounds = candidates(); x = np.linspace(0, 1, 81)
    engine = AdaptiveForcePosterior(bounds, observation_design('sparse'), lengths,
        config['pixel_noise_std'], config['inference_elements'],
        config['length_prior_weights'] if list(lengths)==config['lengths_s'] else None)
    spatial = np.stack([response(b, x, config['inference_elements']) for b in bounds])[engine.boundary_index]
    utility = np.stack([response(b, np.array(config['utility_hidden_x_m']), config['inference_elements']) for b in bounds])[engine.boundary_index]
    models = engine.expand_models(action_models(bounds, action_pool(), config['inference_elements'], config['pixel_noise_std']))
    return engine, spatial, utility, models


def summarize(records, truths):
    w, m, v = [np.stack([r[k] for r in records]) for k in ['weights', 'means', 'variances']]
    uncertainty = uncertainty_metrics(truths, w, m, v)
    lo, hi = [mixture_quantile(q, w, m, v) for q in [.05, .95]]
    pit = mixture_cdf(truths, w, m, v)
    score = hi-lo+20*(lo-truths)*(truths<lo)+20*(truths-hi)*(truths>hi)
    for i, r in enumerate(records):
        r['tip_mean_m'] = float(w[i]@m[i]); r['tip_lo90_m'] = float(lo[i]); r['tip_hi90_m'] = float(hi[i])
        r['tip_pit'] = float(pit[i]); r['tip_interval_score90_mm'] = float(score[i]*1000)
        for k in ['weights', 'means', 'variances']: del r[k]
    return {'hidden_rmse_mm': float(np.mean([r['hidden_rmse_mm'] for r in records])),
        'hidden_predictive_sd_mm': float(np.sqrt(np.mean([r['hidden_risk_m2'] for r in records]))*1000),
        'tip_interval_score90_mm': float(score.mean()*1000), 'tip_uncertainty': uncertainty,
        'mean_length_probabilities': np.mean([r['length_probabilities'] for r in records], axis=0).tolist()}


def run_population(config, spec, index, output):
    data = generate_data(config, spec, index); n = len(data['labels']); budget = config['budget']
    setups = {'fixed_2s': setup(config, [2.]), 'adaptive': setup(config, config['lengths_s'])}
    # Oracle is a diagnostic only: the true TIMESCALE is supplied, not supports
    # or force amplitudes. No oracle value enters either compared estimator.
    oracle_setups = {l: setup(config, [l]) for l in sorted(set(data['true_lengths_s']))}
    fits = {k: s[0].fit(data['initial_observations_pixels']) for k, s in setups.items()}
    oracle_fits = {l: s[0].fit(data['initial_observations_pixels']) for l, s in oracle_setups.items()}
    x = np.array(data['reference_x_m']); hidden = (x<.3)|(x>.8)
    reference = np.array(data['reference_response_m_per_N']); force = np.array(data['force_N'])
    potential = np.array(data['potential_action_observations_pixels'])
    truth_tip = force[:, 4]*reference[np.array(data['labels']), -1]
    records = {m: [[] for _ in range(budget+1)] for m in METHODS}
    logs = {m: [] for m in METHODS}; audits = []
    order_counts = {}; max_mass_error = 0.; max_tower_error = 0.
    first = {'x_m': x.tolist(), 'truth_at_5s_m': (force[0, 4]*reference[data['labels'][0]]).tolist(), 'methods': {}}
    for i in range(n):
        truth = force[i, :, None]*reference[data['labels'][i], None, :]
        for name in METHODS:
            prior, policy = name.split('/')
            if prior == 'oracle_length':
                true_length = data['true_lengths_s'][i]
                engine, spatial, utility, models = oracle_setups[true_length]; w, m = oracle_fits[true_length]
            else:
                engine, spatial, utility, models = setups[prior]; w, m = fits[prior]
            state = State(w[i].copy(), m[i].copy(), engine.cov.copy())
            available = list(range(len(action_pool()))); history = []
            for step in range(budget+1):
                prediction = np.einsum('c,ct,cx->tx', state.weights, state.mean, spatial)
                record = {'hidden_rmse_mm': float(np.sqrt(np.mean((prediction[:, hidden]-truth[:, hidden])**2))*1000),
                    'hidden_risk_m2': target_risk(state, spatial[:, hidden]),
                    'weights': state.weights.copy(), 'means': state.mean[:, 4]*spatial[:, -1],
                    'variances': state.cov[:, 4, 4]*spatial[:, -1]**2,
                    'length_probabilities': engine.length_probabilities(state.weights).tolist()}
                records[name][step].append(record)
                if i == 0 and step in [0, budget]:
                    qmean = state.mean[:, 4, None]*spatial; qvar = state.cov[:, 4, 4, None]*spatial**2
                    ww = np.broadcast_to(state.weights, (len(x), len(state.weights)))
                    first['methods'][f'{name}_{step}'] = {'mean_m': prediction[4].tolist(),
                        'lo_m': mixture_quantile(.05, ww, qmean.T, qvar.T).tolist(),
                        'hi_m': mixture_quantile(.95, ww, qmean.T, qvar.T).tolist()}
                if step == budget: break
                scores = None
                if policy == 'active' and prior == 'adaptive':
                    chosen, scores, diagnostics = select_adaptive_action(state, models, utility, available, config['quadrature_tolerance'])
                    for d in diagnostics.values():
                        key = str(d['order']); order_counts[key] = order_counts.get(key, 0)+1
                        max_mass_error = max(max_mass_error, d['mass_error']); max_tower_error = max(max_tower_error, d['relative_tower_error'])
                    if i < config['quadrature_audit_cases']:
                        fine_best, fine, _ = select_adaptive_action(state, models, utility, available, config['quadrature_audit_tolerance'])
                        scale = target_risk(state, utility)
                        audits.append({'case': i, 'step': step, 'chosen': chosen, 'fine_best': fine_best,
                            'relative_score_error': float(max(abs(scores[a]-fine[a]) for a in available)/scale),
                            'relative_decision_regret': float((fine[chosen]-fine[fine_best])/scale)})
                elif policy == 'active':
                    chosen, scores = select_action(state, models, utility, available, 15)
                elif policy == 'random': chosen = data['random_action_orders'][i][step]
                else: chosen = config['fixed_action_indices'][step]
                # The selected real outcome is accessed only AFTER selection.
                value = potential[i, chosen]
                mu, var, _ = predictive(state, models[0][chosen], models[1][chosen], models[2][chosen])
                pit = float(mixture_cdf(value, state.weights, mu, var))
                logpdf = np.log(np.maximum(state.weights, 1e-300))-.5*((value-mu)**2/var+np.log(2*np.pi*var))
                nll = -float(logpdf.max()+np.log(np.exp(logpdf-logpdf.max()).sum()))
                history.append({'action': chosen, 'observed_pixels': float(value), 'predictive_pit': pit,
                                'predictive_nll': nll, 'expected_posterior_risks_m2': scores})
                state = update(state, models[0][chosen], models[1][chosen], models[2][chosen], value)
                available.remove(chosen)
            logs[name].append(history)
        if (i+1)%25==0 or i+1==n: print(f'{spec["name"]}: {i+1}/{n}', flush=True)
    summary = {name: [summarize(stage, truth_tip) for stage in stages] for name, stages in records.items()}
    comparisons = {}
    pairs = [('adaptive/active', 'adaptive/random'), ('adaptive/active', 'adaptive/fixed')]
    pairs += [(f'adaptive/{p}', f'fixed_2s/{p}') for p in POLICIES]
    for a, b in pairs:
        comparisons[f'{a} minus {b}'] = [bootstrap_difference(
            [r['hidden_rmse_mm'] for r in records[a][step]], [r['hidden_rmse_mm'] for r in records[b][step]],
            config['seed'], config['bootstrap_resamples']) for step in range(budget+1)]
    gates = {}
    for name, stages in summary.items():
        gates[name] = []
        for step, stage in enumerate(stages):
            u = stage['tip_uncertainty']
            gates[name].append({'step': step, 'pass': bool(abs(u['0.9']['coverage']-.9)<=3*np.sqrt(.09/n)
                                                         and u['pit_ks']<=u['pit_ks_99pct_reference'])})
    prequential = {}
    for name, histories in logs.items():
        prequential[name] = [{'mean_nll': float(np.mean([h[s]['predictive_nll'] for h in histories])),
            'extreme_5pct_fraction': float(np.mean([h[s]['predictive_pit']<.025 or h[s]['predictive_pit']>.975 for h in histories]))}
            for s in range(budget)]
    result = {'cases': n, 'specification': spec, 'methods': summary, 'paired_rmse_by_step': comparisons,
        'scalar_calibration_checks': gates, 'prequential_selected_observations': prequential,
        'quadrature_audit': audits, 'quadrature_order_counts': order_counts,
        'max_mass_error': max_mass_error, 'max_relative_tower_error': max_tower_error,
        'max_quadrature_decision_regret': max(a['relative_decision_regret'] for a in audits),
        'actions_by_method': {m: np.bincount([h['action'] for case in logs[m] for h in case], minlength=13).tolist() for m in METHODS}}
    data['action_logs'] = logs; data['per_case_metrics'] = records
    save_gzip(output/f'{spec["name"]}-dataset.json.gz', data)
    (output/f'{spec["name"]}-results.json').write_text(json.dumps(result, indent=2))
    (output/f'{spec["name"]}-first-case.json').write_text(json.dumps(first, indent=2))
    return result


def analytic_prior_audit():
    slow = force_covariance(np.arange(1, 9), length=2.)
    fast = force_covariance(np.arange(1, 9), length=.8)
    obs = [0, 3, 7]; sd = {}
    for name, k in [('slow', slow), ('fast', fast)]:
        c = k-k[:, obs]@np.linalg.solve(k[np.ix_(obs, obs)], k[obs, :])
        sd[name] = float(np.sqrt(c[4, 4]))
    return {'condition_number_slow': float(np.linalg.cond(slow)), 'condition_number_fast': float(np.linalg.cond(fast)),
        'expected_fast_force_slow_prior_standardized_energy_per_dof': float(np.trace(np.linalg.solve(slow, fast))/8),
        'force_sd_at_5s_given_exact_forces_at_1_4_8s_N': sd,
        'interpretation': 'The smooth prior strongly penalizes high-frequency modes. Conditional covariance is independent of observed values at fixed hyperparameters: disagreement can shift means/support weights without widening the within-component force covariance. The adaptive model updates timescale weights and retains between-timescale variance.'}


def make_report(output, results):
    arms = list(results['populations']); budget = results['protocol']['budget']
    fig, axes = plt.subplots(len(arms), 3, figsize=(15, 3.5*len(arms)), squeeze=False)
    rows = []
    for row, arm in enumerate(arms):
        r = results['populations'][arm]
        for p, color in zip(POLICIES, COLORS):
            for prior, ls in [('adaptive', '-'), ('fixed_2s', '--')]:
                metrics = r['methods'][f'{prior}/{p}']; steps = range(len(metrics))
                axes[row, 0].plot(steps, [m['hidden_rmse_mm'] for m in metrics], ls, color=color, label=f'{prior}/{p}')
                axes[row, 1].plot(steps, [m['tip_uncertainty']['0.9']['coverage'] for m in metrics], ls, color=color)
                axes[row, 2].plot(steps, [m['tip_uncertainty']['0.9']['mean_width']*1000 for m in metrics], ls, color=color)
        oracle = r['methods']['oracle_length/fixed']
        axes[row, 0].plot(range(budget+1), [m['hidden_rmse_mm'] for m in oracle], ':', color='black', label='oracle length/fixed')
        axes[row, 1].axhline(.9, color='black', ls=':'); axes[row, 1].set_ylim(0, 1.02)
        for j, title in enumerate(['Hidden RMSE (mm)', '90% hidden-tip coverage', '90% hidden-tip width (mm)']):
            axes[row, j].set(title=f'{arm}: {title}', xlabel='Extra scalar measurements')
        for name, stages in r['methods'].items():
            m = stages[-1]; u = m['tip_uncertainty']['0.9']
            rows.append(f'<tr><td>{arm}</td><td>{name}</td><td>{m["hidden_rmse_mm"]:.3f}</td><td>{u["coverage"]:.1%}</td><td>{u["mean_width"]*1000:.3f}</td><td>{m["tip_interval_score90_mm"]:.3f}</td></tr>')
    axes[0, 0].legend(fontsize=7); fig.tight_layout(); fig.savefig(output/'comparison.png', dpi=150); plt.close(fig)
    fig, axes = plt.subplots(1, len(arms), figsize=(4.5*len(arms), 4), squeeze=False)
    for ax, arm in zip(axes[0], arms):
        probabilities = np.array([m['mean_length_probabilities'] for m in results['populations'][arm]['methods']['adaptive/active']])
        for j, length in enumerate(results['protocol']['lengths_s']): ax.plot(probabilities[:, j], 'o-', label=f'{length}s')
        ax.set(title=arm, xlabel='Extra measurements', ylabel='Mean timescale posterior probability', ylim=(0, 1))
    axes[0, 0].legend(fontsize=8); fig.tight_layout(); fig.savefig(output/'timescales.png', dpi=150); plt.close(fig)
    text = f'''<!doctype html><html><head><meta charset="utf-8"><title>Adaptive force prior</title>
<style>body{{font:16px/1.6 system-ui;max-width:1200px;margin:40px auto;padding:0 24px;color:#172a35}}img{{width:100%}}table{{border-collapse:collapse;width:100%}}td,th{{padding:7px;border-bottom:1px solid #ddd;text-align:left}}</style></head><body>
<h1>Adaptive force timescales for active beam measurements</h1>
<p>The policy uses observations to update the posterior, then selects the measurement with the greatest expected reduction in hidden-deformation uncertainty. It is a one-step numerical decision rule, not a neural or reinforcement-learning policy. The score includes uncertainty over supports, forces and now force timescale.</p>
<p>The old prior fixes the correlation length at 2s. The new prior integrates over six timescales from 0.35 to 3s and updates their weights from acquired observations only. It retains the full mixture instead of locking onto one best timescale. Material, noise scale and force amplitude remain specified.</p>
<p>Solid curves: adaptive prior. Dashed curves: original fixed prior. Colors identify active/random/fixed action policies. The same initial data, potential measurements and budget are shared. Oracle-timescale fixed actions are an explicitly privileged diagnostic, never used by the policy.</p>
<img src="comparison.png" alt="Original versus adaptive force priors by action budget">
<h2>Final results after {budget} actions</h2><table><tr><th>Population</th><th>Model/policy</th><th>RMSE mm</th><th>90% coverage</th><th>Width mm</th><th>Interval score mm ↓</th></tr>{''.join(rows)}</table>
<img src="timescales.png" alt="Observed-data updates of force timescale probabilities">
<h2>Audit and interpretation</h2><p>Under the 2s prior, fast-force trajectories have expected standardized prior energy {results['prior_audit']['expected_fast_force_slow_prior_standardized_energy_per_dof']:.1f} per degree of freedom (1 would be expected under the prior). The problem is strong suppression of rapid force modes, not an unconverged sampling chain. Even with exactly observed forces at 1,4,8s, the 5s force SD is 0.022N under the smooth model versus 0.053N under the faster model.</p>
<p>The slow/fast replay populations reuse the prior study's data to diagnose the change. Fresh hierarchical draws test calibration under the complete new prior. The fresh 0.5s population is off the inference grid and tests limited misspecification. Fixed-timescale conditional coverage and marginal hierarchical calibration are distinct claims; see the full metrics and all-stage gates.</p>
<p>The beam is quasistatic with replayable loads, known cameras/material/correspondences and ideal calibrated probes. Tracks are fitted, not rendered pixels. This is not an online camera controller or a guarantee under jumps, contact, unknown amplitudes or changing smoothness within an episode. Reducing uncertainty alone does not establish correctness.</p>
<p><a href="results.json">Metrics, paired differences and all-stage checks</a> · <a href="execution.json">Execution provenance</a> · <a href="verification.json">Full archive replay</a></p></body></html>'''
    (output/'report.html').write_text(text, encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--config', default='configs/beam_adaptive.json')
    parser.add_argument('--output', default='runs/beam-adaptive')
    parser.add_argument('--workers', type=int, default=1); args = parser.parse_args()
    config = json.loads(Path(args.config).read_text()); output = Path(args.output); output.mkdir(parents=True, exist_ok=True)
    start = time.time(); results = {'protocol': config, 'prior_audit': analytic_prior_audit(), 'populations': {}}
    if args.workers > 1:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(run_population, config, spec, index, output): spec['name']
                       for index, spec in enumerate(config['populations'])}
            completed = {futures[f]: f.result() for f in as_completed(futures)}
        results['populations'] = {spec['name']: completed[spec['name']] for spec in config['populations']}
    else:
        for index, spec in enumerate(config['populations']):
            results['populations'][spec['name']] = run_population(config, spec, index, output)
    results['simulation_audit'] = simulation_audit(candidates()); results['elapsed_seconds'] = time.time()-start
    (output/'results.json').write_text(json.dumps(results, indent=2)); make_report(output, results)
    files = ['experiments/run_beam_adaptive.py', 'experiments/beam_adaptive.py', 'experiments/beam_active.py',
             'experiments/beam_boundary.py', 'experiments/run_beam_active.py', 'tests/test_beam_adaptive.py', args.config]
    manifest = {'utc_time': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), 'python': platform.python_version(),
        'numpy': np.__version__, 'matplotlib': matplotlib.__version__,
        'host': 'GitHub Actions CPU' if os.getenv('GITHUB_ACTIONS') else 'local CPU',
        'base_git_head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        'elapsed_seconds': results['elapsed_seconds'], 'cpu_workers': args.workers,
        'source_sha256_lf': {p: hashlib.sha256(Path(p).read_bytes().replace(b'\r\n', b'\n')).hexdigest() for p in files},
        'dataset_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in output.glob('*-dataset.json.gz')}}
    (output/'execution.json').write_text(json.dumps(manifest, indent=2))
    print('Completed: '+json.dumps({'seconds': results['elapsed_seconds'], 'populations': list(results['populations'])}), flush=True)


if __name__ == '__main__': main()
