"""Paired active/random/fixed beam experiments and reproducible report."""
import argparse
from dataclasses import asdict
import gzip
import hashlib
import html
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
                           ExactBoundaryPosterior, observation_matrix, uncertainty_metrics,
                           mixture_quantile, simulation_audit)
from beam_active import (State, action_pool, action_models, update, target_components,
                         target_risk, select_action, expected_risk)

POLICIES = ['active', 'random', 'fixed']
COLORS = ['#007f86', '#ad5269', '#c28216']


def save_gzip(path, value):
    path.write_bytes(gzip.compress(json.dumps(value, separators=(',', ':')).encode(), mtime=0))


def bootstrap_difference(a, b, seed, count):
    rng = np.random.default_rng(seed)
    delta = np.asarray(a)-np.asarray(b)
    draws = delta[rng.integers(len(delta), size=(count, len(delta)))].mean(axis=1)
    return {'mean_difference': float(delta.mean()),
            'paired_bootstrap_95pct': np.quantile(draws, [.025, .975]).tolist()}


def metric_summary(records, truth_tip):
    weights = np.stack([r['weights'] for r in records])
    means = np.stack([r['tip_means'] for r in records])
    variances = np.stack([r['tip_variances'] for r in records])
    unc = uncertainty_metrics(truth_tip, weights, means, variances)
    lo = mixture_quantile(.05, weights, means, variances)
    hi = mixture_quantile(.95, weights, means, variances)
    score = hi-lo + 20*(lo-truth_tip)*(truth_tip<lo) + 20*(truth_tip-hi)*(truth_tip>hi)
    return {
        'mean_hidden_rmse_mm': float(np.mean([r['hidden_rmse_mm'] for r in records])),
        'mean_missing_time_hidden_rmse_mm': float(np.mean([r['missing_rmse_mm'] for r in records])),
        'root_mean_predictive_variance_mm': float(np.sqrt(np.mean([r['hidden_risk_m2'] for r in records]))*1000),
        'tip_interval_score_90_mm': float(np.mean(score)*1000),
        'hidden_tip_at_5s': unc,
    }


def run_population(config, output, stress=False):
    arm = 'shorter_force' if stress else 'matched'
    n = config['stress_cases'] if stress else config['cases']
    rng = np.random.default_rng(config['seed'] + (100000 if stress else 0))
    bounds = candidates(); actions = action_pool(); budget = config['budget']
    x = np.linspace(0, 1, 81); hidden = (x < .3) | (x > .8)
    inference_n, truth_n = config['inference_elements'], config['truth_elements']
    spatial = np.stack([response(b, x, inference_n) for b in bounds])
    reference = np.stack([response(b, x, truth_n) for b in bounds])
    utility_spatial = np.stack([response(b, np.array(config['utility_hidden_x_m']), inference_n) for b in bounds])
    engine = ExactBoundaryPosterior(bounds, observation_design('sparse'), config['pixel_noise_std'], inference_n)
    models = action_models(bounds, actions, inference_n, config['pixel_noise_std'])
    reference_models = action_models(bounds, actions, truth_n, config['pixel_noise_std'])
    reference_initial = np.stack([observation_matrix(b, engine.design, truth_n) for b in bounds])
    labels = rng.integers(len(bounds), size=n)
    force = rng.normal(size=(n, 8)) @ np.linalg.cholesky(force_covariance(np.arange(1, 9), length=.8 if stress else 2.)).T
    initial = np.einsum('nmt,nt->nm', reference_initial[labels], force)
    initial += rng.normal(scale=config['pixel_noise_std'], size=initial.shape)
    # Potential measurements are sealed away from the policy. Common random
    # numbers give paired comparison; only the selected action is revealed.
    rh, ro, _ = reference_models
    measurements = np.einsum('ant,nt->na', rh[:, labels], force) + ro[:, labels].T
    measurements += rng.normal(scale=config['pixel_noise_std'], size=measurements.shape)
    random_rng = np.random.default_rng(config['seed']+ (300000 if stress else 200000))
    random_orders = np.stack([random_rng.permutation(len(actions)) for _ in range(n)])
    w, m = engine.fit(initial)
    records = {p: [[] for _ in range(budget+1)] for p in POLICIES}
    logs = {p: [] for p in POLICIES}
    audit = []
    first = {'x_m': x.tolist(), 'truth_at_5s_m': (force[0, 4]*reference[labels[0]]).tolist(), 'policies': {}}
    for i in range(n):
        truth = force[i, :, None]*reference[labels[i], None, :]
        for policy in POLICIES:
            state = State(w[i].copy(), m[i].copy(), engine.cov.copy())
            available = list(range(len(actions))); history = []
            for step in range(budget+1):
                cm, cv = target_components(state, spatial[:, hidden])
                prediction = np.einsum('c,ctx->tx', state.weights, cm)
                error = prediction-truth[:, hidden]
                records[policy][step].append({
                    'hidden_rmse_mm': float(np.sqrt(np.mean(error**2))*1000),
                    'missing_rmse_mm': float(np.sqrt(np.mean(error[[1, 2, 4, 5, 6]]**2))*1000),
                    'hidden_risk_m2': target_risk(state, spatial[:, hidden]),
                    'weights': state.weights.tolist(),
                    'tip_means': (state.mean[:, 4]*spatial[:, -1]).tolist(),
                    'tip_variances': (state.cov[:, 4, 4]*spatial[:, -1]**2).tolist(),
                })
                if i == 0 and step in [0, budget]:
                    qmean = state.mean[:, 4, None]*spatial
                    qvar = state.cov[:, 4, 4, None]*spatial**2
                    ww = np.broadcast_to(state.weights, (len(x), len(bounds)))
                    first['policies'][f'{policy}_{step}'] = {
                        'mean_m': (state.weights@qmean).tolist(),
                        'lo_m': mixture_quantile(.05, ww, qmean.T, qvar.T).tolist(),
                        'hi_m': mixture_quantile(.95, ww, qmean.T, qvar.T).tolist(),
                    }
                if step == budget:
                    break
                if policy == 'active':
                    chosen, scores = select_action(state, models, utility_spatial, available, config['quadrature_order'])
                    if i < config['quadrature_audit_cases']:
                        _, fine = select_action(state, models, utility_spatial, available, config['quadrature_audit_order'])
                        fine_best = min(fine, key=fine.get)
                        audit.append({'case': i, 'step': step, 'chosen': chosen, 'fine_best': fine_best,
                            'relative_max_score_error': float(max(abs(scores[a]-fine[a]) for a in available)/target_risk(state, utility_spatial)),
                            'relative_decision_regret': float((fine[chosen]-fine[fine_best])/target_risk(state, utility_spatial))})
                elif policy == 'random':
                    chosen = int(random_orders[i, step]); scores = None
                else:
                    chosen = config['fixed_action_indices'][step]; scores = None
                history.append({'action': chosen, 'observed_pixels': float(measurements[i, chosen]),
                    'prior_utility_risk_m2': target_risk(state, utility_spatial),
                    'expected_posterior_risks_m2': scores})
                state = update(state, models[0][chosen], models[1][chosen], models[2][chosen], measurements[i, chosen])
                available.remove(chosen)
            logs[policy].append(history)
        if (i+1) % 25 == 0 or i+1 == n:
            print(f'{arm}: {i+1}/{n} paired cases', flush=True)
    truth_tip = force[:, 4]*reference[labels, -1]
    summary = {p: [metric_summary(stage, truth_tip) for stage in records[p]] for p in POLICIES}
    paired = {}; paired_by_step = {}
    for baseline in ['random', 'fixed']:
        paired[baseline] = bootstrap_difference(
            [r['hidden_rmse_mm'] for r in records['active'][-1]],
            [r['hidden_rmse_mm'] for r in records[baseline][-1]], config['seed'], config['bootstrap_resamples'])
        paired[baseline]['units'] = 'mm, active minus baseline; negative favors active'
        paired_by_step[baseline] = [bootstrap_difference(
            [r['hidden_rmse_mm'] for r in records['active'][step]],
            [r['hidden_rmse_mm'] for r in records[baseline][step]], config['seed'], config['bootstrap_resamples'])
            for step in range(budget+1)]
    action_counts = {p: np.bincount([v['action'] for row in logs[p] for v in row], minlength=len(actions)).tolist() for p in POLICIES}
    # Reconstruct all posterior states exactly from initial data and action logs.
    dataset = {'arm': arm, 'boundaries': [asdict(b) for b in bounds], 'labels': labels.tolist(),
        'force_N': force.tolist(), 'initial_design': engine.design, 'initial_observations_pixels': initial.tolist(),
        'actions': actions, 'potential_action_observations_pixels': measurements.tolist(),
        'reference_x_m': x.tolist(), 'reference_response_m_per_N': reference.tolist(),
        'random_action_orders': random_orders.tolist(), 'action_logs': logs,
        'per_case_metrics_and_tip_posteriors': records}
    save_gzip(output/f'{arm}-dataset.json.gz', dataset)
    (output/f'{arm}-first-case.json').write_text(json.dumps(first, indent=2))
    # Same criterion as the prior beam study, applied at every adaptive stage.
    calibration_checks = []
    for p in POLICIES:
        for step, metrics in enumerate(summary[p]):
            u = metrics['hidden_tip_at_5s']
            calibration_checks.append({'policy': p, 'step': step,
                'pass': bool(abs(u['0.9']['coverage']-.9) <= 3*np.sqrt(.09/n)
                             and u['pit_ks'] <= u['pit_ks_99pct_reference'])})
    result = {'cases': n, 'policies': summary, 'paired_final_rmse_comparison': paired,
        'paired_rmse_by_step': paired_by_step,
        'action_counts': action_counts, 'quadrature_audit': audit,
        'max_quadrature_relative_decision_regret': max(a['relative_decision_regret'] for a in audit),
        'max_quadrature_relative_score_error': max(a['relative_max_score_error'] for a in audit),
        'scalar_calibration_checks': calibration_checks,
        'scalar_calibration_gate_pass': all(v['pass'] for v in calibration_checks),
        'max_true_slope_rad': float(np.max(abs(force[:, :, None]*np.diff(reference[labels], axis=1)[:, None, :]/np.diff(x))))}
    (output/f'{arm}-results.json').write_text(json.dumps(result, indent=2))
    return result, first


def plots(output, results, examples):
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, axs = plt.subplots(2, 3, figsize=(15, 8))
    for row, arm in enumerate(['matched', 'shorter_force']):
        for p, color in zip(POLICIES, COLORS):
            metrics = results[arm]['policies'][p]
            xs = np.arange(len(metrics))
            axs[row, 0].plot(xs, [m['mean_hidden_rmse_mm'] for m in metrics], 'o-', label=p, color=color)
            axs[row, 1].plot(xs, [m['hidden_tip_at_5s']['0.9']['coverage'] for m in metrics], 'o-', color=color)
            axs[row, 2].plot(xs, [m['hidden_tip_at_5s']['0.9']['mean_width']*1000 for m in metrics], 'o-', color=color)
        axs[row, 1].axhline(.9, ls='--', c='black', lw=1)
        for col in range(3):
            axs[row, col].set_xlabel('Additional scalar measurements (equal budget)')
        axs[row, 0].set(title=arm.replace('_', ' ').title(), ylabel='Mean per-case hidden RMSE (mm)')
        axs[row, 1].set(title='Always-hidden tip, t = 5 s', ylabel='90% interval coverage', ylim=(0, 1.02))
        axs[row, 2].set(title='Narrower is useful only with coverage', ylabel='90% interval width (mm)')
    axs[0, 0].legend(); fig.tight_layout(); fig.savefig(output/'comparison.png', dpi=160); plt.close(fig)
    first = examples['matched']; x = np.array(first['x_m']); budget = results['protocol']['budget']
    fig, axs = plt.subplots(1, 4, figsize=(16, 4), sharey=True)
    for ax, key, title, color in zip(axs, ['active_0']+[f'{p}_{budget}' for p in POLICIES], ['Initial']+POLICIES, ['#666']+COLORS):
        d = first['policies'][key]
        ax.axvspan(.35, .75, color='#ddd', alpha=.5)
        ax.fill_between(x, np.array(d['lo_m'])*1000, np.array(d['hi_m'])*1000, color=color, alpha=.22)
        ax.plot(x, np.array(d['mean_m'])*1000, color=color, label='Posterior mean')
        ax.plot(x, np.array(first['truth_at_5s_m'])*1000, 'k--', label='80-element reference')
        ax.set(title=title, xlabel='Beam position (m)')
    axs[0].set_ylabel('Displacement at 5 s (mm)'); axs[0].legend(fontsize=8)
    fig.suptitle('Predeclared first case; shaded bands are 90% pointwise intervals')
    fig.tight_layout(); fig.savefig(output/'first-case.png', dpi=160); plt.close(fig)
    counts = results['matched']['action_counts']
    fig, ax = plt.subplots(figsize=(13, 4))
    names = [f"view {a['time_s']}s/{a['x_m']}m" if a['kind']=='view' else f"probe {a['load_x_m']}m" for a in action_pool()]
    for j, (p, color) in enumerate(zip(POLICIES, COLORS)):
        ax.bar(np.arange(len(names))+(j-1)*.25, np.array(counts[p])/results['matched']['cases'], .24, color=color, label=p)
    ax.set_xticks(np.arange(len(names)), names, rotation=30, ha='right')
    ax.set(ylabel='Fraction of cases choosing action', title='Matched-model action allocation over six measurements')
    ax.legend(); fig.tight_layout(); fig.savefig(output/'actions.png', dpi=160); plt.close(fig)


def report(output, results):
    rows = []
    for arm in ['matched', 'shorter_force']:
        for p in POLICIES:
            m = results[arm]['policies'][p][-1]; u = m['hidden_tip_at_5s']['0.9']
            rows.append(f'<tr><td>{arm}</td><td>{p}</td><td>{m["mean_hidden_rmse_mm"]:.3f}</td>'
                f'<td>{u["coverage"]:.1%}</td><td>{1000*u["mean_width"]:.3f}</td><td>{m["tip_interval_score_90_mm"]:.3f}</td></tr>')
    comparisons = []
    for arm in ['matched', 'shorter_force']:
        for p, d in results[arm]['paired_final_rmse_comparison'].items():
            comparisons.append(f'<li>{arm}, active minus {p}: {d["mean_difference"]:.4f} mm; paired bootstrap 95% interval '
                               f'[{d["paired_bootstrap_95pct"][0]:.4f}, {d["paired_bootstrap_95pct"][1]:.4f}] mm.</li>')
    text = f'''<!doctype html><html><head><meta charset="utf-8"><title>Active beam experiments</title>
<style>body{{font:17px/1.6 system-ui;max-width:1180px;margin:40px auto;padding:0 25px;color:#182a36}}img{{width:100%}}table{{border-collapse:collapse;width:100%}}td,th{{padding:10px;text-align:left;border-bottom:1px solid #ccd}}code{{background:#eef;padding:2px}}a{{color:#007f86}}</style></head><body>
<h1>Choose the next measurement to predict hidden deformation</h1>
<p>Active, random and fixed policies start from exactly the same three sparse frames. Each gets six additional scalar measurements. Active planning integrates over unknown forces and support hypotheses and minimizes expected hidden-motion prediction variance. The fixed policy observes all five initially missing times at the most distal available visible point, then applies one calibrated probe.</p>
<p>This is a replayable, quasistatic one-metre beam, with known material and camera geometry. All measurements are noisy tracks, not image-based reconstruction. Actions revisit an original loading state or apply a known load to the unloaded beam. Physical probes have ideal known load, reference and unchanged support; all actions have equal measurement cost, not equal real robot execution time.</p>
<img src="comparison.png" alt="Error, interval coverage and width by acquisition budget">
<h2>After six actions</h2><table><tr><th>Population</th><th>Policy</th><th>Hidden RMSE mm</th><th>90% coverage</th><th>Width mm</th><th>Interval score mm ↓</th></tr>{''.join(rows)}</table>
<p>Errors average per-case RMSE over all eight times and unobserved x&lt;0.3 or x&gt;0.8. Coverage is for the never-observed tip at 5s. Width and the proper interval score distinguish useful confidence from overconfidence. Coverage is prior-predictive across independent cases; it is not simultaneous trajectory coverage.</p>
<ul>{''.join(comparisons)}</ul><p>Intervals above resample entire paired cases, not correlated points. A confidence interval spanning zero does not establish an advantage.</p>
<img src="actions.png" alt="Action choices by policy"><img src="first-case.png" alt="Hidden deformation and posterior bands for predeclared first case">
<h2>Audits and limits</h2><p>Matched scalar calibration gate: <b>{results['matched']['scalar_calibration_gate_pass']}</b>.
Shorter-force-prior stress gate: <b>{results['shorter_force']['scalar_calibration_gate_pass']}</b>. Gates examine 90% coverage and PIT at every stage and are descriptive multiple checks, not a familywise test. The stress arm changes the true force correlation from 2s to 0.8s without telling the estimator; uncertainty reduction alone does not demonstrate correctness.</p>
<p>Mechanics refinement and balance: {results['simulation_audit']['pass']}. Quadrature is audited against order {results['protocol']['quadrature_audit_order']} on the first {results['protocol']['quadrature_audit_cases']} cases of each population, every active step. Maximum relative decision regret: {max(results[a]['max_quadrature_relative_decision_regret'] for a in ['matched','shorter_force']):.3g}. This is one-step Bayes-risk planning, not a globally optimal six-action policy. The finite support grid and known material restrict the claim.</p>
<p>The inference code never receives true loads, boundary labels or unselected observations. Different actions are evaluated against the same hidden original trajectory. Every posterior force distribution is integrated analytically; no chain convergence or neural-network training is involved.</p>
<p><a href="results.json">Full aggregate metrics</a> · <a href="execution.json">Execution and source hashes</a> · <a href="matched-dataset.json.gz">Matched data, actions and posteriors</a> · <a href="shorter_force-dataset.json.gz">Stress data, actions and posteriors</a></p></body></html>'''
    (output/'report.html').write_text(text, encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--config', default='configs/beam_active.json')
    parser.add_argument('--output', default='runs/beam-active'); args = parser.parse_args()
    config = json.loads(Path(args.config).read_text()); output = Path(args.output); output.mkdir(parents=True, exist_ok=True)
    assert 0 < config['budget'] <= len(config['fixed_action_indices'])
    assert len(set(config['fixed_action_indices'])) == len(config['fixed_action_indices'])
    start = time.time(); results = {'protocol': config, 'actions': action_pool()}; examples = {}
    for stress in [False, True]:
        arm = 'shorter_force' if stress else 'matched'
        results[arm], examples[arm] = run_population(config, output, stress)
    results['simulation_audit'] = simulation_audit(candidates())
    results['elapsed_seconds'] = time.time()-start
    (output/'results.json').write_text(json.dumps(results, indent=2))
    plots(output, results, examples); report(output, results)
    source_paths = [Path(__file__), Path('experiments/beam_active.py'), Path('experiments/beam_boundary.py'),
                    Path(args.config), Path('tests/test_beam_active.py')]
    execution = {'utc_time': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), 'python': platform.python_version(),
        'numpy': np.__version__, 'matplotlib': matplotlib.__version__,
        'host': 'GitHub Actions free CPU' if os.getenv('GITHUB_ACTIONS') else 'local CPU',
        'base_git_head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        'github_run_id': os.getenv('GITHUB_RUN_ID'), 'elapsed_seconds': results['elapsed_seconds'],
        'source_sha256': {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths},
        'dataset_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in output.glob('*-dataset.json.gz')}}
    (output/'execution.json').write_text(json.dumps(execution, indent=2))
    print(json.dumps({'elapsed_seconds': results['elapsed_seconds'],
        'matched_calibration': results['matched']['scalar_calibration_gate_pass'],
        'paired': results['matched']['paired_final_rmse_comparison']}, indent=2), flush=True)


if __name__ == '__main__':
    main()
