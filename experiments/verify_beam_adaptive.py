"""Replay archived adaptive-prior comparisons; no success-by-narrowness gate."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import numpy as np
from beam_boundary import mixture_cdf
from beam_active import State, update, target_risk, predictive
from run_beam_adaptive import setup, METHODS


def verify(output):
    results = json.loads((output/'results.json').read_text()); config = results['protocol']
    manifest = json.loads((output/'execution.json').read_text())
    assert all(hashlib.sha256(Path(p).read_bytes().replace(b'\r\n', b'\n')).hexdigest()==h for p, h in manifest['source_sha256_lf'].items())
    assert all(hashlib.sha256((output/p).read_bytes()).hexdigest()==h for p, h in manifest['dataset_sha256'].items())
    setups = {'fixed_2s': setup(config, [2.]), 'adaptive': setup(config, config['lengths_s'])}
    replays = 0; max_error = 0.; max_cdf_error = 0.; legacy_error = 0.; same_sets = {}
    for arm in results['populations']:
        data = json.loads(gzip.decompress((output/f'{arm}-dataset.json.gz').read_bytes()))
        initial = data['initial_observations_pixels']; n = len(initial)
        spec = results['populations'][arm]['specification']
        if 'source' in spec:
            payload = Path(spec['source']).read_bytes()
            assert hashlib.sha256(payload).hexdigest()==data['replay_source_sha256']
            old = json.loads(gzip.decompress(payload))
            for key in ['labels', 'force_N', 'initial_observations_pixels', 'potential_action_observations_pixels', 'random_action_orders']:
                assert data[key]==old[key][:n]
            for policy in ['active', 'random', 'fixed']:
                assert data['action_logs']['fixed_2s/'+policy][0][0]['action']==old['action_logs'][policy][0][0]['action']
                for step in range(config['budget']+1):
                    for i in range(n):
                        legacy_error = max(legacy_error, abs(data['per_case_metrics']['fixed_2s/'+policy][step][i]['hidden_rmse_mm']
                            -old['per_case_metrics_and_tip_posteriors'][policy][step][i]['hidden_rmse_mm']))
            del old
        fits = {p: s[0].fit(initial) for p, s in setups.items()}
        oracle = {l: setup(config, [l]) for l in sorted(set(data['true_lengths_s']))}
        oracle_fits = {l: s[0].fit(initial) for l, s in oracle.items()}
        x = np.array(data['reference_x_m']); hidden = (x<.3)|(x>.8); same = 0
        for i in range(n):
            truth = np.array(data['force_N'][i])[:, None]*np.array(data['reference_response_m_per_N'][data['labels'][i]])[None, :]
            final_states = {}; sets = {}
            for name in METHODS:
                prior, policy = name.split('/')
                if prior == 'oracle_length':
                    l = data['true_lengths_s'][i]; engine, spatial, _, models = oracle[l]; w, m = oracle_fits[l]
                else:
                    engine, spatial, _, models = setups[prior]; w, m = fits[prior]
                state = State(w[i].copy(), m[i].copy(), engine.cov.copy())
                history = data['action_logs'][name][i]; selected = [h['action'] for h in history]
                assert len(set(selected))==len(selected)==config['budget']
                if policy=='fixed': assert selected==config['fixed_action_indices'][:config['budget']]
                if policy=='random': assert selected==data['random_action_orders'][i][:config['budget']]
                for step in range(config['budget']+1):
                    r = data['per_case_metrics'][name][step][i]
                    pred = np.einsum('c,ct,cx->tx', state.weights, state.mean, spatial)
                    rmse = float(np.sqrt(np.mean((pred[:, hidden]-truth[:, hidden])**2))*1000)
                    mean = state.mean[:, 4]*spatial[:, -1]; var = state.cov[:, 4, 4]*spatial[:, -1]**2
                    pit = float(mixture_cdf(truth[4, -1], state.weights, mean, var))
                    max_error = max(max_error, abs(rmse-r['hidden_rmse_mm']), abs(pit-r['tip_pit']),
                        abs(float(state.weights@mean)-r['tip_mean_m']),
                        abs(target_risk(state, spatial[:, hidden])-r['hidden_risk_m2']),
                        float(np.max(abs(engine.length_probabilities(state.weights)-np.array(r['length_probabilities'])))))
                    for value, q in [(r['tip_lo90_m'], .05), (r['tip_hi90_m'], .95)]:
                        max_cdf_error = max(max_cdf_error, abs(float(mixture_cdf(value, state.weights, mean, var))-q))
                    assert np.isfinite(state.mean).all() and np.isfinite(state.cov).all()
                    assert abs(state.weights.sum()-1)<1e-12 and np.min(state.weights)>=0
                    assert np.linalg.eigvalsh(state.cov).min()>-1e-13
                    replays += 1
                    if step==config['budget']: break
                    event = history[step]; a = event['action']; y = event['observed_pixels']
                    assert y==data['potential_action_observations_pixels'][i][a]
                    if policy=='active':
                        scores = event['expected_posterior_risks_m2']
                        assert set(map(int, scores))==set(range(13))-set(selected[:step])
                        assert int(min(scores, key=scores.get))==a
                    mu, v, _ = predictive(state, models[0][a], models[1][a], models[2][a])
                    max_error = max(max_error, abs(float(mixture_cdf(y, state.weights, mu, v))-event['predictive_pit']))
                    state = update(state, models[0][a], models[1][a], models[2][a], y)
                final_states[name] = state; sets[name] = set(selected)
            if sets['adaptive/active']==sets['adaptive/fixed']:
                np.testing.assert_allclose(final_states['adaptive/active'].weights, final_states['adaptive/fixed'].weights, atol=1e-9)
                np.testing.assert_allclose(final_states['adaptive/active'].mean, final_states['adaptive/fixed'].mean, atol=1e-9)
                same += 1
        same_sets[arm] = {'cases': n, 'identical_adaptive_active_fixed_sets': same}
        print(f'Replayed {arm}: {n} cases', flush=True)
    assert max_error<1e-8, max_error
    assert max_cdf_error<1e-8, max_cdf_error
    assert legacy_error<1e-7, legacy_error
    assert results['simulation_audit']['pass']
    assert max(r['max_quadrature_decision_regret'] for r in results['populations'].values())<.001
    validation = {'pass': True, 'states_replayed': replays, 'max_absolute_replay_error': max_error,
        'max_previous_study_RMSE_reproduction_error_mm': legacy_error,
        'max_interval_cdf_error': max_cdf_error, 'source_LF_and_dataset_byte_hashes_match': True,
        'all_covariances_PSD': True, 'action_budget_pairing_and_argmin_checks': True,
        'same_final_action_set_checks': same_sets,
        'hierarchical_adaptive_scalar_checks_all_pass': all(v['pass'] for p in ['active', 'random', 'fixed']
            for v in results['populations']['hierarchical_fresh']['scalar_calibration_checks'][f'adaptive/{p}']),
        'scope': 'Numerical validity is separate from conditional coverage, adaptive superiority and model-mismatch robustness.'}
    (output/'verification.json').write_text(json.dumps(validation, indent=2)); print(json.dumps(validation, indent=2))


if __name__=='__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--output', default='examples/beam-adaptive')
    verify(Path(parser.parse_args().output))
