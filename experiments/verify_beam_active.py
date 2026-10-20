"""Replay every saved action and posterior; keep validity separate from benefit."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import numpy as np
from beam_boundary import candidates, observation_design, ExactBoundaryPosterior, response
from beam_active import State, action_pool, action_models, update


def verify(output):
    results = json.loads((output/'results.json').read_text())
    config = results['protocol']; execution = json.loads((output/'execution.json').read_text())
    # A local archive may record an absolute Windows entry-point path. Resolve
    # the repo-relative experiments path when replaying on a different host.
    def source_path(value):
        normalized = value.replace('\\', '/')
        if '/experiments/' in normalized:
            return Path('experiments')/normalized.rsplit('/experiments/', 1)[1]
        return Path(normalized)
    def source_matches(value, digest):
        data = source_path(value).read_bytes()
        lf = data.replace(b'\r\n', b'\n')
        # Git's eol=lf conversion may differ from the Windows executed bytes.
        return digest in {hashlib.sha256(v).hexdigest() for v in [data, lf, lf.replace(b'\n', b'\r\n')]}
    hashes = all(source_matches(p, digest) for p, digest in execution['source_sha256'].items())
    assert hashes, 'Source bytes differ from the executed source manifest'
    assert all(hashlib.sha256((output/p).read_bytes()).hexdigest() == digest
               for p, digest in execution['dataset_sha256'].items())
    bounds = candidates(); actions = action_pool()
    engine = ExactBoundaryPosterior(bounds, observation_design('sparse'), config['pixel_noise_std'], config['inference_elements'])
    models = action_models(bounds, actions, config['inference_elements'], config['pixel_noise_std'])
    x = np.linspace(0, 1, 81); hidden = (x<.3) | (x>.8)
    spatial = np.stack([response(b, x, config['inference_elements']) for b in bounds])
    max_error = 0.; count = 0; same_set = {}
    for arm in ['matched', 'shorter_force']:
        data = json.loads(gzip.decompress((output/f'{arm}-dataset.json.gz').read_bytes()))
        initial = np.array(data['initial_observations_pixels']); w, m = engine.fit(initial)
        potential = np.array(data['potential_action_observations_pixels'])
        same = 0
        for i in range(len(initial)):
            sets = {}
            for p in ['active', 'random', 'fixed']:
                state = State(w[i], m[i], engine.cov.copy())
                history = data['action_logs'][p][i]
                selected = [v['action'] for v in history]
                assert len(selected) == len(set(selected)) == config['budget']
                assert all(0 <= a < len(actions) for a in selected)
                if p == 'fixed':
                    assert selected == config['fixed_action_indices'][:config['budget']]
                if p == 'random':
                    assert selected == data['random_action_orders'][i][:config['budget']]
                sets[p] = set(selected)
                for step in range(config['budget']+1):
                    r = data['per_case_metrics_and_tip_posteriors'][p][step][i]
                    max_error = max(max_error,
                        float(np.max(abs(state.weights-np.array(r['weights'])))),
                        float(np.max(abs(state.mean[:, 4]*spatial[:, -1]-np.array(r['tip_means'])))),
                        float(np.max(abs(state.cov[:, 4, 4]*spatial[:, -1]**2-np.array(r['tip_variances'])))))
                    assert np.isfinite(state.mean).all() and np.isfinite(state.cov).all()
                    assert abs(state.weights.sum()-1)<1e-12 and np.min(state.weights)>=0
                    assert np.min(np.linalg.eigvalsh(state.cov)) > -1e-14
                    truth = np.array(data['force_N'][i])[:, None]*np.array(data['reference_response_m_per_N'][data['labels'][i]])[None, :]
                    pred = np.einsum('c,ct,cx->tx', state.weights, state.mean, spatial)
                    error = float(np.sqrt(np.mean((pred[:, hidden]-truth[:, hidden])**2))*1000)
                    max_error = max(max_error, abs(error-r['hidden_rmse_mm']))
                    count += 1
                    if step == config['budget']:
                        break
                    event = history[step]; a = event['action']
                    assert event['observed_pixels'] == potential[i, a]
                    if p == 'active':
                        scores = event['expected_posterior_risks_m2']
                        assert set(map(int, scores)) == set(range(len(actions)))-set(selected[:step])
                        assert int(min(scores, key=scores.get)) == a
                    state = update(state, models[0][a], models[1][a], models[2][a], event['observed_pixels'])
                if p == 'active': active_state = state
                if p == 'fixed' and sets['active'] == sets['fixed']:
                    np.testing.assert_allclose(state.weights, active_state.weights, atol=1e-10)
                    np.testing.assert_allclose(state.mean, active_state.mean, atol=1e-10)
                    same += 1
        same_set[arm] = {'active_fixed_identical_final_action_set_cases': same, 'cases': len(initial)}
    assert max_error < 1e-9, max_error
    assert results['simulation_audit']['pass']
    assert max(results[a]['max_quadrature_relative_decision_regret'] for a in ['matched', 'shorter_force']) < .005
    verification = {'pass': True, 'source_and_dataset_hashes_match': hashes,
        'source_hash_policy': 'Accept exact bytes or Git LF/CRLF conversion only; dataset hashes are byte-exact.',
        'posterior_states_replayed': count, 'max_absolute_replay_difference': max_error,
        'action_budget_no_repeats_and_observation_pairing_pass': True,
        'fixed_and_random_schedules_replayed': True, 'active_argmin_and_available_pool_pass': True,
        'conditional_covariances_positive_semidefinite': True,
        'matched_scalar_calibration_gate_pass': results['matched']['scalar_calibration_gate_pass'],
        'stress_scalar_calibration_gate_pass': results['shorter_force']['scalar_calibration_gate_pass'],
        'same_final_action_set_audit': same_set,
        'scope': 'Replay/inference validity does not assert superiority, model-mismatch calibration or real-world performance.'}
    (output/'verification.json').write_text(json.dumps(verification, indent=2))
    print(json.dumps(verification, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--output', default='examples/beam-active')
    verify(Path(parser.parse_args().output))
