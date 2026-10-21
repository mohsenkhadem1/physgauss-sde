"""Enumerated three-stiffness reference with matched Isaac depth templates.

This privileged, one-unknown diagnostic is deliberately smaller than the full
inverse problem. A normalized flow-density slice retains its learned prior;
it must not be described as an equal-prior numerical posterior or a marginal.
"""
import os
os.environ.setdefault('TORCH_DISABLE_NATIVE_JIT', '1')
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from censored_range_observation import CensoredRanges, CensoredRangeModel
from gaussian_acquisition_sensor import public_ray_queries
from validate_isaac_dataset import ray_depth
from gaussian_operator import physical_vector, category_labels
from evaluate_gaussian_inverse import load_model, encode_episode
from diagnose_gaussian_forward import load_protocol, query, sha


def normalize_log_mass(log_mass):
    values = np.asarray(log_mass, dtype=float)
    if values.ndim != 1 or np.isnan(values).any() or not np.isfinite(values).any():
        raise ValueError('At least one finite log mass required')
    weights = np.exp(values-np.max(values))
    return weights/weights.sum()


def credible_set(weights, mass=.9):
    weights = np.asarray(weights, dtype=float)
    if weights.ndim != 1 or (weights < 0).any() or not np.isclose(weights.sum(), 1):
        raise ValueError('Probability vector required')
    order = np.argsort(-weights, kind='stable')
    count = min(len(order), int(np.searchsorted(np.cumsum(weights[order]), mass))+1)
    return order[:count]


def matched_hypotheses(cases, specs, references, cameras):
    """Validate the nuisance condition; never silently compare different layouts."""
    for case, spec, ref, camera in zip(cases[1:], specs[1:], references[1:], cameras[1:]):
        first = cases[0]
        for key in ['group_id', 'boundary_type', 'force_type', 'seed', 'cells']:
            if case[key] != first[key]:
                raise ValueError('Different nuisance condition: '+key)
        for key in first['material']:
            if key != 'youngs_modulus_Pa' and case['material'][key] != first['material'][key]:
                raise ValueError('Material nuisance changed')
        if spec != specs[0]:
            raise ValueError('Support/load/camera specification changed')
        for key in ['rest_points_m', 'tetrahedra', 'surface_faces']:
            np.testing.assert_array_equal(ref[key], references[0][key])
        for key in ['K', 'camera_to_world_usd_row', 'times_s', 'frame_ids']:
            np.testing.assert_array_equal(camera[key], cameras[0][key])


def covariance(frames, rays, sigma):
    count = frames*rays
    return sigma**2*(np.eye(count)+np.kron(np.eye(frames), np.ones((rays, rays)))+np.ones((count, count)))


def posterior_record(model, template, observation, true_index):
    posterior, diagnostics = model.condition(template, observation)
    included = credible_set(posterior)
    return {'probabilities': posterior.tolist(), 'true_probability': float(posterior[true_index]),
            'map_correct': bool(np.argmax(posterior) == true_index),
            'credible90_contains_truth': bool(true_index in included),
            'credible90_size': len(included),
            'entropy_nats': float(-np.sum(posterior*np.log(posterior.clip(1e-300)))),
            **diagnostics}


def sensor_check(model, simulator, surrogate, count, seed):
    fantasies, generating = model.fantasies(simulator, count=count, seed=seed)
    result = {}
    for name, template in [('simulator', simulator), ('neural_forward', surrogate)]:
        records = [posterior_record(model, template,
                   CensoredRanges(fantasies.visible[i], fantasies.residual_m[i]),
                   int(generating['generating_particle_indices'][i])) for i in range(count)]
        result[name] = {key: float(np.mean([r[key] for r in records])) for key in
            ['true_probability', 'map_correct', 'credible90_contains_truth', 'credible90_size', 'entropy_nats']}
    return result


def run(protocol, output, device):
    cfg = load_protocol(protocol)
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    root = Path(cfg['secondary_dataset'])
    plan = json.loads((root/'plan.json').read_text())
    bundle = load_model(cfg['inverse_checkpoint'], cfg['passive_checkpoint'], device)
    inputs = {str(protocol): sha(protocol), str(root/'plan.json'): sha(root/'plan.json')}
    results, checks = [], []
    for group_index, group in enumerate(cfg['reference_groups']):
        cases = sorted([c for c in plan['episodes'] if c['group_id'] == group and c['split'] == 'test'],
                       key=lambda c: c['material']['youngs_modulus_Pa'])
        if [c['material']['youngs_modulus_Pa'] for c in cases] != cfg['reference_stiffness_Pa']:
            raise ValueError('Unexpected reference hypotheses')
        episodes = [root/'episodes'/c['episode_id'] for c in cases]
        specs, refs, cameras = [], [], []
        for episode in episodes:
            if not json.loads((episode/'validation.json').read_text())['passed']:
                raise ValueError('Unaudited depth bank')
            specs.append(json.loads((episode/'truth/boundaries.json').read_text()))
            with np.load(episode/'reference.npz') as raw:
                refs.append({k: raw[k] for k in raw.files})
            with np.load(episode/'observations/cameras.npz') as raw:
                cameras.append({k: raw[k] for k in raw.files})
            for p in [episode/'reference.npz', episode/'validation.json', episode/'truth/boundaries.json', episode/'observations/cameras.npz']:
                inputs[str(p)] = sha(p)
        matched_hypotheses(cases, specs, refs, cameras)
        rest, camera = refs[0]['rest_points_m'], cameras[0]
        physical = np.stack([physical_vector(c, s, rest) for c, s in zip(cases, specs)])
        bc, family = category_labels(cases[0])
        neural_q = query(bundle['forward'], rest, camera['times_s'], physical, bc, family, device)
        all_queries = [public_ray_queries(refs[0], pose, camera['K'], count=12, seed=cfg['seed']+j)
                       for j, pose in enumerate(camera['camera_to_world_usd_row'])]
        bank_visible, bank_residual, nn_visible, nn_residual = [], [], [], []
        for i, episode in enumerate(episodes):
            bdepth, ndepth = [], []
            for j, frame in enumerate(camera['frame_ids']):
                query_spec = all_queries[j]
                path = episode/'observations/depth'/f'{int(frame):05d}.npz'
                with np.load(path) as raw:
                    bdepth.append(raw['depth_m'][query_spec['ys'], query_spec['xs']])
                ndepth.append(ray_depth(neural_q[i, j], refs[0]['surface_faces'], query_spec['pose'],
                                       query_spec['K'], query_spec['xs'], query_spec['ys']))
                inputs[str(path)] = sha(path)
            base = np.stack([q['rest_depth_m'] for q in all_queries])
            for depth, visible_list, residual_list in [(np.asarray(bdepth), bank_visible, bank_residual),
                                                       (np.asarray(ndepth), nn_visible, nn_residual)]:
                valid = np.isfinite(depth) & (depth > 0)
                visible_list.append(valid)
                residual_list.append(np.where(valid, depth-base, 0))
        bank_visible, bank_residual = np.asarray(bank_visible), np.asarray(bank_residual)
        nn_visible, nn_residual = np.asarray(nn_visible), np.asarray(nn_residual)
        np.savez_compressed(output/(group+'-sensor-templates.npz'),
            stiffness_Pa=cfg['reference_stiffness_Pa'], physical=physical,
            times_s=camera['times_s'], frame_ids=camera['frame_ids'],
            xs=np.stack([q['xs'] for q in all_queries]), ys=np.stack([q['ys'] for q in all_queries]),
            simulator_visible=bank_visible, simulator_residual_m=bank_residual,
            neural_visible=nn_visible, neural_residual_m=nn_residual)
        for mode in ['prefix', 'full']:
            ids = np.flatnonzero((camera['times_s'] > 0) & ((camera['times_s'] <= 1) if mode == 'prefix' else True))
            simulator = CensoredRanges(bank_visible[:, ids].reshape(3, -1), bank_residual[:, ids].reshape(3, -1))
            surrogate = CensoredRanges(nn_visible[:, ids].reshape(3, -1), nn_residual[:, ids].reshape(3, -1))
            flow_slices = []
            for episode in episodes:
                encoded = encode_episode(bundle, episode, max_time_s=1 if mode == 'prefix' else None)
                with torch.no_grad():
                    context = encoded['context'].expand(3, -1)
                    log_density = bundle['inverse'].posterior.log_prob(
                        torch.as_tensor(physical, device=device), context,
                        torch.full((3,), bc, device=device, dtype=torch.long),
                        torch.full((3,), family, device=device, dtype=torch.long)).cpu().numpy()
                flow_slices.append((log_density, normalize_log_mass(log_density)))
                # Record every RGB/mask/depth input potentially used by encoder.
                for kind, suffix in [('rgb', '.png'), ('masks', '.png'), ('depth', '.npz')]:
                    for frame in encoded['inputs']['cameras']['frame_ids']:
                        p = episode/'observations'/kind/(f'{int(frame):05d}'+suffix)
                        inputs[str(p)] = sha(p)
            for noise_index, noise_mm in enumerate(cfg['noise_std_mm']):
                sensor = CensoredRangeModel(covariance(len(ids), 12, noise_mm/1000),
                    cfg['visibility_error_probability'], cfg['background_std_m'], cfg['outlier_probability'])
                for i, case in enumerate(cases):
                    observation = CensoredRanges(simulator.visible[i], simulator.residual_m[i])
                    exact = posterior_record(sensor, simulator, observation, i)
                    neural = posterior_record(sensor, surrogate, observation, i)
                    log_density, flow = flow_slices[i]
                    row = {'group_id': group, 'episode_id': case['episode_id'], 'true_index': i,
                           'true_stiffness_Pa': case['material']['youngs_modulus_Pa'], 'mode': mode,
                           'sensor_frame_ids': camera['frame_ids'][ids].tolist(), 'noise_std_mm': noise_mm,
                           'simulator_reference': exact, 'neural_forward_reference': neural,
                           'flow_density_slice': {'log_density': log_density.tolist(), 'normalized_scores': flow.tolist(),
                                                  'map_correct': bool(np.argmax(flow) == i), 'true_score': float(flow[i])},
                           'total_variation_simulator_vs_neural': float(.5*np.abs(np.asarray(exact['probabilities'])-neural['probabilities']).sum()),
                           'total_variation_simulator_vs_flow_slice': float(.5*np.abs(np.asarray(exact['probabilities'])-flow).sum())}
                    results.append(row)
                check = sensor_check(sensor, simulator, surrogate, cfg['synthetic_sensor_replicates_per_group'],
                                     cfg['seed']+group_index*100+noise_index*2+(mode == 'full'))
                checks.append({'group_id': group, 'mode': mode, 'noise_std_mm': noise_mm,
                               'replicates': cfg['synthetic_sensor_replicates_per_group'], **check})
            print(f'{group} {mode}: enumerated posterior and sensor checks done', flush=True)
    for key in ['passive', 'controlled', 'inverse']:
        inputs[cfg[key+'_checkpoint']] = sha(cfg[key+'_checkpoint'])
    inputs[str(Path(__file__))] = sha(__file__)
    result = {'protocol': cfg, 'protocol_sha256': sha(protocol), 'episodes': results,
              'synthetic_sensor_checks': checks, 'input_sha256': inputs,
              'qualification': 'Exploratory conditional three-point posterior, no parameter fitting. Flow slice is not an equal-prior marginal posterior. Synthetic coverage concerns the stipulated sensor model, not real physical calibration.'}
    (output/'results.json').write_text(json.dumps(result, indent=2)+'\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, default=Path('configs/gaussian_diagnosis_v01.json'))
    parser.add_argument('--output', type=Path, default=Path('runs/gaussian-diagnosis-reference-v01'))
    parser.add_argument('--device', default='cuda:1')
    args = parser.parse_args()
    run(args.protocol, args.output, args.device)
