"""Frozen matched-budget conditional inference comparison with clean observations.

Source-frozen development experiment, not an external-data benchmark. Each model
gets the same clean parameter bank, minibatch indices and single acquisition-noise
draw. Conditional flow and mixture both average 32 forward physical trajectories.
"""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(key, '2')
import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch

from scripts.gaussian_boundary_flow.model import observe, PARAMETERS
from scripts.gaussian_boundary_flow.refined import RefinedBlock
from scripts.gaussian_boundary_flow.flow import ConditionalFlow
from scripts.gaussian_boundary_flow_v02.noise import noisy_context
from scripts.gaussian_boundary_flow_v02.probabilistic import (
    GaussianMixtureRegressor, ParameterRegressor,
)

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = ROOT / 'runs/gaussian-boundary-flow-v02-matched'
REPORT = ROOT / 'reports/gaussian-boundary-flow-v02/matched-results.json'
METHODS = ('flow', 'mixture', 'point')
PROGRAMS = ('sine', 'release', 'triangle', 'reverse')
SIGMA = .00005


def json_write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def tensor_write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    torch.save(value, temporary)
    temporary.replace(path)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def make_method(method):
    if method == 'flow': return ConditionalFlow(144)
    if method == 'mixture': return GaussianMixtureRegressor(144)
    if method == 'point': return ParameterRegressor(144)
    raise ValueError(method)


def sources():
    paths = [Path(__file__).resolve(),
             ROOT / 'scripts/gaussian_boundary_flow_v02/noise.py',
             ROOT / 'scripts/gaussian_boundary_flow_v02/probabilistic.py',
             ROOT / 'scripts/gaussian_boundary_flow/model.py',
             ROOT / 'scripts/gaussian_boundary_flow/refined.py',
             ROOT / 'scripts/gaussian_boundary_flow/qualification.py',
             ROOT / 'scripts/gaussian_boundary_flow/flow.py',
             ROOT / 'src/physgauss/gaussian_boundary.py',
             ROOT / 'src/physgauss/physics.py']
    return {str(p.relative_to(ROOT)): digest(p) for p in paths}


def plan_for(device):
    return {
        'scope': 'Conditional synthetic small-strain 64-node block; no external-data or general boundary-topology claim',
        'train_seed': 43091, 'train_count': 8192,
        'test_seed': 10035, 'test_count': 128,
        'test_observation_seed': 10036, 'sampling_seed': 10037,
        'batch_noise_stream_seed': 6135, 'initialization_seeds': [7301, 7302, 7303],
        'steps': 15000, 'batch_size': 256, 'optimizer': 'AdamW',
        'learning_rate': .0003, 'weight_decay': .00001, 'gradient_clip': 5.,
        'sigma_m': SIGMA, 'bank_noise': 0., 'minibatch_noise_draws': 1,
        'observation': 'Nine frames through .4 s of eight known corner correspondences, orthographic 2D displacement',
        'known': ['geometry', 'density', 'zero initial displacement/velocity', 'gravity',
                  'camera', 'load direction', 'Gaussian widths', 'top/bottom face topology', 'future command'],
        'inferred_parameters': PARAMETERS,
        'future_programs': list(PROGRAMS), 'future_starts_index': 17,
        'methods': {
            name: {'parameters': sum(p.numel() for p in make_method(name).parameters()),
                   'samples': 1 if name == 'point' else 32,
                   'loss': {'flow': 'conditional rectified-flow MSE',
                            'mixture': 'four-component diagonal Gaussian mixture logit NLL',
                            'point': 'parameter-logit MSE'}[name]}
            for name in METHODS},
        'precision': 'float32', 'physics_device': device,
        'source_sha256': sources(),
        'selection': 'Architectures and train steps fixed before test; no tuning on this test cohort',
    }


def assert_sources(plan):
    for path, expected in plan['source_sha256'].items():
        if digest(ROOT / path) != expected:
            raise RuntimeError(f'Source changed after plan freeze: {path}')


def build_bank(out, plan, block):
    generator = torch.Generator().manual_seed(plan['train_seed'])
    z = torch.logit(torch.rand(plan['train_count'], 6, generator=generator).clamp(.001, .999))
    shard_dir = out / 'shards'; shard_dir.mkdir(exist_ok=True)
    clean = []
    with torch.no_grad():
        for start in range(0, len(z), 128):
            path = shard_dir / f'{start:05d}.pt'
            if path.exists():
                saved = torch.load(path, weights_only=True)
                if not torch.equal(saved['z'], z[start:start + 128]):
                    raise RuntimeError('Training shard belongs to a different parameter bank')
                features = saved['clean']
            else:
                displacement = block.rollout(z[start:start + 128].to(block.device))
                features = observe(displacement, noise=False).cpu()
                tensor_write(path, {'z': z[start:start + 128], 'clean': features})
            clean.append(features)
            if start % 1024 == 0:
                json_write(out / 'progress.json', {'phase': 'data', 'completed': start + len(features), 'total': len(z)})
                print(f'data {start + len(features)}/{len(z)}', flush=True)
    x = torch.cat(clean)
    if not bool(torch.isfinite(x).all()): raise RuntimeError('Nonfinite training observation')
    bank = {'z': z, 'clean': x, 'mean': x.mean(0), 'scale': x.std(0).clamp_min(.0001)}
    tensor_write(out / 'training.pt', bank)
    return bank


def train_method(out, bank, plan, name, initialization_seed):
    torch.manual_seed(initialization_seed)
    model = make_method(name)
    optimizer = torch.optim.AdamW(model.parameters(), lr=plan['learning_rate'],
                                 weight_decay=plan['weight_decay'])
    stream = torch.Generator().manual_seed(plan['batch_noise_stream_seed'])
    checkpoint_path = out / f'{name}-checkpoint.pt'
    start = 0
    if checkpoint_path.exists():
        saved = torch.load(checkpoint_path, weights_only=False)
        if saved['plan_sha256'] != digest(out / 'plan.json'):
            raise RuntimeError('Checkpoint/plan mismatch')
        model.load_state_dict(saved['model']); optimizer.load_state_dict(saved['optimizer'])
        stream.set_state(saved['batch_rng']); torch.set_rng_state(saved['rng'])
        start = saved['step']
    losses = []
    model.train()
    for step in range(start, plan['steps']):
        ids = torch.randint(len(bank['z']), (plan['batch_size'],), generator=stream)
        context = noisy_context(bank['clean'][ids], bank['mean'], bank['scale'], SIGMA, stream)
        loss = model.loss(bank['z'][ids], context)
        if not bool(torch.isfinite(loss)):
            raise RuntimeError(f'Nonfinite {name} training objective at step {step}')
        optimizer.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), plan['gradient_clip'])
        optimizer.step(); losses.append(float(loss.detach()))
        if (step + 1) % 500 == 0 or step + 1 == plan['steps']:
            tensor_write(checkpoint_path, {'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                         'step': step + 1, 'rng': torch.get_rng_state(), 'batch_rng': stream.get_state(),
                         'plan_sha256': digest(out / 'plan.json')})
            progress = {'phase': 'training', 'method': name, 'step': step + 1,
                        'loss_last_up_to_500': float(np.mean(losses[-500:]))}
            json_write(out / 'progress.json', progress)
            print(progress, flush=True)
    return model.eval()


def evaluate(out, bank, plan, block):
    assert_sources(plan)
    # Seal every final checkpoint before constructing/reading a test observation.
    frozen = {}
    for name in METHODS:
        checkpoint = torch.load(out / f'{name}-checkpoint.pt', weights_only=False)
        if checkpoint['step'] != plan['steps']:
            raise RuntimeError('All models must finish training before test evaluation')
        frozen[name] = digest(out / f'{name}-checkpoint.pt')
    freeze_path = out / 'evaluation-freeze.json'
    freeze = {'checkpoint_sha256': frozen, 'plan_sha256': digest(out / 'plan.json')}
    if freeze_path.exists() and json.loads(freeze_path.read_text()) != freeze:
        raise RuntimeError('Cannot replace an already evaluated checkpoint in this run')
    json_write(freeze_path, freeze)
    device = block.device
    generator = torch.Generator().manual_seed(plan['test_seed'])
    truth = torch.logit(torch.rand(plan['test_count'], 6, generator=generator).clamp(.001, .999)).to(device)
    with torch.no_grad():
        base = block.rollout(truth)
        clean = observe(base, noise=False).cpu()
        context = noisy_context(clean, bank['mean'], bank['scale'], SIGMA,
                                torch.Generator().manual_seed(plan['test_observation_seed']))
        tensor_write(out / 'test-observations.pt', {'truth': truth.cpu(), 'clean': clean, 'context': context})
        physical_truth = block.parameters(truth).cpu()
        all_samples = {}
        parameter_scores = {}
        for name in METHODS:
            model = make_method(name)
            checkpoint = torch.load(out / f'{name}-checkpoint.pt', weights_only=False)
            model.load_state_dict(checkpoint['model']); model.eval()
            if name == 'point': samples = model(context)[:, None]
            else: samples = model.sample(context, count=32,
                    generator=torch.Generator().manual_seed(plan['sampling_seed']))
            all_samples[name] = samples
            transformed = block.parameters(samples.to(device)).cpu()
            lo, hi = torch.quantile(transformed, torch.tensor([.05, .95]), dim=1)
            parameter_scores[name] = {
                'absolute_mean_error': (transformed.mean(1)-physical_truth).abs().mean(0).tolist(),
                'coverage_90': None if name == 'point' else ((physical_truth >= lo)&(physical_truth <= hi)).float().mean(0).tolist(),
                'interval_width': None if name == 'point' else (hi-lo).mean(0).tolist(),
                'units': 'Log SI parameters except support_x_m and load_x_m; see inferred_parameters',
            }
        tensor_write(out / 'parameter-samples.pt', all_samples)
        program_results = {}
        for kind in PROGRAMS:
            target = block.rollout(truth, kind=kind)
            rows = {}
            means = {}
            for name in METHODS:
                samples = all_samples[name]; flat = samples.flatten(0, 1).to(device)
                parts = []
                for start in range(0, len(flat), 128):
                    parts.append(block.rollout(flat[start:start + 128], kind=kind).cpu())
                pred = torch.cat(parts).reshape(len(truth), samples.shape[1], *target.shape[1:])
                true = target.cpu(); mean = pred.mean(1); means[name] = mean
                errors = (mean[:, 17:]-true[:, 17:]).square().mean((1, 2, 3)).sqrt()*1000
                if name != 'point':
                    lo, hi = torch.quantile(pred[:, :, 17:], torch.tensor([.05, .95]), dim=1)
                    inside = (true[:, 17:] >= lo)&(true[:, 17:] <= hi)
                    coverage = float(inside.float().mean()); full = float(inside.flatten(1).all(1).float().mean())
                    width = float((hi-lo).mean()*1000)
                    # Central-90% interval score: width + 2/alpha exceedance penalties.
                    score = (hi-lo)+20*(lo-true[:, 17:]).clamp_min(0)+20*(true[:, 17:]-hi).clamp_min(0)
                    interval_score = float(score.mean()*1000)
                else:
                    coverage = full = width = interval_score = None
                rows[name] = {'future_rmse_mm_mean': float(errors.mean()), 'per_case_mm': errors.tolist(),
                              'coordinate_coverage_90': coverage, 'trajectory_coverage_90': full,
                              'interval_width_mm': width, 'interval_score_90_mm': interval_score}
            program_results[kind] = rows
            tensor_write(out / f'predictions-{kind}.pt', {'truth': target.cpu(), 'means': means})
            json_write(out / f'scores-{kind}.json', rows)
            print(kind, {name: row['future_rmse_mm_mean'] for name, row in rows.items()}, flush=True)
        cpu = RefinedBlock(device='cpu')
        comparison = cpu.rollout(truth[:4].cpu())
        cpu_gpu = float((comparison-base[:4].cpu()).abs().max())
    # Paired resampling is across independent worlds, not coordinates or frames.
    rng = np.random.default_rng(22137)
    comparisons = {}
    for kind, rows in program_results.items():
        comparisons[kind] = {}
        flow_error = np.array(rows['flow']['per_case_mm'])
        for other in ('mixture', 'point'):
            difference = np.array(rows[other]['per_case_mm'])-flow_error
            bootstrap = difference[rng.integers(0, len(difference), size=(5000, len(difference)))].mean(1)
            comparisons[kind][other] = {
                'other_minus_flow_mm_mean': float(difference.mean()),
                'paired_bootstrap_95CI_mm': np.quantile(bootstrap, [.025, .975]).tolist(),
                'flow_wins': int((difference > 0).sum()), 'cases': len(difference)}
    assert_sources(plan)
    result = {'scope': plan['scope'], 'plan_sha256': digest(out / 'plan.json'),
              'checkpoint_sha256': frozen, 'methods': plan['methods'],
              'single_noise_draw_sigma_m': SIGMA, 'training_observations_clean': True,
              'same_test_context_all_methods': True, 'parameter_scores': parameter_scores,
              'programs': program_results, 'paired_comparisons': comparisons,
              'cpu_gpu_max_displacement_difference_m_4cases': cpu_gpu,
              'source_hashes_unchanged': True,
              'limitations': ['Same block/topology/prior family; OOD is future command waveform only',
                              'Known geometry, mass, initial state, directions and Gaussian widths',
                              'Point baseline uses latent MSE, not a trajectory-optimized objective',
                              'No independent reference posterior or claim of global physical calibration']}
    json_write(out / 'results.json', result)
    json_write(REPORT, result)
    json_write(out / 'progress.json', {'phase': 'complete'})
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', choices=('cpu', 'cuda:0'), default='cuda:0')
    parser.add_argument('--phase', choices=('all', 'data', 'train', 'evaluate'), default='all')
    args = parser.parse_args()
    torch.set_num_threads(2); torch.backends.cuda.matmul.allow_tf32 = False
    out = DEFAULT_OUT; out.mkdir(parents=True, exist_ok=True)
    plan = plan_for(args.device); plan_path = out / 'plan.json'
    if plan_path.exists():
        if json.loads(plan_path.read_text()) != plan:
            raise RuntimeError('Immutable plan changed; preserve this run and use a new campaign')
    else:
        json_write(plan_path, plan)
        for relative in plan['source_sha256']:
            destination = out / 'source-snapshot' / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes((ROOT / relative).read_bytes())
    start = time.time()
    block = RefinedBlock(device=args.device)
    if (out / 'training.pt').exists(): bank = torch.load(out / 'training.pt', weights_only=True)
    else: bank = build_bank(out, plan, block)
    if args.phase == 'data': return
    if args.phase in ('all', 'train'):
        for name, seed in zip(METHODS, plan['initialization_seeds']):
            train_method(out, bank, plan, name, seed)
    if args.phase in ('all', 'evaluate'):
        evaluate(out, bank, plan, block)
    print('elapsed_this_invocation_seconds', time.time()-start, flush=True)


if __name__ == '__main__':
    main()
