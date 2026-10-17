"""Read-only timing and decomposition of the previous failed posterior audit."""
import json
import time
from pathlib import Path
import torch
from physgauss.benchmark import Observations
from physgauss.neural_marginal_study import calibrate, construct


def load_case(study, case):
    study = Path(study)
    manifest = json.loads((study/'manifest.json').read_text())
    plan = manifest['plan']
    torch.set_num_threads(plan['threads'])
    try:
        import torch.backends.opt_einsum as backend
        backend.enabled = True
        backend.strategy = 'greedy'
    except ImportError:
        pass
    network, calibration = calibrate(plan, study/'calibration.json')
    record = json.loads((study/'cases'/case/'summary.json').read_text())
    model = construct(plan, network, calibration, record['mode'])
    inputs = json.loads((study/'datasets'/f"{record['condition']}__{record['seed']}"/'inputs.json').read_text())
    obs = Observations(model.view_frames, model.times.new_tensor(inputs['observed_rgb']), .015)
    checkpoint = study/'cases'/case/'fit.pt'
    if checkpoint.exists():
        fit = torch.load(checkpoint, weights_only=True)
    else:
        state = json.loads((study/'cases'/case/'solver_state.json').read_text())
        mean = model.times.new_tensor(state['coordinates'])
        precision, _ = model.fisher_precision(mean, obs)
        with torch.no_grad():
            conditional = model.elimination_at(mean[None]).conditional_mean(model.residual(mean[None]))[0]
            value = float(model.objective(mean[None], obs)[0])
        if abs(value-record['fit']['marginal_objective']) > 1e-7:
            raise ValueError('Published normalized target did not reproduce.')
        fit = {'mean': mean, 'precision_cholesky': torch.linalg.cholesky(precision),
            'conditional_force_mean': conditional, 'record': record['fit']}
    return model, obs, fit, inputs


if __name__ == '__main__':
    model, obs, fit, inputs = load_case('runs/calibrated-noise-published/comparison', 'pulse_and_stochastic__4101__neural')
    chains = torch.load('runs/calibrated-posterior-optimized/chains.pt', weights_only=True)
    summary = json.loads(Path('runs/calibrated-posterior-optimized/summary.json').read_text())
    print('Counts', chains['transition_counts']['retained'])
    print('Chain parameter means', chains['coordinates'][..., :8].mean(1).tolist())
    print('Chain correction ranges', [(float(v.min()), float(v.max()), float(v.mean())) for v in chains['log_ratio']])
    for name in [f'state_{i}' for i in range(8)]+['log_target']:
        print(name, summary['diagnostics']['quantities'][name])
    for count in [1, 4]:
        z = fit['mean'][None].repeat(count, 1).requires_grad_()
        for _ in range(2):
            began = time.perf_counter()
            value = model.objective(z, obs)
            grad, = torch.autograd.grad(value.sum(), z)
            print('Batch', count, 'value_grad_seconds', time.perf_counter()-began)
