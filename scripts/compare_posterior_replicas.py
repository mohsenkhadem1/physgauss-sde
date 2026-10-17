"""Independent full-state check of replica differences and cached exact targets."""
import argparse
import json
from pathlib import Path
import torch
from physgauss.accurate_study import digest_tensor
from physgauss.benchmark import Observations
from physgauss.neural_marginal_study import calibrate, construct
from physgauss.study import write_json
from physgauss.cli import source_fingerprint


def run(primary, replica, output):
    torch.set_num_threads(2)
    import torch.backends.opt_einsum as backend
    backend.enabled = True; backend.strategy = 'greedy'
    base = Path('examples/calibrated-noise/comparison')
    plan = json.loads((base/'manifest.json').read_text())['plan']
    net, calibration = calibrate(plan, base/'calibration.json')
    model = construct(plan, net, calibration, 'neural')
    data = json.loads((base/'datasets'/'pulse_and_stochastic__4101'/'inputs.json').read_text())
    obs = Observations(model.view_frames, model.times.new_tensor(data['observed_rgb']), .015)
    gap = (model.times > .3+1e-8)&(model.times < .7-1e-8)
    records = []
    for root in [Path(primary), Path(replica)]:
        chain_file, = root.rglob('chains.pt')
        reference_file, = root.rglob('reference.pt')
        chain = torch.load(chain_file, weights_only=True)
        fit = torch.load(reference_file, weights_only=True)
        transform = torch.linalg.solve_triangular(fit['precision_cholesky'].T,
            torch.eye(model.state_dimension, dtype=model.times.dtype), upper=True)
        c, n, d = chain['white'].shape
        with torch.no_grad():
            z = fit['mean']+chain['white'].flatten(0, 1)@transform.T
            pop = model.trajectory(z).x[:, gap][:, :, model.free].flatten(1).reshape(c, n, 84)
            block = int(n**.5); blocks = n//block
            means = pop[:, :blocks*block].reshape(c, blocks, block, 84).mean(2).flatten(0, 1)
            covariance = torch.cov(means.T)/len(means)
            center = model.objective(fit['mean'][None], obs)[0]
            indices = torch.tensor([0, n//4, n//2, n-1])
            probe_w = chain['white'][:, indices].flatten(0, 1)
            probe_z = fit['mean']+probe_w@transform.T
            values = torch.cat([model.objective(b, obs) for b in probe_z.split(4)])
            expected = -values+center+.5*probe_w.square().sum(1)
            error = float((expected-chain['log_ratio'][:, indices].flatten()).abs().max())
            if error > 1e-7:
                raise ValueError('Cached target is not the full normalized nonlinear target.')
        records.append({'mean': pop.mean((0, 1)), 'covariance': covariance, 'cached_target_max_abs_error': error,
            'chains': c, 'retained_per_chain': n, 'batch_size': block, 'batch_count': len(means)})
    delta = records[0]['mean']-records[1]['mean']
    cov = records[0]['covariance']+records[1]['covariance']
    z_score = delta/cov.diagonal().sqrt()
    threshold = float(torch.special.ndtri(model.times.new_tensor(1-.05/(2*84))))
    write_json(output, {'case': 'pulse_and_stochastic__4101__neural', 'source_fingerprint_sha256': source_fingerprint(),
        'observations_sha256': digest_tensor(obs.rgb),
        'max_gap_mean_difference_m': float(delta.abs().max()), 'differences_m': delta.tolist(),
        'combined_batch_mcse_m': cov.diagonal().sqrt().tolist(), 'mean_difference_z_scores': z_score.tolist(),
        'max_absolute_z_score': float(z_score.abs().max()),
        'approximate_84_coordinate_bonferroni_95_threshold': threshold,
        'coordinates_exceeding_threshold': int((z_score.abs() > threshold).sum()),
        'replicas': [{k: v for k, v in r.items() if k not in ['mean', 'covariance']} for r in records],
        'qualification': 'batch-means CLT/normal MCSE approximation with correlated coordinates; diagnostic agreement, not a proof of model or posterior calibration'})


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--primary', type=Path, required=True); p.add_argument('--replica', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args(); run(a.primary, a.replica, a.output)
