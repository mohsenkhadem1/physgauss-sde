"""Full-target batched gradient pilot; no evaluation truth is used in sampling."""
import argparse
import time
from pathlib import Path
import torch
from physgauss.physics import InvalidDeformation
from physgauss.split_hmc import SplitHMCConfig, sample_split_hmc
from physgauss.chain_diagnostics import diagnose_quantities
from physgauss.study import write_json
from physgauss.white_posterior import WhitePosterior
from inspect_posterior_inference import load_case


class WhiteTarget:
    def __init__(self, model, observations, mean, chol):
        self.model, self.observations, self.mean, self.chol = model, observations, mean, chol
        self.transform = torch.linalg.solve_triangular(chol.T, torch.eye(len(mean), dtype=mean.dtype), upper=True)
        with torch.no_grad():
            self.center = model.objective(mean[None], observations)[0]

    def decode(self, white):
        return self.mean+white@self.transform.T

    def __call__(self, white):
        w = white.detach().requires_grad_()
        try:
            values = self.model.objective(self.decode(w), self.observations)
            ratios = -values+self.center+.5*w.square().sum(-1)
            grad, = torch.autograd.grad(ratios.sum(), w)
            invalid = (~torch.isfinite(ratios)) | (~torch.isfinite(grad).all(-1))
            return ratios.detach(), torch.nan_to_num(grad.detach()), invalid
        except (InvalidDeformation, ValueError):
            if len(w) == 1:
                return w.new_full((1,), -torch.inf), torch.zeros_like(w), torch.ones(1, dtype=torch.bool)
            rows = [self(row[None]) for row in w]
            return tuple(torch.cat([row[i] for row in rows]) for i in range(3))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--warmup', type=int, default=100)
    p.add_argument('--draws', type=int, default=300)
    p.add_argument('--step-size', type=float, default=.3)
    p.add_argument('--image-taylor', action='store_true')
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    model, obs, fit, inputs = load_case('runs/calibrated-noise-published/comparison', 'pulse_and_stochastic__4101__neural')
    fit = torch.load('runs/calibrated-posterior-optimized/observed_hessian.pt', weights_only=True)['fit']
    target = WhitePosterior(model, obs, fit['mean'], fit['precision_cholesky'], image_taylor=True) if args.image_taylor else WhiteTarget(model, obs, fit['mean'], fit['precision_cholesky'])
    began = time.perf_counter()
    r = sample_split_hmc(target, len(fit['mean']), SplitHMCConfig(warmup=args.warmup, draws=args.draws, step_size=args.step_size), progress=print)
    r['coordinates'] = target.decode(r['white'].flatten(0, 1)).reshape_as(r['white'])
    r['seconds'] = time.perf_counter()-began
    torch.save(r, args.output/'chains.pt')
    diagnostics = diagnose_quantities(r['coordinates'], [f'state_{i}' for i in range(model.state_dimension)])
    write_json(args.output/'diagnostics.json', diagnostics)
    print('Failed', len(diagnostics['failed_quantities']), 'seconds', r['seconds'], flush=True)
