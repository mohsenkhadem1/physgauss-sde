"""Separate posterior-volume effects from an inadequately optimized local mode."""
import argparse
import json
import math
from pathlib import Path
import torch
from physgauss.neural_marginal_inference import fit_neural_marginal
from physgauss.marginal_inference import TrajectoryFitConfig
from physgauss.study import write_json
from physgauss.cli import source_fingerprint
from inspect_posterior_inference import load_case


def density_terms(model, obs, z):
    e = model.elimination_at(z[None]); residual = model.residual(z[None])
    a = e.conditional_mean(residual)
    white = e.whiten_q(residual)-torch.einsum('bsdk,bk->bsd', e.white_mapping, a)
    return {'profile_quadratic': .5*(white.square().sum()+a.square().sum()),
        'half_logdet_Q': .5*e.logdet_q[0], 'half_logdet_A': .5*(e.logdet[0]-e.logdet_q[0]),
        'image_nll': .5*model.image_residual(z[None], obs).square().sum(),
        'parameter_prior': .5*z[:8].square().sum(), 'negative_dynamic_log_jacobian': -model.log_dynamic_jacobian(z[None])[0],
        'fixed_constants': z.new_tensor(.5*(model.state_dimension-8)*math.log(2*math.pi)-(model.state_dimension-8)*math.log(model.marginal_config.velocity_coordinate_scale))}


def run(study, posterior, output):
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    case = 'pulse_and_stochastic__4101__neural'
    model, obs, original, inputs = load_case(study, case)
    post = json.loads(Path(posterior).read_text())
    mean = model.times.new_tensor(post['normalized_state_mean'])
    checkpoint = output/'refit.pt'
    if checkpoint.exists():
        fitted = torch.load(checkpoint, weights_only=True)
    else:
        fitted = fit_neural_marginal(model, obs, mean,
            TrajectoryFitConfig(iterations=40, decrement_tolerance=.001))
        torch.save(fitted, checkpoint)
    truth = model.times.new_tensor(inputs['positions_m_evaluation_only'])
    gap = (model.times > .3+1e-8)&(model.times < .7-1e-8)
    records = {}
    for name, point in [('original_MAP', original['mean']), ('posterior_mean', mean), ('refit_from_posterior_mean', fitted['mean'])]:
        z = point.detach().requires_grad_()
        terms = density_terms(model, obs, z)
        total = sum(terms.values())
        torch.testing.assert_close(total, model.objective(z[None], obs)[0], atol=1e-8, rtol=1e-11)
        projected = {}
        for k, value in terms.items():
            if value.requires_grad:
                g, = torch.autograd.grad(value, z, retain_graph=True)
                projected[k] = float(g@(mean-original['mean']))
            else:
                projected[k] = 0.
        with torch.no_grad():
            path = model.trajectory(z[None]).x[0]
            error = path[:, model.free]-truth[:, model.free]
            e = model.elimination_at(z[None]); r = model.residual(z[None]); a = e.conditional_mean(r)
            joint = model.objective(z[None], obs, a)[0]
            identity = joint+.5*torch.logdet(e.precision[0])-.5*model.force_dimension*math.log(2*math.pi)
        records[name] = {'objective': float(total.detach()), 'terms': {k: float(v.detach()) for k, v in terms.items()},
            'normalized_state': point.tolist(),
            'term_directional_derivatives_toward_posterior_mean': projected,
            'gap_rmse_m': float(error[gap].square().mean().sqrt()),
            'force_profile_marginal_identity_abs_error': float((identity-total.detach()).abs()),
            'young_pa': float(model.prior.decode(z[None])[0][0]), 'drag_per_s': float(model.prior.decode(z[None])[1][0])}
    write_json(output/'summary.json', {'case': case, 'source_fingerprint_sha256': source_fingerprint(),
        'records': records, 'refit': fitted['record'],
        'normalized_distance_original_to_refit': float((original['mean']-fitted['mean']).norm()),
        'qualification': 'observation-only local optimization restarted from sampled posterior mean; rules out a loose stopping tolerance or this alternative local start, not all possible modes'})


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--study', type=Path, default=Path('runs/calibrated-noise-published/comparison'))
    p.add_argument('--posterior', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args(); run(a.study, a.posterior, a.output)
