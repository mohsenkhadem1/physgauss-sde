"""Small conditional numerical posterior; no unconditional learned-posterior claim.

Only support log-stiffness and x-location are unknown. All other physical logits
are fixed to the true world's values. Operational quadrature convergence checks
are explicit; global mode exclusion is supported by a scan/multistart search, not
proved mathematically. No full-six-parameter reference is attempted.
"""
import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import minimize
from scipy.special import logsumexp

from scripts.gaussian_boundary_flow.model import observe, LOW, HIGH
from scripts.gaussian_boundary_flow.refined import RefinedBlock
from scripts.gaussian_boundary_flow_v02.noise import noisy_context

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'runs/gaussian-boundary-flow-v02-reference'
REPORT = ROOT / 'reports/gaussian-boundary-flow-v02/reference-posterior'
ACTIVE = (1, 3)
SIGMA = .00005


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2) + '\n'); tmp.replace(path)


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def trapezoid_weights(axis):
    axis = np.asarray(axis, float)
    if axis.ndim != 1 or len(axis) < 2 or not np.all(np.diff(axis) > 0):
        raise ValueError('Increasing one-dimensional quadrature axis required')
    w = np.empty_like(axis)
    w[0] = (axis[1]-axis[0])/2; w[-1] = (axis[-1]-axis[-2])/2
    w[1:-1] = (axis[2:]-axis[:-2])/2
    return w


def weighted_quantile(values, weights, probabilities=(.05, .5, .95)):
    """Centered mass-CDF interpolation, combining coincident quadrature nodes."""
    values = np.asarray(values, float); weights = np.asarray(weights, float)
    good = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    values, weights = values[good], weights[good]
    if not len(values): raise ValueError('Positive finite probability mass required')
    unique, inverse = np.unique(values, return_inverse=True)
    mass = np.bincount(inverse, weights=weights, minlength=len(unique))
    mass /= mass.sum()
    cdf = np.cumsum(mass)-mass/2
    return np.interp(probabilities, cdf, unique)


def summarize_grid(center, transform, extent, n, log_density):
    """Integrate a density in original q coordinates on an affine square grid."""
    axis = np.linspace(-extent, extent, n)
    w = trapezoid_weights(axis)
    base = np.stack(np.meshgrid(axis, axis, indexing='ij'), -1).reshape(-1, 2)
    q = np.asarray(center)[None] + base @ np.asarray(transform).T
    log_density = np.asarray(log_density, float).reshape(n, n)
    det = abs(np.linalg.det(transform))
    if det <= 0 or not np.isfinite(det): raise ValueError('Nonsingular finite transform required')
    log_weight = log_density + np.log(w[:, None]*w[None, :]) + math.log(det)
    logZ = logsumexp(log_weight)
    if not np.isfinite(logZ): raise ValueError('Quadrature has no finite posterior mass')
    mass = np.exp(log_weight-logZ).ravel()
    mean = mass @ q; delta = q-mean
    covariance = np.einsum('n,ni,nj->ij', mass, delta, delta)
    std = np.sqrt(np.diag(covariance))
    quantiles = np.stack([weighted_quantile(q[:, j], mass) for j in range(2)])
    edge = np.zeros((n, n), bool); edge[[0, -1], :] = True; edge[:, [0, -1]] = True
    return {'extent': extent, 'n': n, 'log_evidence': float(logZ),
            'mean_q': mean.tolist(), 'std_q': std.tolist(),
            'covariance_q': covariance.tolist(), 'quantiles_q': quantiles.tolist(),
            'outer_edge_mass': float(mass[edge.ravel()].sum())}


def compare_quadratures(previous, current, thresholds):
    scale = np.maximum(np.asarray(current['std_q']), 1e-12)
    changes = {
        'log_evidence_change': abs(current['log_evidence']-previous['log_evidence']),
        'mean_change_in_posterior_sd': float(np.max(np.abs(np.asarray(current['mean_q'])-previous['mean_q'])/scale)),
        'quantile_change_in_posterior_sd': float(np.max(np.abs(np.asarray(current['quantiles_q'])-previous['quantiles_q'])/scale[:, None])),
    }
    changes['passed'] = all(changes[key] <= thresholds[key] for key in thresholds)
    return changes


class Likelihood:
    def __init__(self, block, true_z, observed, deadline, max_evaluations):
        self.block = block; self.true_z = true_z; self.observed = observed
        self.deadline = deadline; self.max_evaluations = max_evaluations
        self.count = 0; self.cache = {}

    def log_density(self, q):
        if time.monotonic() > self.deadline: raise TimeoutError('20 minute reference budget exhausted')
        q = np.atleast_2d(np.asarray(q, float))
        result = np.full(len(q), -np.inf)
        pending = []
        for i, point in enumerate(q):
            if np.any(point < 0) or np.any(point > 1): continue
            key = tuple(point)
            if key in self.cache: result[i] = self.cache[key]
            else: pending.append((i, key))
        for start in range(0, len(pending), 128):
            chunk = pending[start:start+128]
            self.count += len(chunk)
            if self.count > self.max_evaluations: raise RuntimeError('Per-world evaluation cap exceeded')
            if time.monotonic() > self.deadline: raise TimeoutError('20 minute reference budget exhausted')
            pts = torch.tensor([item[1] for item in chunk], dtype=torch.float64, device=self.block.device)
            z = self.true_z.expand(len(pts), -1).clone()
            z[:, list(ACTIVE)] = torch.logit(pts)
            with torch.no_grad():
                prediction = observe(self.block.rollout(z, duration=.4), noise=False)
                ll = -.5*((prediction-self.observed)/SIGMA).square().sum(-1)
                # Include the known 144-dimensional Gaussian likelihood constant.
                ll -= prediction.shape[-1]*math.log(SIGMA*math.sqrt(2*math.pi))
            values = ll.cpu().numpy()
            if not np.isfinite(values).all(): raise RuntimeError('Nonfinite likelihood')
            for (i, key), value in zip(chunk, values):
                self.cache[key] = float(value); result[i] = value
        return result

    def objective(self, q):
        h = 1e-5
        points = [np.asarray(q).copy()]
        for j in range(2):
            lo = np.asarray(q).copy(); hi = lo.copy()
            lo[j] = max(1e-8, lo[j]-h); hi[j] = min(1-1e-8, hi[j]+h)
            points.extend([lo, hi])
        values = -self.log_density(np.array(points))
        gradient = np.array([(values[2*j+2]-values[2*j+1])/(points[2*j+2][j]-points[2*j+1][j]) for j in range(2)])
        return values[0], gradient

    def fisher_transform(self, mode):
        h = 1e-5; points = []
        for j in range(2):
            lo = mode.copy(); hi = mode.copy()
            lo[j] = max(0., lo[j]-h); hi[j] = min(1., hi[j]+h)
            points.extend([lo, hi])
        q = torch.tensor(np.array(points), dtype=torch.float64, device=self.block.device)
        z = self.true_z.expand(4, -1).clone(); z[:, list(ACTIVE)] = torch.logit(q)
        with torch.no_grad():
            mu = observe(self.block.rollout(z, duration=.4), noise=False).cpu().numpy()
        J = np.stack([(mu[2*j+1]-mu[2*j])/(points[2*j+1][j]-points[2*j][j])/SIGMA for j in range(2)], -1)
        H = J.T @ J
        values, vectors = np.linalg.eigh(H)
        if np.min(values) <= 0 or not np.isfinite(values).all():
            raise RuntimeError('Conditional Fisher metric is singular')
        return vectors @ np.diag(values**-.5), values, H


def integrate(engine, center, transform, extent, n, directory):
    path = directory / f'quadrature-L{extent:g}-n{n}.npz'
    axis = np.linspace(-extent, extent, n)
    base = np.stack(np.meshgrid(axis, axis, indexing='ij'), -1).reshape(-1, 2)
    q = center[None]+base @ transform.T
    if path.exists():
        saved = np.load(path)
        if not np.array_equal(saved['q'], q): raise RuntimeError('Saved quadrature coordinates changed')
        density = saved['log_density']
    else:
        density = engine.log_density(q)
        tmp = path.with_suffix('.tmp')
        with tmp.open('wb') as f:
            np.savez_compressed(f, q=q, log_density=density)
        tmp.replace(path)
    return summarize_grid(center, transform, extent, n, density)


def world_reference(block, true_z, observed, plan, directory, deadline):
    directory.mkdir(parents=True, exist_ok=True)
    engine = Likelihood(block, true_z, observed, deadline, plan['max_evaluations_per_world'])
    axis = np.linspace(0, 1, plan['global_scan_n'])
    grid = np.stack(np.meshgrid(axis, axis, indexing='ij'), -1).reshape(-1, 2)
    ll = engine.log_density(grid)
    starts = []
    for i in np.argsort(-ll):
        point = np.clip(grid[i], 1e-8, 1-1e-8)
        if all(np.linalg.norm(point-other) >= .15 for other in starts): starts.append(point)
        if len(starts) == plan['multistarts']: break
    fits = []
    for initial in starts:
        fit = minimize(engine.objective, initial, method='L-BFGS-B', jac=True,
                       bounds=[(1e-8, 1-1e-8)]*2,
                       options={'maxiter': 80, 'maxfun': 120, 'ftol': 1e-12, 'gtol': 1e-4})
        fits.append({'q': fit.x.tolist(), 'log_density': float(-fit.fun),
                     'success': bool(fit.success), 'message': str(fit.message)})
    best = max(fits, key=lambda fit: fit['log_density']); mode = np.array(best['q'])
    transform, eigenvalues, fisher = engine.fisher_transform(mode)
    inv = np.linalg.inv(transform)
    competing = [fit for fit in fits if best['log_density']-fit['log_density'] < 12
                 and np.linalg.norm(inv @ (np.array(fit['q'])-mode)) > 2]
    result = {'map_fits': fits, 'map_q': mode.tolist(), 'fisher_eigenvalues': eigenvalues.tolist(),
              'fisher_condition_number': float(eigenvalues.max()/eigenvalues.min()),
              'transform': transform.tolist(), 'competing_modes': competing,
              'tail_checks': [], 'resolution_checks': [], 'converged': False}
    if competing:
        result['failure'] = 'Multiple competitive modes outside one local quadrature window'
        return result
    previous = None; chosen = None; tail_ok = False
    for extent in plan['tail_extents']:
        current = integrate(engine, mode, transform, extent, 65, directory)
        check = None if previous is None else compare_quadratures(previous, current, plan['tail_thresholds'])
        result['tail_checks'].append({'quadrature': current, 'change': check})
        if previous is not None and check['passed'] and current['outer_edge_mass'] < plan['max_edge_mass']:
            tail_ok = True; chosen = current; break
        previous = current
    if not tail_ok:
        result['failure'] = 'Tail expansion did not converge'
        result['forward_evaluations'] = engine.count
        return result
    previous = None; resolution_ok = False
    for n in plan['resolution_sizes']:
        current = integrate(engine, mode, transform, chosen['extent'], n, directory)
        check = None if previous is None else compare_quadratures(previous, current, plan['resolution_thresholds'])
        result['resolution_checks'].append({'quadrature': current, 'change': check})
        if previous is not None and check['passed']:
            resolution_ok = True; chosen = current; break
        previous = current
    result['converged'] = bool(tail_ok and resolution_ok)
    result['forward_evaluations'] = engine.count
    if not resolution_ok: result['failure'] = 'Resolution refinement did not converge'
    result['posterior'] = chosen if resolution_ok else current
    posterior = result['posterior']
    covariance = np.array(posterior['covariance_q']); std = np.sqrt(np.diag(covariance))
    result['posterior_correlation'] = float(covariance[0, 1]/(std[0]*std[1]))
    quantile_q = np.asarray(posterior['quantiles_q'])
    truth_q = torch.sigmoid(true_z[0, list(ACTIVE)]).cpu().numpy()
    result['truth_q'] = truth_q.tolist()
    result['truth_in_conditional_90_intervals'] = ((truth_q >= quantile_q[:, 0]) & (truth_q <= quantile_q[:, 2])).tolist()
    low = np.array([float(LOW[1])+.8, float(LOW[3])])
    width = np.array([float(HIGH[1]-LOW[1]), float(HIGH[3]-LOW[3])])
    result['conditional_quantiles'] = {
        'support_stiffness_N_per_m3': np.exp(low[0]+width[0]*quantile_q[0]).tolist(),
        'support_x_m': (low[1]+width[1]*quantile_q[1]).tolist(),
    }
    result['truth_physical'] = {
        'support_stiffness_N_per_m3': float(np.exp(low[0]+width[0]*truth_q[0])),
        'support_x_m': float(low[1]+width[1]*truth_q[1]),
    }
    return result


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--device', choices=('cpu', 'cuda:0'), default='cuda:0')
    args = parser.parse_args(); torch.set_num_threads(2); torch.backends.cuda.matmul.allow_tf32 = False
    paths = [Path(__file__).resolve(), ROOT/'scripts/gaussian_boundary_flow/model.py',
             ROOT/'scripts/gaussian_boundary_flow/refined.py', ROOT/'scripts/gaussian_boundary_flow/qualification.py',
             ROOT/'src/physgauss/gaussian_boundary.py', ROOT/'src/physgauss/physics.py',
             ROOT/'scripts/gaussian_boundary_flow_v02/noise.py']
    plan = {'scope': 'Two-parameter conditional numerical posterior; all other true parameters fixed',
            'world_seed': 729131, 'noise_seed': 729132, 'world_count': 4,
            'unknown_parameters': ['support log-stiffness', 'support x-location'],
            'prior': 'Independent uniform in normalized q; logK in [11.8,14.2], x in [-.035,.035]m',
            'sigma_m': SIGMA, 'noise_draws': 1, 'precision': 'float64', 'device': args.device,
            'global_scan_n': 17, 'multistarts': 6, 'max_evaluations_per_world': 120000,
            'budget_seconds': 1200, 'tail_extents': [4., 6., 8., 12.],
            'resolution_sizes': [33, 65, 129, 257], 'max_edge_mass': 1e-5,
            'tail_thresholds': {'log_evidence_change': .001, 'mean_change_in_posterior_sd': .005,
                                'quantile_change_in_posterior_sd': .02},
            'resolution_thresholds': {'log_evidence_change': .001, 'mean_change_in_posterior_sd': .005,
                                      'quantile_change_in_posterior_sd': .02},
            'limitation': 'Operational grid/tail stability plus multistart search; no mathematical exclusion of undiscovered remote modes',
            'source_sha256': {str(path.relative_to(ROOT)): sha(path) for path in paths}}
    OUT.mkdir(parents=True, exist_ok=True); REPORT.mkdir(parents=True, exist_ok=True)
    if (OUT/'plan.json').exists():
        if json.loads((OUT/'plan.json').read_text()) != plan: raise RuntimeError('Immutable reference plan differs')
    else: write_json(OUT/'plan.json', plan)
    deadline = time.monotonic()+plan['budget_seconds']; started = time.monotonic()
    generator = torch.Generator().manual_seed(plan['world_seed'])
    z = torch.logit(torch.rand(4, 6, dtype=torch.float64, generator=generator).clamp(.001, .999)).to(args.device)
    block = RefinedBlock(dtype=torch.float64, device=args.device)
    with torch.no_grad():
        clean = observe(block.rollout(z, duration=.4), noise=False).cpu()
        observed = noisy_context(clean, torch.zeros_like(clean), torch.ones_like(clean), SIGMA,
                                 torch.Generator().manual_seed(plan['noise_seed'])).to(args.device)
    torch.save({'truth_z': z.cpu(), 'clean': clean, 'observed': observed.cpu()}, OUT/'observations.pt')
    results = {'plan_sha256': sha(OUT/'plan.json'), 'scope': plan['scope'], 'worlds': []}
    for i in range(4):
        path = OUT/f'world-{i}'/'result.json'
        if path.exists(): result = json.loads(path.read_text())
        else:
            try: result = world_reference(block, z[i:i+1], observed[i:i+1], plan, path.parent, deadline)
            except Exception as error: result = {'converged': False, 'failure': f'{type(error).__name__}: {error}'}
            result['world_index'] = i; write_json(path, result)
        results['worlds'].append(result)
        print(i, result.get('converged'), result.get('failure'), result.get('posterior_correlation'), flush=True)
        write_json(REPORT/'results.json', results)
    results['elapsed_seconds'] = time.monotonic()-started
    results['all_converged'] = all(world['converged'] for world in results['worlds'])
    results['source_hashes_unchanged'] = all(sha(ROOT/p) == h for p, h in plan['source_sha256'].items())
    results['conditioning_warning'] = 'Do not compare these intervals to unconditional six-parameter MDN/flow intervals as if conditioning were identical'
    results['calibration_warning'] = 'Four worlds cannot establish frequentist coverage or posterior calibration'
    write_json(REPORT/'results.json', results)


if __name__ == '__main__': main()
