"""Continuous log-parameter quadrature with joint Gaussian numerical discrepancy.

FEM is the forward model, not a learned PhysTwin posterior. CUDA accelerates
likelihoods and predictive moments; all posterior calculations use float64.
"""
from pathlib import Path
import argparse
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.special import ndtri
from scripts.phystwin.uncertainty_reference import BASE, BOUNDS, frozen_protocol

Z90 = float(ndtri(.95))


def make_interpolator(size=9):
    fields = np.stack([np.load(BASE / 'fields' / f'bank{size}-{i}-{j}.npz')['fields']
                       for i in range(size) for j in range(size)]).reshape(size, size, 729)
    axes = tuple(np.linspace(*b, size) for b in BOUNDS)
    return RegularGridInterpolator(axes, fields, method='linear', bounds_error=True)


def grid(size):
    axes = [np.linspace(*b, size) for b in BOUNDS]
    nodes = np.stack(np.meshgrid(*axes, indexing='ij'), axis=-1).reshape(-1, 2)
    weights = np.ones((size, size)); weights[[0, -1], :] *= .5; weights[:, [0, -1]] *= .5
    return nodes, weights.ravel() / weights.sum()


def indices():
    # Match the structured coarse material ordering used by the reference.
    from scripts.phystwin.controlled_reference import ControlledBlock
    rest = ControlledBlock(2, 2000., 1000.).rest
    right = np.flatnonzero(np.isclose(rest[:, 0], .24))
    prefix = (right[:, None] * 3 + np.arange(3)).ravel()
    marker = np.argmin(np.linalg.norm(rest - [.12, .04, .04], axis=1))
    marker_coords = marker * 3 + np.arange(3)
    return [prefix, np.r_[prefix, marker_coords], np.r_[prefix, 243+marker_coords]], np.arange(486, 729)


def discrepancy(residuals):
    """Fit only development residuals; no calibration/test labels enter here."""
    bias = residuals.mean(0)
    empirical = np.cov(residuals, rowvar=False, ddof=1)
    covariance = .9 * empirical + .1 * np.diag(np.diag(empirical)) + np.eye(residuals.shape[1]) * 1e-12
    return bias, covariance


def conditional_components(fields, bias, covariance, observed, target, sigma):
    """Analytically condition the shared discrepancy on noisy observations."""
    S = covariance[np.ix_(observed, observed)] + np.eye(len(observed)) * sigma**2
    gain = np.linalg.solve(S, covariance[np.ix_(observed, target)]).T
    fobs = fields[:, observed] + bias[observed]
    adjusted = fields[:, target] + bias[target] - fobs @ gain.T
    conditional = covariance[np.ix_(target, target)] - gain @ covariance[np.ix_(observed, target)]
    return fobs, adjusted, gain, np.maximum(np.diag(conditional), 0), np.linalg.cholesky(S)


def marginal_quantiles(mass, axis):
    """Integrate the trapezoidal marginal density before inverting its CDF."""
    trapezoid = np.ones(len(axis)); trapezoid[[0, -1]] = .5
    density = mass / trapezoid[None, :]
    area = (density[:, :-1] + density[:, 1:]) * np.diff(axis)[None, :] / 2
    cdf = np.c_[np.zeros(len(mass)), np.cumsum(area, axis=1)]
    cdf /= cdf[:, -1:]
    return np.array([np.interp([.05, .5, .95], row, axis) for row in cdf])


class Posterior:
    def __init__(self, fields, nodes, prior, bias, covariance, observed, target, sigma, device='cuda:0'):
        import torch
        self.torch, self.device = torch, device
        self.nodes, self.prior = nodes, prior
        fobs, adjusted, gain, conditional_var, chol = conditional_components(fields, bias, covariance, observed, target, sigma)
        self.chol = chol
        self.fobs = self.tensor(np.linalg.solve(chol, fobs.T).T)
        self.adjusted = self.tensor(adjusted)
        self.gain = self.tensor(gain)
        self.var = self.tensor(conditional_var)
        self.logprior = self.tensor(np.log(prior))
        self.size = int(round(np.sqrt(len(prior))))

    def tensor(self, x):
        return self.torch.as_tensor(x, dtype=self.torch.float64, device=self.device)

    def infer(self, y):
        torch = self.torch
        yt = self.tensor(np.linalg.solve(self.chol, np.asarray(y).T).T)
        squared = (yt*yt).sum(1)[:, None] + (self.fobs*self.fobs).sum(1)[None, :] - 2 * yt @ self.fobs.T
        weights = torch.softmax(self.logprior[None, :] - .5*squared, dim=1)
        centered_mean = weights @ self.adjusted
        mean = centered_mean + self.tensor(y) @ self.gain.T
        variance = weights @ (self.adjusted*self.adjusted) - centered_mean**2 + self.var
        sd = torch.sqrt(torch.clamp(variance, min=1e-24))
        marginal = weights.reshape(-1, self.size, self.size)
        marginals = [marginal.sum(2).cpu().numpy(), marginal.sum(1).cpu().numpy()]
        axes = [np.unique(self.nodes[:, d]) for d in range(2)]
        quantiles = [marginal_quantiles(m, axis) for m, axis in zip(marginals, axes)]
        return dict(mean=mean.cpu().numpy(), sd=sd.cpu().numpy(),
                    log_parameter_quantiles=np.stack(quantiles, axis=1),
                    effective_nodes=(1/(weights*weights).sum(1)).cpu().numpy())


def observations(spec, split):
    # Separate fixed seeds by split; all policies share the same prefix/outcome noise.
    rng = np.random.default_rng({'development': 201, 'calibration': 202, 'test': 203}[split])
    obs_indices, target = indices()
    ys = [[] for _ in obs_indices]; truth = []
    for i in range(len(spec[split])):
        field = np.load(BASE/'fields'/f'{split}-{i}-n8.npz')['fields'].ravel()
        for _ in range(spec['noise_repeats']):
            noise = rng.normal(0, spec['sigma_m'], size=729)
            for a, ix in enumerate(obs_indices):
                ys[a].append(field[ix]+noise[ix])
            truth.append(field[target])
    return [np.array(y) for y in ys], np.array(truth)


def predict_split(spec, interpolator, split, size, bias, covariance, device):
    nodes, prior = grid(size); fields = interpolator(nodes)
    obs_indices, target = indices(); ys, truth = observations(spec, split)
    results = []
    for ix, y in zip(obs_indices, ys):
        model = Posterior(fields, nodes, prior, bias, covariance, ix, target, spec['sigma_m'], device)
        results.append(model.infer(y))
        del model
    return results, truth


def convergence(a, b, spec):
    checks = dict(mean_m=max(float(np.max(abs(x['mean']-y['mean']))) for x,y in zip(a,b)),
        sd_m=max(float(np.max(abs(x['sd']-y['sd']))) for x,y in zip(a,b)),
        log_parameter_quantile=max(float(np.max(abs(x['log_parameter_quantiles']-y['log_parameter_quantiles']))) for x,y in zip(a,b)))
    checks['passed'] = all(checks[k] <= v for k,v in spec['quadrature_gate'].items())
    return checks


def conformal_scale(predictions, truth, repeats):
    # One independent exchangeable unit is an object, not a coordinate or repeat.
    scores = np.stack([np.max(abs(truth-p['mean'])/(Z90*p['sd']), axis=1) for p in predictions])
    object_scores = scores.reshape(3, -1, repeats).max(axis=(0,2))
    rank = int(np.ceil((len(object_scores)+1)*.9))
    assert rank <= len(object_scores)
    return max(1., float(np.sort(object_scores)[rank-1])), object_scores


def score(predictions, truth, parameters, repeats, scale=1.):
    reports=[]
    true_log = np.repeat(np.log(parameters), repeats, axis=0)
    for p in predictions:
        error = truth-p['mean']; half=Z90*scale*p['sd']; covered=abs(error)<=half
        q=p['log_parameter_quantiles']
        reports.append(dict(rmse_m=float(np.sqrt(np.mean(error**2))), coordinate_coverage=float(covered.mean()),
            mean_width_m=float(np.mean(2*half)), whole_field_trial_coverage=float(covered.all(1).mean()),
            object_all_repeats_coverage=float(covered.reshape(-1,repeats,243).all((1,2)).mean()),
            E_coverage90=float(((true_log[:,0]>=q[:,0,0]) & (true_log[:,0]<=q[:,0,2])).mean()),
            K_coverage90=float(((true_log[:,1]>=q[:,1,0]) & (true_log[:,1]<=q[:,1,2])).mean())))
    allcovered=np.stack([abs(truth-p['mean'])<=Z90*scale*p['sd'] for p in predictions])
    simultaneous=allcovered.reshape(3,-1,repeats,243).all((0,2,3))
    return dict(policies=reports, simultaneous_object_coverage=float(simultaneous.mean()),
                objects_passed=int(simultaneous.sum()), objects=len(simultaneous))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--device',default='cuda:0')
    parser.add_argument('--bank-grid',type=int,choices=[9,17],default=9)
    parser.add_argument('--stage',choices=['development','calibrate','test'],default='development')
    args=parser.parse_args();spec=frozen_protocol();out=BASE/'analysis';out.mkdir(exist_ok=True)
    interp=make_interpolator(args.bank_grid)
    dev=np.array(spec['development']);predicted=interp(np.log(dev))
    fine=np.stack([np.load(BASE/'fields'/f'development-{i}-n8.npz')['fields'].ravel() for i in range(len(dev))])
    direct=np.stack([np.load(BASE/'fields'/f'development-{i}-n4.npz')['fields'].ravel() for i in range(len(dev))])
    interpolation_rms=np.sqrt(np.mean((predicted-direct)**2,axis=1))
    numerical_rms=np.sqrt(np.mean((fine-direct)**2,axis=1))
    bias,covariance=discrepancy(fine-predicted)
    diagnostic=dict(bank_grid=args.bank_grid,interpolation_rms_m=interpolation_rms.tolist(),
        numerical_rms_m=numerical_rms.tolist(),interpolation_passed=bool(max(interpolation_rms)<spec['interpolation_gate_rms_m']),
        discrepancy_mean_rms_m=float(np.sqrt(np.mean(bias**2))),discrepancy_sd_rms_m=float(np.sqrt(np.mean(np.diag(covariance)))))
    (out/'development.json').write_text(json.dumps(diagnostic,indent=2)+'\n')
    print('DEVELOPMENT',json.dumps(diagnostic),flush=True)
    if not diagnostic['interpolation_passed']:
        raise RuntimeError('Development interpolation gate failed; refine bank before sealing calibration')
    split={'development':'development','calibrate':'calibration','test':'test'}[args.stage]
    models={'continuous_only':(np.zeros(729),np.zeros((729,729))), 'discrepancy':(bias,covariance)}
    if args.stage=='test':
        seal=json.loads((out/'calibration-seal.json').read_text())
        assert seal['bank_grid']==args.bank_grid and seal['protocol']==spec
    summary={};saved={}
    for name,(mu,C) in models.items():
        previous=None;checks=[]
        for size in spec['quadrature_sizes']:
            result,truth=predict_split(spec,interp,split,size,mu,C,args.device)
            if previous is not None:
                check=convergence(previous,result,spec);check.update(coarse=previous_size,fine=size);checks.append(check)
                print('QUADRATURE',split,name,json.dumps(check),flush=True)
                if check['passed']:break
            previous=result;previous_size=size
        if not checks[-1]['passed']:
            raise RuntimeError('Quadrature did not converge; no posterior claim or test report')
        summary[name]=dict(grid_size=size,convergence=checks,raw=score(result,truth,spec[split],spec['noise_repeats']))
        if args.stage=='calibrate':
            scale,scores=conformal_scale(result,truth,spec['noise_repeats'])
            summary[name]['conformal_scale']=scale;summary[name]['calibration_object_scores']=scores.tolist()
        if args.stage=='test':
            scale=seal['models'][name]['conformal_scale']
            summary[name]['conformal_scale']=scale
            summary[name]['calibrated']=score(result,truth,spec[split],spec['noise_repeats'],scale)
        for a,p in enumerate(result):
            saved.update({f'{name}_{a}_{k}':v for k,v in p.items()})
    np.savez_compressed(out/f'{split}-predictions.npz',truth=truth,**saved)
    (out/f'{split}-summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    if args.stage=='calibrate':
        seal=dict(protocol=spec,bank_grid=args.bank_grid,models=summary)
        sealpath=out/'calibration-seal.json'
        if sealpath.exists():assert json.loads(sealpath.read_text())==seal,'Calibration already frozen with different outputs'
        else:sealpath.write_text(json.dumps(seal,indent=2)+'\n')
    print('COMPLETE',split,json.dumps(summary),flush=True)

if __name__=='__main__':main()
