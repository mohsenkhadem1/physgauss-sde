"""Run a four-arm, sparse dynamic-image PAC-NeRF transfer benchmark.

Without marginalization means explicit joint Gaussian force variables with
the SAME prior. Their integrated path posterior is mathematically identical.
Ground truth after frame zero is accessible only in score(), after inference.
"""
import argparse
from dataclasses import replace
import hashlib
import json
import math
from pathlib import Path
import time
import shutil
import pickle
import types
import subprocess
from pathlib import PurePosixPath, PureWindowsPath
import torch
from physgauss.pacnerf_data import load_scene, evaluation_particles
from physgauss.pacnerf_model import PACModel
from physgauss.split_hmc import SplitHMCConfig, sample_split_hmc
from physgauss.chain_diagnostics import diagnose_quantities
from physgauss.cli import source_fingerprint
from physgauss.likelihood_audit import _environment


def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def cases():
    return [(scene, views, noise, force) for scene in ['torus', 'bird'] for views in [4, 6]
            for noise in ['constant', 'neural'] for force in ['marginal', 'joint']]


def load_prepared(path):
    """Read our own verified artifact across Linux/Windows path classes.

    Only the inert raw-data root metadata is a pathlib object. It is not used
    for fitting/scoring after preparation. Preserve all tensor bytes unchanged.
    This is not a loader for arbitrary third-party pickle checkpoints.
    """
    class PortableUnpickler(pickle.Unpickler):
        def find_class(self, module, name):
            if module == 'pathlib' and name == 'PosixPath': return PurePosixPath
            if module == 'pathlib' and name == 'WindowsPath': return PureWindowsPath
            return super().find_class(module,name)
    portable = types.ModuleType('pacnerf_portable_pickle')
    portable.Unpickler = PortableUnpickler
    portable.load,portable.loads = pickle.load,pickle.loads
    return torch.load(path,weights_only=False,pickle_module=portable,map_location='cpu')


def prepare(root, output, size=32):
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    calibration = json.loads(Path('examples/calibrated-noise-setup/calibration.json').read_text())
    for path in Path(root).glob('*-manifest.json'): shutil.copyfile(path, output/path.name)
    for scene in ['torus', 'bird']:
        data = load_scene(root, scene, size)
        model = PACModel(data, calibration)
        static = model.renderer.calibrate()
        torch.save({'data': data, 'colors': model.renderer.colors, 'calibration': calibration,
                    'static': static, 'mechanics': model.mechanics}, output/f'{scene}-inputs.pt')
        # Future particles go into a SEPARATE evaluation file, not model input.
        torch.save({'particles': evaluation_particles(data)}, output/f'{scene}-evaluation.pt')
        write(output/f'{scene}-manifest.json', {'scene': scene, 'source': 'official UCLA Box PAC-NeRF',
            'source_geometry_sha256': data['geometry_sha256'], 'particles': data['particle_count'],
            'static': static, 'gaussian_indices': data['gaussian_indices'].tolist(),
            'mode_basis': model.basis.tolist(), 'mode_eigenvalues_per_s2': model.eigen.tolist(),
            'image_size': size, 'times_s': data['times'].tolist(),
            'inputs_sha256': hashlib.sha256((output/f'{scene}-inputs.pt').read_bytes()).hexdigest(),
            'evaluation_sha256': hashlib.sha256((output/f'{scene}-evaluation.pt').read_bytes()).hexdigest()})
        print('prepared', scene, static, flush=True)


def make_model(inputs, index):
    scene, views, noise, force = cases()[index]
    path=Path(inputs)/f'{scene}-inputs.pt'
    manifest=json.loads((Path(inputs)/f'{scene}-manifest.json').read_text())
    if hashlib.sha256(path.read_bytes()).hexdigest()!=manifest['inputs_sha256']:
        raise ValueError('Prepared input hash differs from its frozen manifest')
    bundle = load_prepared(path)
    # Internal artifact is produced by this script, never an arbitrary checkpoint.
    frames = [0, 3, 10, 13] if views == 4 else [0, 2, 4, 9, 11, 13]
    model = PACModel(bundle['data'], bundle['calibration'], noise, force, frames, schedule=0, mechanics=bundle.get('mechanics'))
    model.renderer.colors = bundle['colors']
    return model, bundle


def initialize(model):
    z = torch.zeros(model.dimension, dtype=torch.float64)
    m = model.modes
    # Initial v=0, ballistic seed with a soft landing. No future particles or
    # withheld observations supply an initial guess.
    for t in range(1, 14):
        nodes = model.data['rest'].clone()
        nodes[:, 1] = (nodes[:, 1]-.5*9.8*(t/24)**2).clamp_min(.025)
        q = torch.einsum('nc,ncm->m', nodes-model.data['rest'], model.basis)
        z[2+m+(t-1)*2*m:2+m+(t-1)*2*m+m] = q/.10
    q, _ = model.decode(z[None])
    velocity = (q[:, 1:]-q[:, :-1])/model.dt
    state = z[2+m:model.base_dimension].reshape(13, 2*m)
    state[:, m:] = velocity[0]/.5
    return z


def optimize(model, initial, max_iter=350):
    z = initial.clone().requires_grad_()
    optimizer = torch.optim.LBFGS([z], lr=1., max_iter=max_iter, max_eval=max_iter*2,
        history_size=40, tolerance_grad=1e-6, tolerance_change=1e-10, line_search_fn='strong_wolfe')
    calls = 0
    def closure():
        nonlocal calls
        optimizer.zero_grad(); loss = model.objective(z[None]).sum()
        if not torch.isfinite(loss): raise ValueError('Nonfinite optimization target')
        loss.backward(); calls += 1
        if calls % 50 == 0: print('fit', calls, float(loss.detach()), flush=True)
        return loss
    optimizer.step(closure)
    value = closure()
    return z.detach(), {'objective': float(value.detach()), 'gradient_max_abs': float(z.grad.abs().max()),
                        'evaluations': calls, 'optimizer_iterations': optimizer.state[z]['n_iter']}


class Target:
    def __init__(self, model, mean, precision):
        self.model, self.mean = model, mean
        self.chol = torch.linalg.cholesky(precision)
        self.transform = torch.linalg.solve_triangular(self.chol.T, torch.eye(len(mean), dtype=mean.dtype), upper=True)
        self.center = model.objective(mean[None])[0].detach()

    def decode(self, w): return self.mean+w@self.transform.T

    def __call__(self, white):
        w = white.detach().requires_grad_(); z = self.decode(w)
        try:
            ratio = -self.model.objective(z)+self.center+.5*w.square().sum(-1)
            grad, = torch.autograd.grad(ratio.sum(), w)
            bad = ~torch.isfinite(ratio)|~torch.isfinite(grad).all(-1)
            return ratio.detach(), torch.nan_to_num(grad.detach()), bad
        except (ValueError, torch.linalg.LinAlgError):
            if len(w) > 1:
                rows = [self(row[None]) for row in white]
                return tuple(torch.cat([row[i] for row in rows]) for i in range(3))
            return w.new_full((len(w),), -torch.inf), torch.zeros_like(w), torch.ones(len(w), dtype=torch.bool)

    @torch.no_grad()
    def log_ratio_no_grad(self, w):
        try:
            ratio = -self.model.objective(self.decode(w))+self.center+.5*w.square().sum(-1)
            return ratio, ~torch.isfinite(ratio)
        except (ValueError, torch.linalg.LinAlgError):
            if len(w) > 1:
                rows = [self.log_ratio_no_grad(row[None]) for row in w]
                return tuple(torch.cat([row[i] for row in rows]) for i in range(2))
            return w.new_full((len(w),), -torch.inf), torch.ones(len(w), dtype=torch.bool)


def observed_hessian(function,mean,block_size=16):
    """Exact second derivatives with bounded vectorized reverse-mode batches."""
    z=mean.detach().requires_grad_();g,=torch.autograd.grad(function(z),z,create_graph=True)
    eye=torch.eye(len(z),dtype=z.dtype,device=z.device);result=torch.empty_like(eye)
    for start in range(0,len(z),block_size):
        stop=min(start+block_size,len(z))
        rows,=torch.autograd.grad(g,z,grad_outputs=eye[start:stop],is_grads_batched=True,retain_graph=stop<len(z))
        result[start:stop]=rows.detach()
    return result


def reference_precision(model, mean, metric_floor=None):
    # Exact observed Hessian is used ONLY as a proposal metric. A ridge/negative
    # curvature adjustment cannot alter the exact MH target. Record adjustment.
    began = time.time()
    h = observed_hessian(lambda z: model.dynamics(z[None]).sum(),mean)
    # Image curvature is local in just twelve pose coordinates, avoiding a
    # batch of 326 full-renderer second derivatives and excessive CI memory.
    q, _ = model.decode(mean[None]); m = model.modes
    for slot, spec in enumerate(model.specs):
        if spec[0] == 0: continue  # supplied initial pose has no latent coordinate
        def image(u):
            rgb = model.renderer(u[None, None], [spec])[0, 0]
            return .5*((rgb-model.obs[slot])/.08).square().sum()
        local = torch.autograd.functional.hessian(image, q[0, spec[0]].detach())
        start = 2+m+(spec[0]-1)*2*m
        scale=model.scale[start:start+m]
        h[start:start+m, start:start+m] += local*scale[:,None]*scale[None,:]
    h = (h+h.T)/2
    eigen, vectors = torch.linalg.eigh(h)
    floor = max(.01 if metric_floor is None else metric_floor, float(eigen.max())*1e-7)
    precision = (vectors*eigen.clamp_min(floor))@vectors.T
    return precision, {'minimum_observed_eigenvalue': float(eigen.min()), 'metric_eigenvalue_floor': floor,
                       'adjusted_eigenvalues': int((eigen < floor).sum()), 'seconds': time.time()-began}


def refine_mode(model, mean, rounds=100, checkpoint=None):
    history = []
    for iteration in range(rounds):
        z = mean.detach().requires_grad_()
        value = model.objective(z[None]).sum(); g, = torch.autograd.grad(value, z)
        precision, curvature = reference_precision(model, mean, metric_floor=.1)
        step = torch.linalg.solve(precision, g.detach())
        decrement = float((step@g).detach().clamp_min(0).sqrt())
        history.append({'iteration': iteration, 'objective': float(value.detach()),
                        'decrement': decrement, 'curvature': curvature})
        if checkpoint is not None:
            write(checkpoint/'newton_progress.json', history)
            torch.save({'mean':mean,'precision':precision}, checkpoint/'newton_progress.pt')
        print('Newton', iteration, history[-1]['objective'], decrement, flush=True)
        if decrement < .02 and curvature['minimum_observed_eigenvalue'] > 0:
            return mean, precision, {'newton_decrement': decrement, 'history': history, 'stationary': True}
        if iteration % 5 == 0:
            # Optimize in a fixed local precision metric before further Newton
            # steps. Direct pose/velocity coordinates are badly conditioned.
            transform = torch.linalg.solve_triangular(torch.linalg.cholesky(precision).T,
                torch.eye(len(mean),dtype=mean.dtype),upper=True)
            center = mean
            class Whitened:
                def objective(self, w): return model.objective(center+w@transform.T)
            w, white_fit = optimize(Whitened(), torch.zeros_like(mean), 150)
            candidate = center+w@transform.T
            with torch.no_grad(): trial = float(model.objective(candidate[None])[0])
            if math.isfinite(trial) and trial < float(value.detach()):
                history[-1]['whitened_refinement'] = white_fit
                mean = candidate.detach()
                continue
        # Bound the change in standardized coordinates, then use an exact
        # objective Armijo line search. No truth or withheld images are used.
        fraction = min(1., 2./max(float(step.norm()), 1e-12))
        slope = float(g.detach()@step)
        for _ in range(24):
            candidate = mean-fraction*step
            try:
                with torch.no_grad(): trial = float(model.objective(candidate[None])[0])
            except (ValueError, torch.linalg.LinAlgError): trial = math.inf
            if math.isfinite(trial) and trial <= float(value.detach())-1e-4*fraction*slope:
                mean = candidate.detach(); break
            fraction *= .5
        else: raise ValueError('Newton line search did not improve the observed-data target')
    raise ValueError('MAP failed the predeclared stationarity threshold; sampling not started')


@torch.no_grad()
def score(model, draws, inputs):
    truth = torch.load(Path(inputs)/f"{model.data['scene']}-evaluation.pt", weights_only=True)['particles']
    q, _ = model.decode(draws)
    # Mean uses every retained draw; interval quantiles use <=2000 evenly
    # spaced draws to bound memory. This approximation is explicitly recorded.
    mean_q = q.mean(0)
    selected = torch.linspace(0, len(q)-1, min(2000, len(q))).round().long()
    position = model.data['gaussians'][None, None]+torch.einsum('btm,gcm->btgc', q[selected], model.renderer.gaussian_basis)
    lower, upper = torch.quantile(position, position.new_tensor([.05, .95]), dim=0)
    mean = model.data['gaussians'][None]+torch.einsum('tm,gcm->tgc', mean_q, model.renderer.gaussian_basis)
    missing = [f for f in range(1, 14) if f not in model.frames]
    gap = list(range(4, 10)) if len(model.frames) == 4 else list(range(5, 9))
    # Evaluation-only oracle projection: the smallest possible particle error
    # in this fixed linear shape subspace. Never used to initialize or fit.
    basis = model.renderer.gaussian_basis.flatten(0,1)
    oracle_q = torch.linalg.lstsq(basis, (truth-model.data['gaussians'][None]).flatten(1).T).solution.T
    oracle_x = model.data['gaussians'][None]+torch.einsum('tm,gcm->tgc',oracle_q,model.renderer.gaussian_basis)
    record = {'gap_particle_coordinate_rmse_m': float((mean[gap]-truth[gap]).square().mean().sqrt()),
        'missing_particle_coordinate_rmse_m': float((mean[missing]-truth[missing]).square().mean().sqrt()),
        'gap_component_90_coverage': float(((truth[gap] >= lower[gap])&(truth[gap] <= upper[gap])).double().mean()),
        'gap_mean_width_m': float((upper[gap]-lower[gap]).mean()), 'interval_draws': len(selected),
        'evaluation_only_subspace_oracle_gap_rmse_m': float((oracle_x[gap]-truth[gap]).square().mean().sqrt()),
        'qualification': '128 fixed frame-zero-selected particle correspondences; conditional on supplied initial geometry; deterministic dataset, no force ground-truth claim'}
    # True held-out images: all cameras and times except frame-zero bootstrap
    # and explicitly observed camera/time pairs. Do not synthesize new truth.
    all_specs = [(f, c) for f in range(1, 14) for c in range(11) if (f, c) not in model.specs]
    errors, foreground_error, foreground_count = [], [], []
    for start in range(0, len(all_specs), 8):
        specs = all_specs[start:start+8]
        rgb = model.renderer(mean_q[[f for f, c in specs]][None], specs)[0]
        target = torch.stack([model.data['images'][c, f] for f, c in specs])
        errors.append((rgb-target).square().flatten(1).mean(-1))
        bg = torch.stack([model.data['backgrounds'][c] for f,c in specs])
        region = (target-bg).abs().amax(-1) > .10
        foreground_error.append(((rgb-target).square()*region[...,None]).sum())
        foreground_count.append(region.sum()*3)
    record['heldout_rgb_psnr_db'] = float(-10*torch.log10(torch.cat(errors).mean()))
    record['heldout_images'] = len(all_specs)
    record['heldout_background_difference_region_psnr_db'] = float(-10*torch.log10(torch.stack(foreground_error).sum()/torch.stack(foreground_count).sum().clamp_min(1)))
    record['region_metric_definition'] = 'evaluation-only abs(target-empty_background).max(channel) > .10; includes shadows; not a true object mask'
    return record, {'modal_pose_mean_m': mean_q.tolist(), 'particle_mean_m': mean.tolist(), 'particle_q05_m': lower.tolist(), 'particle_q95_m': upper.tolist()}


def run(inputs, output, index, pilot=False):
    scene, views, noise, force = cases()[index]
    name = f'{scene}__{views}views__{noise}__{force}'
    out = Path(output)/name; out.mkdir(parents=True, exist_ok=True)
    model, bundle = make_model(inputs, index)
    began = time.time()
    settings = {'case_index': index, 'case': name, 'source_fingerprint': source_fingerprint(),
        'worker_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=Path(__file__).resolve().parent,text=True).strip(),
        'environment': _environment(),
        'worker_sha256_lf': hashlib.sha256(Path(__file__).read_bytes().replace(b'\r\n',b'\n')).hexdigest(),
        'observed_pairs': model.specs, 'dimension': model.dimension,
        'input_sha256': hashlib.sha256((Path(inputs)/f'{scene}-inputs.pt').read_bytes()).hexdigest(),
        'bootstrap': bundle['static'], 'model': '12-mode linear elastic graph + smooth frame-held ground reaction; exact linear propagator/covariance/constant-load integrals; transferred frozen neural noise',
        'transition_integrals':'balanced 5x5 modal-pair and 3x3 affine matrix exponentials; no fixed covariance/load quadrature',
        'noise_units': 'floor .15 and balanced edge scale .20 in acceleration diffusion; transfer assumptions, not PAC-calibrated',
        'force_comparison': 'same Gaussian unknown-load prior; analytic marginalization versus explicit joint variables',
        'proposal': 'Gaussian-reference split HMC; full nonlinear image and physics gradients; no image surrogate',
        'pilot': pilot}
    settings=json.loads(json.dumps(settings,allow_nan=False))
    if (out/'settings.json').exists() and json.loads((out/'settings.json').read_text()) != settings:
        raise ValueError('Source, plan or inputs changed. Choose a fresh output directory.')
    write(out/'settings.json', settings)
    if (out/'reference.pt').exists():
        fit = torch.load(out/'reference.pt', weights_only=True)
        if not fit.get('stationarity', {}).get('stationary'):
            raise ValueError('Saved reference predates stationarity checks; choose a fresh output.')
        mean, precision = fit['mean'], fit['precision']
    else:
        mean, fit_info = optimize(model, initialize(model), 350)
        mean, precision, stationarity = refine_mode(model, mean, checkpoint=out)
        fit = {'mean': mean, 'precision': precision, 'fit': fit_info, 'stationarity': stationarity}
        torch.save(fit, out/'reference.pt')
        write(out/'fit.json', {k:v for k,v in fit.items() if k not in ['mean', 'precision']})
    target = Target(model, mean, precision)
    cfg = SplitHMCConfig(warmup=100 if pilot else 500, draws=100 if pilot else 2000,
        min_time=1., max_time=4., target_accept=.9, seed=140001+index)
    chains = torch.load(out/'chains.pt',weights_only=True) if (out/'chains.pt').exists() else None
    history = []
    for budget in ([100] if pilot else [500, 1000, 2000, 4000, 8000]):
        if chains is not None and chains['white'].shape[1] > budget: continue
        config = cfg if chains is None else replace(cfg, warmup=0, draws=budget-chains['white'].shape[1], step_size=chains['step_size'])
        if chains is None or config.draws:
            if chains is None: config = replace(config, draws=budget)
            chains = sample_split_hmc(target, model.dimension, config, resume=chains,
                progress=lambda a: print(name, a, flush=True))
        torch.save(chains, out/'chains.pt')
        z = target.decode(chains['white'].flatten(0, 1))
        log_target = chains['log_ratio']-.5*chains['white'].square().sum(-1)-target.center
        with torch.no_grad():
            conditional = [model.force_conditionals(batch) for batch in z.split(32)]
            fm,fc,normalizers = [torch.cat([batch[i] for batch in conditional]) for i in range(3)]
        monitor = torch.cat((z.reshape(4,budget,-1),log_target[...,None],
            fm.reshape(4,budget,-1),normalizers.reshape(4,budget,2)),-1)
        diagnostic = diagnose_quantities(monitor,
            [f'z{i}' for i in range(model.dimension)]+['log_target']+
            [f'conditional_force_mean{i}' for i in range(model.force_dimension)]+['logdet_Q','logdet_A'])
        passed = diagnostic['all_monitored_quantities_pass']
        history.append({'draws_per_chain': budget, 'all_pass': passed})
        write(out/'diagnostics.json', diagnostic)
        if (passed and budget >= 2000) or pilot: break
    if pilot:
        write(out/'pilot_summary.json',{'case':name,'settings':settings,'development_only':True,
            'valid_for_ranking':False,'retained_draws_per_chain':budget,'step_size':chains['step_size'],
            'counts':chains['counts'],'gradient_batches':chains['gradient_evaluations'],
            'sampling_history':history,'seconds':time.time()-began,
            'qualification':'observed-data-only development pilot; no future geometry, reconstruction or coverage score; not final convergence evidence'})
        return
    # Scoring deliberately follows the saved chain and diagnostics.
    scores, positions = score(model, z, inputs)
    map_scores, _ = score(model, mean[None], inputs)
    q_all, _ = model.decode(z)
    q_all = q_all.reshape(4, budget, 14, model.modes)
    batch_length = max(20, int(budget**.5)); batch_count = budget//batch_length
    batch_means = q_all[:, :batch_count*batch_length].reshape(4, batch_count, batch_length, 14, model.modes).mean(2).flatten(0, 1)
    positions['modal_mean_batch_mcse_m'] = (batch_means.std(0)/len(batch_means)**.5).tolist()
    positions['mcse_batch_length'] = batch_length
    positions['mcse_batches'] = len(batch_means)
    conditional_covariance = fc.mean(0); between_covariance = torch.cov(fm.T)
    positions['force'] = {'coefficient_mean':fm.mean(0).tolist(),
        'coefficient_covariance':(conditional_covariance+between_covariance).tolist(),
        'mean_conditional_covariance':conditional_covariance.tolist(),
        'between_trajectory_covariance':between_covariance.tolist(),
        'definition':'24 whitened acceleration-load coefficients; total covariance E[S]+Cov[m]; no force ground truth or Newton-unit calibration'}
    if model.force == 'joint':
        positions['force']['explicit_sample_coefficient_mean'] = z[:,model.base_dimension:].mean(0).tolist()
        positions['force']['explicit_sample_coefficient_covariance'] = torch.cov(z[:,model.base_dimension:].T).tolist()
    write(out/'posterior.json', positions)
    write(out/'summary.json', {'case': name, 'settings': settings, 'fit': fit.get('fit'),
        'map_scores': map_scores, 'posterior_scores': scores, 'convergence': {k:v for k,v in diagnostic.items() if k != 'quantities'},
        'minimum_bulk_ess': min(v['bulk_ess'] or 0 for v in diagnostic['quantities'].values()),
        'minimum_tail_ess': min(v['tail_ess'] or 0 for v in diagnostic['quantities'].values()),
        'worst_rhat': max(v['rank_folded_split_rhat'] or 999 for v in diagnostic['quantities'].values()),
        'sampling_history': history, 'step_size': chains['step_size'], 'counts': chains['counts'],
        'seconds': time.time()-began, 'valid_for_ranking': passed and not pilot})


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--prepare', action='store_true'); p.add_argument('--root', default='data/pacnerf')
    p.add_argument('--inputs', default='runs/pacnerf-inputs'); p.add_argument('--output', default='runs/pacnerf-comparison')
    p.add_argument('--case-index', type=int); p.add_argument('--pilot', action='store_true')
    a = p.parse_args(); torch.set_num_threads(2)
    import torch.backends.opt_einsum as backend
    backend.enabled = True; backend.strategy = 'greedy'
    if a.prepare: prepare(a.root, a.inputs)
    else: run(a.inputs, a.output, a.case_index, a.pilot)
