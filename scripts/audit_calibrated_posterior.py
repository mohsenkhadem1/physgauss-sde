"""Observed-curvature refinement and exact-target nonlinear sampling audit.

Uses observed RGB only. Evaluation truth is read after sampling to score the
saved trajectories. This is separate from the frozen conformal procedure.
"""
import argparse
import hashlib
import json
from pathlib import Path
import torch
from physgauss.benchmark import Observations
from physgauss.accurate_study import digest_tensor
from physgauss.cli import source_fingerprint
from physgauss.neural_marginal_study import calibrate, construct
from physgauss.neural_marginal_inference import fit_neural_marginal
from physgauss.marginal_inference import TrajectoryFitConfig
from physgauss.marginal_study import audit_gaussian
from physgauss.uncertainty_validation import finite_difference_precision
from sparse_observed_curvature import observed_precision, directional_refinement
from physgauss.mcmc import SamplerConfig, sample_nonlinear
from physgauss.chain_diagnostics import diagnose_quantities
from physgauss.physics import InvalidDeformation
from physgauss.study import write_json


class SingleReference:
    def __init__(self, fitted):
        self.mean = fitted["mean"]; self.chol = fitted["precision_cholesky"]
        self.dimension = len(self.mean)
        self.log_weights = self.mean.new_zeros(1); self.prior_weights = self.mean.new_ones(1)
        self.scales = self.mean.new_ones(1)

    def decode(self, white, indices):
        return self.mean+torch.linalg.solve_triangular(self.chol.T, white.T, upper=True).T

    def whiten_gradient(self, gradient, indices):
        return torch.linalg.solve_triangular(self.chol, gradient.T, upper=False).T


class ExactCorrection:
    def __init__(self, model, observations, reference):
        self.model = model; self.observations = observations; self.reference = reference
        with torch.no_grad(): self.center = model.objective(reference.mean[None], observations)[0]

    def value_and_grad(self, coordinates, scales):
        ratios, targets, invalid, gradients = [], [], [], []
        # Isolate invalid proposals, retaining correct rejection probability.
        for point in coordinates:
            z = point.detach().requires_grad_()
            try:
                value = self.model.objective(z[None], self.observations)[0]
                derivative, = torch.autograd.grad(value, z)
                if not bool(torch.isfinite(value)) or not bool(torch.isfinite(derivative).all()):
                    raise ValueError("Nonfinite target.")
                delta = z.detach()-self.reference.mean
                white = delta@self.reference.chol
                ratios.append((-value.detach()+self.center+.5*white.square().sum()))
                gradients.append(-derivative.detach()+white@self.reference.chol.T)
                targets.append(-value.detach()); invalid.append(False)
            except (InvalidDeformation, ValueError):
                ratios.append(z.new_tensor(-torch.inf)); targets.append(z.new_tensor(-torch.inf))
                gradients.append(torch.zeros_like(z)); invalid.append(True)
        return ((torch.stack(ratios), torch.stack(targets), torch.tensor(invalid)), torch.stack(gradients))

    @torch.no_grad()
    def __call__(self, coordinates, scales):
        try:
            values = self.model.objective(coordinates, self.observations)
        except (InvalidDeformation, ValueError):
            rows = []
            for point in coordinates:
                try: rows.append(self.model.objective(point[None], self.observations)[0])
                except (InvalidDeformation, ValueError): rows.append(point.new_tensor(torch.inf))
            values = torch.stack(rows)
        invalid = ~torch.isfinite(values)
        white = (coordinates-self.reference.mean)@self.reference.chol
        ratio = -values+self.center+.5*white.square().sum(-1)
        return torch.where(invalid, -torch.inf, ratio), -values, invalid


def audit(study, case, output, warmup=1000, draws=2000, seed=97001):
    study, output = Path(study), Path(output); output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((study/"manifest.json").read_text())
    if manifest["source_fingerprint_sha256"] != source_fingerprint(): raise ValueError("Source mismatch.")
    plan = manifest["plan"]; torch.set_num_threads(plan["threads"])
    network, calibration = calibrate(plan, study/"calibration.json")
    record = json.loads((study/"cases"/case/"summary.json").read_text())
    model = construct(plan, network, calibration, record["mode"])
    dataset = study/"datasets"/f"{record['condition']}__{record['seed']}"/"inputs.json"
    inputs = json.loads(dataset.read_text())
    observed = Observations(model.view_frames, model.times.new_tensor(inputs["observed_rgb"]), .015)
    original = torch.load(study/"cases"/case/"fit.pt", weights_only=True)
    settings = {"case": case, "source_fingerprint_sha256": source_fingerprint(),
                "warmup": warmup, "draws": draws, "seed": seed, "chains": 4,
                "directional_epsilons_white": [.01, .005, .001, .0005, .0001, .00005], "refit_decrement": .001,
                "curvature": "exact local AD/force Schur/logdet volume Hessian; independently checked against dense AD and directional differences",
                "sampler": "gradient-free Gaussian-reference pCN plus independent refresh, Metropolis-corrected full target",
                "observations_sha256": digest_tensor(observed.rgb),
                "original_fit_mean_sha256": digest_tensor(original["mean"]),
                "audit_script_sha256_normalized_lf": hashlib.sha256(Path(__file__).read_bytes().replace(b"\r\n", b"\n")).hexdigest(),
                "curvature_script_sha256_normalized_lf": hashlib.sha256((Path(__file__).parent/"sparse_observed_curvature.py").read_bytes().replace(b"\r\n", b"\n")).hexdigest()}
    if (output/"settings.json").exists() and json.loads((output/"settings.json").read_text()) != settings:
        raise ValueError("Audit source/input/settings changed; choose a new output.")
    write_json(output/"settings.json", settings)
    if (output/"mode.pt").exists(): mode = torch.load(output/"mode.pt", weights_only=True)
    else:
        mode = fit_neural_marginal(model, observed, original["mean"], TrajectoryFitConfig(iterations=20, decrement_tolerance=.001))
        torch.save(mode, output/"mode.pt")
    if not mode["record"]["stationarity_screen_pass"]: raise RuntimeError("Audit mode is not stationary.")
    file = output/"observed_hessian.pt"
    if file.exists(): checkpoint = torch.load(file, weights_only=True)
    else:
        print("Assembling exact observed curvature from local blocks", flush=True)
        fit, details, white = observed_precision(model, observed, mode)
        checks = directional_refinement(model, observed, mode, white)
        checkpoint = {"fit": fit, "audit": details, "white_hessian": white, "directional_checks": checks}
        torch.save(checkpoint, file)
    refinement = max(checkpoint["directional_checks"][-1]["relative_Hv_errors"])
    fitted = checkpoint["fit"]
    weights = {"fisher": audit_gaussian(model, observed, mode, 512, seed+1),
               "observed_hessian": audit_gaussian(model, observed, fitted, 512, seed+1)}
    write_json(output/"curvature.json", {"settings": settings, "refined_mode": mode["record"],
        "hessians": [checkpoint["audit"]], "relative_refinement": refinement,
        "directional_checks": checkpoint["directional_checks"], "importance": weights})
    reference = SingleReference(fitted)
    if (output/"chains.pt").exists(): chains = torch.load(output/"chains.pt", weights_only=True)
    else:
        config = SamplerConfig(chains=4, warmup=warmup, draws=draws, seed=seed, beta=.5,
            independent_probability=.1, transport_probability=0., langevin=False)
        chains = sample_nonlinear(reference, ExactCorrection(model, observed, reference), config,
                                   progress=lambda s: print(s, flush=True))
        torch.save(chains, output/"chains.pt")
    coordinates = chains["coordinates"]
    with torch.no_grad():
        path = model.trajectory(coordinates.flatten(0, 1)).x.reshape(4, draws, 21, 8, 3)
        gap = (model.times > .3+1e-8)&(model.times < .7-1e-8)
        gap_path = path[:, :, gap][:, :, :, model.free].flatten(2)
        force_means, force_covariances, force_samples = [], [], []
        generator = torch.Generator().manual_seed(seed+60000)
        for batch in coordinates.flatten(0, 1).split(16):
            elimination = model.elimination_at(batch)
            conditional = elimination.conditional_mean(model.residual(batch))
            white = torch.randn(conditional.shape, dtype=conditional.dtype, generator=generator)
            sample = conditional+torch.linalg.solve_triangular(elimination.cholesky.transpose(-1, -2), white[..., None], upper=True)[..., 0]
            force_means.append(conditional); force_covariances.append(elimination.conditional_covariance); force_samples.append(sample)
        force_means = torch.cat(force_means); force_covariances = torch.cat(force_covariances); force_samples = torch.cat(force_samples)
        # Monitor conditional means, so added independent force draws cannot
        # conceal poor exploration of the latent trajectories.
        monitored = torch.cat((coordinates, gap_path, chains["log_likelihood"][..., None], force_means.reshape(4, draws, 15)), -1)
        names = [f"state_{i}" for i in range(model.state_dimension)]+[f"gap_xyz_{i}" for i in range(84)]+["log_target"]+[f"force_conditional_mean_{i}" for i in range(15)]
        diagnostics = diagnose_quantities(monitored, names)
        truth = model.times.new_tensor(inputs["positions_m_evaluation_only"])
        population = path.flatten(0, 1)[:, :, model.free]
        lower, upper = torch.quantile(population, population.new_tensor([.05, .95]), dim=0)
        mean = population.mean(0); error = mean-truth[:, model.free]
        scores = {"gap_component_90_coverage": float(((truth[:, model.free] >= lower)&(truth[:, model.free] <= upper))[gap].double().mean()),
            "gap_mean_width_m": float((upper-lower)[gap].mean()), "gap_rmse_m": float(error[gap].square().mean().sqrt()),
            "qualification": "finite MCMC sample summary; usable only with adequate exploration; not repeated-case calibration"}
        force_total = model.force_prior.total_force(force_samples, model.times)
        lower_force, upper_force = torch.quantile(force_total, force_total.new_tensor([.05, .95]), dim=0)
        conditional_covariance = force_covariances.mean(0); between = torch.cov(force_means.T)
        force_summary = {"times_s": model.times.tolist(), "mean_total_N": model.force_prior.total_force(force_means.mean(0)[None], model.times)[0].tolist(),
            "q05_total_N": lower_force.tolist(), "q95_total_N": upper_force.tolist(),
            "whitened_coefficient_mean": force_means.mean(0).tolist(),
            "whitened_coefficient_covariance": (conditional_covariance+between).tolist(),
            "mean_conditional_coefficient_covariance": conditional_covariance.tolist(), "between_trajectory_coefficient_covariance": between.tolist(),
            "qualification": "Rao-Blackwell mixture over finite exact-target MCMC trajectories; conditional on model and adequate chain exploration; not force calibration"}
    summary = {"settings": settings, "curvature_relative_refinement": refinement, "importance": weights,
        "diagnostics": diagnostics, "scores": scores, "final_beta_per_chain": chains["final_beta_per_chain"],
        "transition_counts": chains["transition_counts"],
        "force_posterior": force_summary,
        "gap_draws_per_chain_m": gap_path[:, ::max(1, draws//200)].tolist(),
        "log_target_traces": chains["log_likelihood"][:, ::max(1, draws//200)].tolist(),
        "posterior_status": "finite_diagnostics_pass" if diagnostics["all_monitored_quantities_pass"] else "not_validated",
        "qualification": "Metropolis targets exact nonlinear marginalized-force density; observed curvature preconditions only. Neither local proposal ESS nor finite-chain diagnostics establishes global/posterior calibration."}
    write_json(output/"summary.json", summary)
    print("Posterior audit:", summary["posterior_status"], flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, required=True); parser.add_argument("--case", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=1000); parser.add_argument("--draws", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=97001)
    args = parser.parse_args(); audit(args.study, args.case, args.output, args.warmup, args.draws, args.seed)
