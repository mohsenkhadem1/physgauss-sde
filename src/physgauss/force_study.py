"""Structured force inference and isolated synthetic force evaluation."""
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import torch
from .benchmark import Config, ParameterPrior, SceneModel, make_benchmark
from .camera import orbit
from .chain_diagnostics import diagnose_quantities
from .force_inference import fitted_reference, GaussianTargetCorrection
from .forces import ForceConfig, ForceSceneModel
from .likelihood_audit import _environment
from .mcmc import SamplerConfig, sample_nonlinear
from .metrics import crps, evaluate, physics_diagnostics
from .physics import simulate, edge_impulses
from .stochastic import calibration_law, train_diffusion
from .study import normalized, write_json


METHODS = {"force_only": (True, False), "sde_only": (False, True), "force_and_sde": (True, True)}
CONDITIONS = {"no_load": (False, False), "external_pulse": (True, False),
              "stochastic": (False, True), "pulse_and_stochastic": (True, True)}


def quick_force_plan():
    return {"base": asdict(Config.quick()), "seeds": [4101], "conditions": ["external_pulse"],
        "methods": list(METHODS), "force_prior": asdict(ForceConfig()), "exact_hessian": False,
        "sampler": asdict(SamplerConfig(chains=4, warmup=12, draws=32, seed=81001,
            independent_probability=.25, transport_probability=0., langevin=True)), "evaluation_samples": 16}


def validate_plan(plan):
    if not plan["seeds"] or len(set(plan["seeds"])) != len(plan["seeds"]):
        raise ValueError("Need distinct realization seeds.")
    if not plan["conditions"] or not set(plan["conditions"]) <= set(CONDITIONS):
        raise ValueError("Unknown force benchmark condition.")
    if len(set(plan["conditions"])) != len(plan["conditions"]) or len(set(plan["methods"])) != len(plan["methods"]):
        raise ValueError("Repeated condition or model.")
    if not plan["methods"] or not set(plan["methods"]) <= set(METHODS):
        raise ValueError("Unknown force inference model.")
    Config(**plan["base"]).validate(); ForceConfig(**plan["force_prior"]).validate()
    sampler = SamplerConfig(**plan["sampler"]).validate()
    if plan["evaluation_samples"] % sampler.chains or not sampler.chains <= plan["evaluation_samples"] <= sampler.chains*sampler.draws:
        raise ValueError("Evaluation count must be a multiple of chain count, within retained draws.")
    if any(not isinstance(seed, int) or seed < 0 or seed == plan["base"].get("diffusion_seed", 918) for seed in plan["seeds"]):
        raise ValueError("Evaluation seeds must be nonnegative and separate from diffusion calibration.")


@torch.no_grad()
def truth_force_record(config, benchmark):
    """Evaluation only: reconstruct true substep loads/impulses and verify states."""
    model = ForceSceneModel(config, force=False)
    generator = torch.Generator().manual_seed(config.seed)
    white = (config.truth_process_scale*torch.randn(1, model.nsteps, config.noise_rank,
        dtype=torch.float64, generator=generator)) if config.truth_stochastic else None
    forces, impulses = [], []
    step = 0
    def external(t, x, v):
        nonlocal step
        acceleration = torch.zeros_like(x)
        if config.truth_external_pulse:
            strength = np.exp(-.5*((t/config.duration-.5)/.08)**2)
            acceleration = strength*(~model.mesh.pinned).to(x.dtype)[None, :, None]*x.new_tensor([.45, 0., -.30])
        forces.append(acceleration*model.mesh.mass[None, :, None])
        impulse = (edge_impulses(x, v, model.mesh, white[:, step]*model.step_sizes[step].sqrt(),
            calibration_law, model.basis) if white is not None else torch.zeros_like(x))
        impulses.append(impulse); step += 1
        return acceleration
    params = benchmark.truth_parameters
    tensor = model.times.new_tensor
    trajectory = simulate(model.mesh, model.times, tensor([params["young_pa"]]),
        tensor([params["drag_per_second"]]), tensor([params["offset_m"]]), tensor([params["velocity_m_per_second"]]),
        config.substeps, white=white, diffusion=calibration_law if white is not None else None,
        noise_basis=model.basis, external_acceleration=external)
    if not torch.allclose(trajectory.x, benchmark.truth.x, atol=1e-12, rtol=1e-12):
        raise RuntimeError("Evaluation force logger does not reproduce benchmark physics.")
    return torch.stack(forces, 1), torch.stack(impulses, 1)


def window_average_force(forces, impulses, step_sizes, substeps):
    """Finite exposure/subinterval average in N; never pointwise white force."""
    impulse = forces*step_sizes[None, :, None, None]+impulses
    batch, steps, nodes, axes = impulse.shape
    if steps % substeps:
        raise ValueError("Substep count must divide the recorded solver steps.")
    grouped = impulse.reshape(batch, steps//substeps, substeps, nodes, axes).sum(2)
    duration = step_sizes.reshape(-1, substeps).sum(1)
    return grouped/duration[None, :, None, None]


def field_scores(samples, target, unit):
    lo, hi = torch.quantile(samples, samples.new_tensor([.05, .95]), dim=0)
    return {"rmse_"+unit: float((samples.mean(0)-target).square().mean().sqrt()),
        "crps_"+unit: float(crps(samples, target)), "component_90_coverage": float(((target >= lo)&(target <= hi)).double().mean()),
        "component_90_mean_width_"+unit: float((hi-lo).mean())}


@torch.no_grad()
def predict_force_samples(model, coordinates):
    groups = {key: [] for key in ["x", "v", "rgb", "novel_rgb", "external_force", "stochastic_impulse"]}
    novel_eye, novel_rotation = orbit(model.times, offset_degrees=100)
    for z in coordinates.split(8):
        trajectory, force, impulse = model.force_record(z)
        for key, value in [("x", trajectory.x), ("v", trajectory.v),
            ("rgb", model.renderer(trajectory.x, model.eye, model.rotation)),
            ("novel_rgb", model.renderer(trajectory.x, novel_eye, novel_rotation)),
            ("external_force", force), ("stochastic_impulse", impulse)]:
            groups[key].append(value)
    return {key: torch.cat(value) for key, value in groups.items()}


def run_force_study(plan, output, resume=False):
    from .cli import source_fingerprint
    from .force_report import make_force_report
    validate_plan(plan)
    plan = normalized(plan)
    output = Path(output).resolve(); output.mkdir(parents=True, exist_ok=True)
    fingerprint = source_fingerprint()
    manifest = {"plan": plan, "source_fingerprint_sha256": fingerprint, "environment": _environment()}
    manifest_path = output/"manifest.json"
    if manifest_path.exists():
        if not resume:
            raise FileExistsError("Force study exists; use matching --resume or a new output.")
        if json.loads(manifest_path.read_text()) != manifest:
            raise ValueError("Force plan, source or numerical environment changed.")
    elif resume:
        raise FileNotFoundError("No force study exists to resume.")
    write_json(manifest_path, manifest)
    first = Config(**plan["base"])
    torch.set_num_threads(first.threads)
    diffusion, calibration = train_diffusion(first.diffusion_seed, first.diffusion_steps)
    torch.save(diffusion.state_dict(), output/"diffusion.pt")
    records = []
    for condition in plan["conditions"]:
        pulse, stochastic = CONDITIONS[condition]
        for seed in plan["seeds"]:
            config = Config(**dict(plan["base"], seed=seed, truth_stochastic=stochastic, truth_external_pulse=pulse))
            benchmark = make_benchmark(config)
            for method in plan["methods"]:
                if source_fingerprint() != fingerprint:
                    raise RuntimeError("Source changed during force inference.")
                label = f"{condition}__{seed}__{method}"
                destination = output/"cases"/label
                if resume and (destination/"summary.json").exists():
                    records.append(json.loads((destination/"summary.json").read_text())); continue
                print("Force study: "+label, flush=True)
                start = time.perf_counter()
                has_force, has_sde = METHODS[method]
                model = ForceSceneModel(config, diffusion if has_sde else None, has_force, ForceConfig(**plan["force_prior"]))
                reference, fit_record = fitted_reference(model, benchmark.observations,
                    progress=lambda s: print(s, flush=True), exact_hessian=plan["exact_hessian"])
                settings = SamplerConfig(**dict(plan["sampler"], seed=plan["sampler"]["seed"]+seed))
                correction = GaussianTargetCorrection(model, benchmark.observations, reference)
                def progress(message):
                    if source_fingerprint() != fingerprint:
                        raise RuntimeError("Source changed during force sampling.")
                    print(message, flush=True)
                chain = sample_nonlinear(reference, correction, settings, progress)
                names = list(ParameterPrior.names)
                if has_force:
                    names += [f"force_mode_{j}_{axis}" for j in range(model.force_prior.config.knots) for axis in "xyz"]
                names += [f"white_increment_{j}" for j in range(model.dimension-model.noise_start)]
                diagnostics = diagnose_quantities(torch.cat((chain["coordinates"], chain["log_likelihood"][..., None]), 2),
                                                   names+["observed_log_likelihood"])
                selected = torch.linspace(0, settings.draws-1, plan["evaluation_samples"]//settings.chains).round().long()
                z = chain["coordinates"][:, selected].flatten(0, 1)
                prediction = predict_force_samples(model, z)
                # All force truth is accessed only after the posterior sampling.
                true_force, true_impulse = truth_force_record(config, benchmark)
                total = prediction["external_force"].sum(2)
                true_total = true_force[0].sum(1)
                external_impulse = (prediction["external_force"]*model.step_sizes[None, :, None, None]).sum((1, 2))
                true_external_impulse = (true_force[0]*model.step_sizes[:, None, None]).sum((0, 1))
                averaged = window_average_force(prediction["external_force"], prediction["stochastic_impulse"], model.step_sizes, config.substeps)
                true_averaged = window_average_force(true_force, true_impulse, model.step_sizes, config.substeps)[0]
                free = ~model.mesh.pinned
                force_scores = {"external_total": field_scores(total, true_total, "N"),
                    "external_nodal": field_scores(prediction["external_force"][:, :, free], true_force[0, :, free], "N"),
                    "external_impulse": field_scores(external_impulse, true_external_impulse, "Ns"),
                    "combined_window_nodal": field_scores(averaged[:, :, free], true_averaged[:, free], "N")}
                # Force waveform diagnostics use all retained states, without renders.
                if has_force:
                    flat = chain["coordinates"].flatten(0, 1)
                    all_total = model.force_prior.total_force(flat[:, 8:model.noise_start], model.step_times)
                    total_chain = all_total.reshape(settings.chains, settings.draws, model.nsteps, 3)
                    impulse_chain = (total_chain*model.step_sizes[None, None, :, None]).sum(2)
                    diagnostics_force = diagnose_quantities(impulse_chain, ["external_impulse_x", "external_impulse_y", "external_impulse_z"])
                else:
                    diagnostics_force = {"all_monitored_quantities_pass": False,
                        "reason": "external load is structurally excluded; no force posterior sampled"}
                lo, hi = torch.quantile(total, total.new_tensor([.05, .95]), dim=0)
                centered = total-total.mean(0)
                marginal_covariance = torch.einsum("bti,btj->tij", centered, centered)/(len(z)-1)
                fields = {"times_s": model.step_times.tolist(), "external_total_mean_N": total.mean(0).tolist(),
                    "external_total_q05_N": lo.tolist(), "external_total_q95_N": hi.tolist(),
                    "external_total_marginal_covariance_N2": marginal_covariance.tolist(),
                    "truth_external_total_N_evaluation_only": true_total.tolist(),
                    "external_impulse_mean_Ns": external_impulse.mean(0).tolist(),
                    "external_impulse_covariance_Ns2": torch.cov(external_impulse.T).tolist()}
                phi = model.force_prior.config.total_force_std_n*model.step_features
                oracle = phi@torch.linalg.lstsq(phi, true_total).solution
                record = {"label": label, "condition": condition, "seed": seed, "method": method,
                    "config": asdict(config), "source_fingerprint_sha256": fingerprint,
                    "observations_sha256": hashlib.sha256(benchmark.observations.rgb.numpy().tobytes()).hexdigest(),
                    "latent_dimension": model.dimension, "fit": fit_record, "sampler": asdict(settings),
                    "diagnostics": diagnostics, "force_functional_diagnostics": diagnostics_force,
                    "force_scores": force_scores, "trajectory_scores": evaluate(prediction, benchmark, model.mesh),
                    "force_basis_oracle_rmse_N_evaluation_only": float((oracle-true_total).square().mean().sqrt()),
                    "external_component": "inferred smooth load" if has_force else "excluded by model; conditional point mass at zero",
                    "transition_counts": chain["transition_counts"], "final_beta_per_chain": chain["final_beta_per_chain"],
                    "physics": physics_diagnostics(prediction, model.mesh, model.prior.decode(z)[0]),
                    "evaluation_samples": len(z), "seconds": time.perf_counter()-start}
                record["trajectory_scores"]["definitions"]["uncertainty"] = "nonlinear MCMC, provisional unless convergence checks pass"
                destination.mkdir(parents=True, exist_ok=True)
                torch.save({"chain": chain, "reference_mean": reference.components[0].mean,
                    "reference_precision_cholesky": reference.components[0].precision_cholesky,
                    "observed_rgb": benchmark.observations.rgb, "observed_indices": benchmark.observations.indices,
                    "observation_sigma": benchmark.observations.sigma, "source_fingerprint": fingerprint}, destination/"posterior.pt.tmp")
                (destination/"posterior.pt.tmp").replace(destination/"posterior.pt")
                np.savez_compressed(destination/"force_samples.npz", external_force_N=prediction["external_force"].numpy(),
                    stochastic_impulse_Ns=prediction["stochastic_impulse"].numpy(), window_average_force_N=averaged.numpy(),
                    joint_total_force_covariance_N2=torch.cov(total.flatten(1).T).numpy())
                write_json(destination/"fields.json", fields)
                write_json(destination/"summary.json", record)
                records.append(record)
                print(f"  force RMSE={force_scores['external_total']['rmse_N']:.5f} N; "
                    f"diagnostic pass={diagnostics['all_monitored_quantities_pass']}; {record['seconds']:.1f}s", flush=True)
    summary = {**manifest, "diffusion_calibration": calibration, "cases": records,
        "scope": "known geometry/camera, conditional fixed-model force posteriors; no model probabilities or cross-scene calibration",
        "force_field_scope": "one mass-weighted Gaussian spatial load profile, three directions, projected Matern temporal prior",
        "interpretation": "finite-chain diagnostics must pass before trusting intervals; white-noise component reported as finite-window impulse/force"}
    write_json(output/"summary.json", summary)
    make_force_report(output, summary)
    print(f"Force report: {output/'report.html'}", flush=True)
    return summary
