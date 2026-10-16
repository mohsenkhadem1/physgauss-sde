"""Observation-only nonlinear sampling, followed by isolated truth evaluation."""
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time
import torch

from .benchmark import Config, SceneModel, ParameterPrior, make_benchmark
from .chain_diagnostics import diagnose_quantities
from .likelihood_audit import (verify_source_compatibility, restore_posterior,
    restore_observations, verify_forward_reference, _environment)
from .mcmc import MixtureReference, SamplerConfig, sample_nonlinear
from .metrics import evaluate, physics_diagnostics
from .physics import InvalidDeformation
from .scale_inference import draw_coordinates
from .stochastic import NeuralDiffusion, train_diffusion
from .study import METHODS, predict_mixture, write_json


class NonlinearCorrection:
    def __init__(self, model, observations, posterior):
        self.model, self.observations, self.posterior = model, observations, posterior
        self.models = {c.scale: SceneModel(model.config, model.diffusion, process_scale=c.scale)
                       for c in posterior.components}

    def _values(self, coordinates, scales):
        observed = self.observations
        prediction = coordinates.new_empty((len(coordinates), *observed.rgb.shape))
        invalid = torch.zeros(len(coordinates), dtype=torch.bool)
        for scale in scales.unique().tolist():
            component = self.models[scale]
            indices = torch.nonzero(scales == scale).flatten()
            for selected in indices.split(8):
                try:
                    prediction[selected] = component.predict(coordinates[selected], observed.indices)
                except InvalidDeformation:
                    for i in selected.tolist():
                        try:
                            prediction[i] = component.predict(coordinates[i:i+1], observed.indices)[0]
                        except InvalidDeformation:
                            invalid[i] = True
                            prediction[i] = 0
        if not bool(torch.isfinite(prediction).all()):
            raise RuntimeError("Nonfinite renderer output; diagnose the model before sampling.")
        scaled = coordinates.clone()
        scaled[:, 8:] *= scales[:, None]
        linear = self.posterior.reference_rgb.flatten()[None] + observed.sigma * (
            (scaled-self.posterior.reference[None]) @ self.posterior.jacobian.T)
        residual = (observed.rgb.flatten()[None]-linear)/observed.sigma
        delta = (prediction.flatten(1)-linear)/observed.sigma
        ratio = (residual*delta - .5*delta.square()).sum(1)
        likelihood = -.5*((prediction-observed.rgb[None])/observed.sigma).flatten(1).square().sum(1)
        return torch.where(invalid, -torch.inf, ratio), likelihood, invalid

    @torch.no_grad()
    def __call__(self, coordinates, scales):
        return self._values(coordinates, scales)

    def value_and_grad(self, coordinates, scales):
        with torch.enable_grad():
            variable = coordinates.detach().requires_grad_(True)
            values = self._values(variable, scales)
            safe = torch.where(values[2], 0., values[0])
            gradient, = torch.autograd.grad(safe.sum(), variable)
        if not bool(torch.isfinite(gradient).all()):
            raise RuntimeError("Nonfinite likelihood gradient; sampler stopped.")
        return tuple(v.detach() for v in values), gradient.detach()


def _digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run_nonlinear(study_path, output, source_ref, sampler, cases=None, method="adaptive_sde",
                  evaluation_samples=256, resume=False):
    from .cli import source_fingerprint
    from .nonlinear_report import make_nonlinear_report
    sampler.validate()
    if evaluation_samples < sampler.chains or evaluation_samples % sampler.chains:
        raise ValueError("Evaluation sample count must be a positive multiple of the chain count.")
    if evaluation_samples > sampler.chains*sampler.draws:
        raise ValueError("Evaluation sample count exceeds retained states.")
    if method not in METHODS:
        raise ValueError("Unknown fixed/adaptive method.")
    study_path, output = Path(study_path).resolve(), Path(output).resolve()
    study = json.loads((study_path/"summary.json").read_text())
    provenance = verify_source_compatibility(study, source_ref)
    labels = [f"{r['condition']}__{r['schedule']}__{r['seed']}" for r in study["cases"]]
    if cases is not None and (len(set(cases)) != len(cases) or not set(cases) <= set(labels)):
        raise ValueError("Requested cases must be distinct study labels.")
    selected_cases = [(label, record) for label, record in zip(labels, study["cases"])
                      if cases is None or label in cases]
    if not selected_cases:
        raise ValueError("Select at least one study case.")
    fingerprint = source_fingerprint()
    manifest = {"sampler_source_fingerprint_sha256": fingerprint,
        "study_source_fingerprint_sha256": study["source_fingerprint_sha256"],
        "study_summary_sha256": _digest(study_path/"summary.json"),
        "checkpoint_sha256": {label: _digest(study_path/"cases"/label/"posterior.pt")
                              for label, _ in selected_cases},
        "diffusion_checkpoint_sha256": _digest(study_path/"diffusion.pt") if (study_path/"diffusion.pt").exists() else None,
        "sampler": asdict(sampler), "method": method, "case_labels": [l for l, _ in selected_cases],
        "evaluation_samples": evaluation_samples, **provenance}
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output/"manifest.json"
    if manifest_path.exists():
        if not resume:
            raise FileExistsError("Sampling output exists; use a new output or matching --resume.")
        if json.loads(manifest_path.read_text()) != manifest:
            raise ValueError("Source, observations, checkpoints, or sampling settings changed.")
    elif resume:
        raise FileNotFoundError("No sampling manifest exists to resume.")
    write_json(manifest_path, manifest)
    first = Config(**selected_cases[0][1]["config"])
    torch.set_num_threads(first.threads)
    if (study_path/"diffusion.pt").exists():
        diffusion = NeuralDiffusion()
        diffusion.load_state_dict(torch.load(study_path/"diffusion.pt", weights_only=True))
        for parameter in diffusion.parameters():
            parameter.requires_grad_(False)
    else:
        diffusion, _ = train_diffusion(first.diffusion_seed, first.diffusion_steps)
    records = []
    for label, record in selected_cases:
        if source_fingerprint() != fingerprint:
            raise RuntimeError("Source changed during nonlinear sampling.")
        destination = output/"cases"/label
        if resume and (destination/"summary.json").exists():
            records.append(json.loads((destination/"summary.json").read_text()))
            continue
        print("Sampling " + label, flush=True)
        start = time.perf_counter()
        config = Config(**record["config"])
        state = torch.load(study_path/"cases"/label/"posterior.pt", weights_only=True)
        if state["source_fingerprint"] != study["source_fingerprint_sha256"]:
            raise ValueError("Posterior checkpoint provenance differs from the study.")
        observations, regenerated = restore_observations(state, config, study["environment"])
        posterior = restore_posterior(state, record)
        model = SceneModel(config, diffusion)
        verify_forward_reference(model, observations, posterior)
        reference = MixtureReference(posterior, METHODS[method])
        correction = NonlinearCorrection(model, observations, posterior)
        def progress(message):
            if source_fingerprint() != fingerprint:
                raise RuntimeError("Source changed during a chain; stop rather than mixing implementations.")
            print(message, flush=True)
        chain = sample_nonlinear(reference, correction, sampler, progress)
        # Diagnose every scene/noise coordinate as well as scale and likelihood.
        coordinates = chain["coordinates"]
        names = list(ParameterPrior.names) + [f"white_increment_{i}" for i in range(model.dimension-8)]
        quantities = [coordinates, chain["log_likelihood"][..., None]]
        names += ["observed_log_likelihood"]
        if len(reference.components) > 1:
            quantities.append(chain["scales"][..., None]); names.append("process_scale")
        diagnostics = diagnose_quantities(torch.cat(quantities, 2), names)
        per_chain = evaluation_samples//sampler.chains
        retained_indices = torch.linspace(0, sampler.draws-1, per_chain).round().long()
        evaluated_coordinates = coordinates[:, retained_indices].flatten(0, 1)
        evaluated_scales = chain["scales"][:, retained_indices].flatten()
        # Ground truth is first accessed here, after the sampling result is fixed.
        benchmark = make_benchmark(config)
        if not torch.equal(benchmark.observations.indices, observations.indices) or not torch.equal(benchmark.observations.rgb, observations.rgb):
            raise ValueError("Evaluation benchmark does not match the actual fitting observations.")
        evaluations = {}
        local_z, local_scales, _ = draw_coordinates(posterior, evaluation_samples, sampler.seed+10000, METHODS[method])
        for name, z, scales in [("local_reference", local_z, local_scales),
                                 ("nonlinear_mcmc", evaluated_coordinates, evaluated_scales)]:
            prediction = predict_mixture(model, z, scales)
            scores = evaluate(prediction, benchmark, model.mesh)
            scores["definitions"]["uncertainty"] = ("local Gaussian reference" if name == "local_reference"
                else "finite correlated nonlinear MCMC; reliability is conditional on diagnostics")
            young, _, _, _ = model.prior.decode(z)
            evaluations[name] = {"scores": scores, "physics": physics_diagnostics(prediction, model.mesh, young)}
        counts = [{str(s): int((row == s).sum()) for s in reference.scales.tolist()}
                  for row in chain["scales"]]
        result = {"label": label, "condition": record["condition"], "schedule": record["schedule"],
            "realization_seed": record["seed"], "sampler": asdict(sampler),
            "sampler_source_fingerprint_sha256": fingerprint,
            "observations_sha256": hashlib.sha256(observations.rgb.numpy().tobytes()).hexdigest(),
            "legacy_observations_regenerated": regenerated, "diagnostics": diagnostics,
            "scale_counts_per_chain": counts, "final_beta_per_chain": chain["final_beta_per_chain"],
            "transition_counts": chain["transition_counts"], "evaluation_samples": evaluation_samples,
            "evaluation_subsampling": "equal evenly spaced retained states per chain for rendering cost only; diagnostics use every retained state",
            "evaluations": evaluations, "seconds": time.perf_counter()-start}
        destination.mkdir(parents=True, exist_ok=True)
        torch.save(chain, destination/"chains.pt.tmp")
        (destination/"chains.pt.tmp").replace(destination/"chains.pt")
        write_json(destination/"summary.json", result)
        records.append(result)
        print(f"  diagnostic pass={diagnostics['all_monitored_quantities_pass']}; "
              f"{len(diagnostics['failed_quantities'])} flagged quantities, {result['seconds']:.1f}s", flush=True)
    summary = {**manifest, "environment": _environment(), "cases": records,
        "interpretation": "Metropolis targets the nonlinear model; finite runs still require convergence checks. No real-world or cross-scene calibration claim."}
    write_json(output/"summary.json", summary)
    make_nonlinear_report(output, summary)
    print(f"Sampling report: {output/'report.html'}", flush=True)
    return summary
