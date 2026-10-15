"""Audit local posterior proposals against the full nonlinear RGB likelihood.

The proposal is the exact posterior of the declared linearized model. Because
its priors are identical to the nonlinear target, the importance log ratio is
log L_nonlinear - log L_linear. Low ESS is a warning, not a corrected posterior
or a calibration claim. High finite-sample ESS cannot exclude missed modes.
"""
import hashlib
import json
import math
from pathlib import Path
import platform
import subprocess

import numpy as np
import torch

from .benchmark import Config, Observations, SceneModel, make_benchmark
from .physics import InvalidDeformation
from .scale_inference import LinearComponent, ScalePosterior, draw_coordinates
from .stochastic import NeuralDiffusion, train_diffusion
from .study import METHODS, normalized, write_json


CRITICAL_MODULES = ("benchmark.py", "camera.py", "physics.py", "render.py",
                    "stochastic.py", "inference.py", "scale_inference.py")


def likelihood_log_ratio(actual, linear, observed, sigma):
    if sigma <= 0 or not math.isfinite(sigma):
        raise ValueError("Observation sigma must be finite and positive.")
    actual_residual = ((actual - observed[None]) / sigma).flatten(1)
    linear_residual = ((linear - observed[None]) / sigma).flatten(1)
    return -.5 * (actual_residual.square().sum(1) - linear_residual.square().sum(1))


def importance_diagnostics(log_ratios):
    if log_ratios.ndim != 1 or len(log_ratios) < 2:
        raise ValueError("Need at least two independent proposal log ratios.")
    if bool(torch.isnan(log_ratios).any()) or bool(torch.isposinf(log_ratios).any()):
        raise ValueError("Log ratios must be finite or negative infinity.")
    finite = torch.isfinite(log_ratios)
    if not bool(finite.any()):
        raise RuntimeError("No proposal has finite target likelihood.")
    weights = torch.softmax(log_ratios, dim=0)
    ess = float(1 / weights.square().sum())
    positive = weights > 0
    return {
        "samples": len(weights), "importance_ess": ess, "relative_ess": ess / len(weights),
        "maximum_normalized_weight": float(weights.max()),
        "weight_entropy_nats": float(-(weights[positive] * weights[positive].log()).sum()),
        "finite_log_ratio_min": float(log_ratios[finite].min()),
        "finite_log_ratio_max": float(log_ratios[finite].max()),
        "zero_target_weight_count": int((~finite).sum()),
        "log_mean_likelihood_ratio": float(torch.logsumexp(log_ratios, 0) - math.log(len(weights))),
        "low_relative_ess_warning": ess / len(weights) < .1,
        "warning_definition": "relative ESS < 0.1 is a predeclared diagnostic flag, not a calibration test",
    }


def _git(root, *args):
    result = subprocess.run(["git", "-C", str(root), *args], capture_output=True, check=False)
    if result.returncode:
        raise RuntimeError("Git provenance lookup failed; use a local source commit matching the study.")
    return result.stdout


def verify_source_compatibility(study_summary, source_ref):
    root = Path(__file__).resolve().parents[2]
    commit = _git(root, "rev-parse", "--verify", "--end-of-options", source_ref + "^{commit}").decode().strip()
    paths = _git(root, "ls-tree", "-r", "--name-only", commit, "src").decode().splitlines()
    paths = sorted(path for path in paths if path.endswith(".py"))
    digest = hashlib.sha256()
    contents = {}
    for path in paths:
        content = _git(root, "show", commit + ":" + path).replace(b"\r\n", b"\n")
        digest.update(path.encode())
        digest.update(content)
        contents[path] = content
    if digest.hexdigest() != study_summary["source_fingerprint_sha256"]:
        raise ValueError("Source ref does not reproduce the study's recorded full source fingerprint.")
    critical_hashes = {}
    for name in CRITICAL_MODULES:
        path = "src/physgauss/" + name
        current = (root / path).read_bytes().replace(b"\r\n", b"\n")
        if contents.get(path) != current:
            raise ValueError(f"Model module {name} changed; do not audit the old posterior with different physics.")
        critical_hashes[path] = hashlib.sha256(current).hexdigest()
    return {"study_source_commit": commit, "verified_model_module_sha256": critical_hashes}


def restore_posterior(state, record):
    scale = record["scale_posterior"]
    components = [LinearComponent(alpha, state["means"][i], state["precision_cholesky"][i],
        scale["component_negative_log_evidence_up_to_common_constant"][i])
        for i, alpha in enumerate(scale["grid"])]
    return ScalePosterior(components, torch.tensor(scale["prior_weights"], dtype=torch.float64),
        state["weights"], state["reference"], state["reference_rgb"], state["jacobian"])


def _environment():
    return {"python": platform.python_version(), "torch": str(torch.__version__),
            "numpy": np.__version__, "platform": platform.platform()}


def restore_observations(state, config, study_environment):
    if "observed_rgb" in state:
        return Observations(state["observed_indices"], state["observed_rgb"], state["observation_sigma"]), False
    # Historical milestone-2 checkpoints predate persisted observed images.
    # Never regenerate these under a different numerical environment.
    if study_environment != _environment():
        raise ValueError("Legacy observations are absent and the environment differs. Regenerate the study with persisted observations.")
    return make_benchmark(config).observations, True


def verify_forward_reference(model, observations, posterior):
    with torch.no_grad():
        current = model.predict(posterior.reference[None], observations.indices)[0]
        if not torch.allclose(current, posterior.reference_rgb, atol=1e-10, rtol=1e-9):
            raise ValueError("The nonlinear reference render differs from the stored study.")
        # Probe noise columns as well: deterministic renders alone do not check
        # the orientation of a numerically computed stochastic spatial basis.
        cols = [8 + i for i in range(model.config.noise_rank)]
        h = .001
        shifts = torch.zeros(len(cols), model.dimension, dtype=posterior.reference.dtype)
        shifts[torch.arange(len(cols)), torch.tensor(cols)] = h
        points = torch.cat((posterior.reference[None] + shifts, posterior.reference[None] - shifts))
        rendered = model.predict(points, observations.indices).flatten(1)
        derivative = ((rendered[:len(cols)] - rendered[len(cols):]) / (2*h*observations.sigma)).T
        if not torch.allclose(derivative, posterior.jacobian[:, cols], atol=1e-7, rtol=1e-5):
            raise ValueError("Stochastic reference derivatives changed; numerical basis compatibility is unverified.")


def nonlinear_observation_prediction(model, observations, coordinates, scales):
    rgb = coordinates.new_empty((len(coordinates), *observations.rgb.shape))
    invalid = torch.zeros(len(coordinates), dtype=torch.bool)
    with torch.no_grad():
        for scale in scales.unique(sorted=True):
            component = SceneModel(model.config, model.diffusion, process_scale=float(scale))
            indices = torch.nonzero(scales == scale).flatten()
            for selected in indices.split(8):
                try:
                    rgb[selected] = component.predict(coordinates[selected], observations.indices)
                except InvalidDeformation:
                    # An invalid proposal has zero target density. Count it and
                    # keep its zero weight; do not quietly replace the draw.
                    for index in selected.tolist():
                        try:
                            rgb[index] = component.predict(coordinates[index:index+1], observations.indices)[0]
                        except InvalidDeformation:
                            invalid[index] = True
                            rgb[index] = 0
    return rgb, invalid


def audit_study(study_path, output, source_ref, samples=256, seed=55001, resume=False):
    from .cli import source_fingerprint
    from .audit_report import make_audit_report
    if samples < 16:
        raise ValueError("Use at least 16 independent proposals per diagnostic.")
    study_path, output = Path(study_path).resolve(), Path(output).resolve()
    study = json.loads((study_path / "summary.json").read_text())
    provenance = verify_source_compatibility(study, source_ref)
    fingerprint = source_fingerprint()
    manifest = {"audit_source_fingerprint_sha256": fingerprint,
                "study_source_fingerprint_sha256": study["source_fingerprint_sha256"],
                "study_summary_sha256": hashlib.sha256((study_path / "summary.json").read_bytes()).hexdigest(),
                "samples_per_method_case": samples, "seed": seed, **provenance}
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "manifest.json"
    if manifest_path.exists():
        if not resume:
            raise FileExistsError("Audit exists. Use --resume or a new output directory.")
        if json.loads(manifest_path.read_text()) != manifest:
            raise ValueError("Audit source, study, or sampling plan changed.")
    elif resume:
        raise FileNotFoundError("No audit manifest exists to resume.")
    write_json(manifest_path, manifest)
    first = Config(**study["cases"][0]["config"])
    torch.set_num_threads(first.threads)
    if (study_path / "diffusion.pt").exists():
        diffusion = NeuralDiffusion()
        diffusion.load_state_dict(torch.load(study_path / "diffusion.pt", weights_only=True))
        for parameter in diffusion.parameters():
            parameter.requires_grad_(False)
    else:
        diffusion, _ = train_diffusion(first.diffusion_seed, first.diffusion_steps)
    records = []
    for case_index, record in enumerate(study["cases"]):
        if source_fingerprint() != fingerprint:
            raise RuntimeError("Audit source changed during the run.")
        label = f"{record['condition']}__{record['schedule']}__{record['seed']}"
        print(f"Audit [{case_index+1}/{len(study['cases'])}] {label}", flush=True)
        destination = output / "cases" / (label + ".json")
        if resume and destination.exists():
            records.append(json.loads(destination.read_text()))
            continue
        config = Config(**record["config"])
        state = torch.load(study_path / "cases" / label / "posterior.pt", weights_only=True)
        if state["source_fingerprint"] != study["source_fingerprint_sha256"]:
            raise ValueError("Posterior checkpoint has different source provenance.")
        observations, regenerated = restore_observations(state, config, study["environment"])
        model = SceneModel(config, diffusion)
        posterior = restore_posterior(state, record)
        verify_forward_reference(model, observations, posterior)
        result = {"condition": record["condition"], "schedule": record["schedule"],
                  "seed": record["seed"], "legacy_observations_regenerated_with_matching_environment": regenerated,
                  "observations_sha256": hashlib.sha256(observations.rgb.numpy().tobytes()).hexdigest(),
                  "methods": {}}
        for method, fixed_scale in METHODS.items():
            coordinates, scales, _ = draw_coordinates(posterior, samples, seed + case_index, fixed_scale)
            actual, invalid = nonlinear_observation_prediction(model, observations, coordinates, scales)
            with torch.no_grad():
                scaled = coordinates.clone()
                scaled[:, 8:] *= scales[:, None]
                linear = posterior.reference_rgb.flatten()[None] + observations.sigma * (
                    (scaled - posterior.reference[None]) @ posterior.jacobian.T)
                ratios = likelihood_log_ratio(actual.flatten(1), linear,
                                               observations.rgb.flatten(), observations.sigma)
                ratios[invalid] = -torch.inf
            diagnostics = importance_diagnostics(ratios)
            diagnostics["invalid_physical_proposals"] = int(invalid.sum())
            result["methods"][method] = diagnostics
            print(f"  {method}: relative ESS={diagnostics['relative_ess']:.3f}, "
                  f"max weight={diagnostics['maximum_normalized_weight']:.3f}", flush=True)
        destination.parent.mkdir(parents=True, exist_ok=True)
        write_json(destination, result)
        records.append(result)
    summary = {**manifest, "environment": _environment(), "case_count": len(records),
        "interpretation": "importance-overlap diagnostic only; no corrected-posterior or calibration claim; high ESS cannot rule out unseen modes",
        "predeclared_warning_threshold_relative_ess": .1, "cases": records}
    write_json(output / "summary.json", summary)
    make_audit_report(output, summary)
    print(f"Audit report: {output / 'report.html'}", flush=True)
    return summary
