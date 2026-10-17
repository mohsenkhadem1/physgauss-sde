"""Validate and publish compact evidence, exact input images and fitted paths.

Raw Gaussian precisions and paired draws stay in ignored runs/. This script
exports enough to inspect point reconstructions and query force distributions;
full scene-posterior generation requires raw checkpoints or a fresh run.
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import torch
from physgauss.benchmark import Config
from physgauss.forces import ForceConfig, evaluate_force_posterior
from physgauss.likelihood_audit import verify_source_compatibility
from physgauss.marginal_forces import MarginalConfig, TrajectoryForceModel
from physgauss.marginal_inference import recover_force_moments
from physgauss.marginal_report import make_marginal_report
from physgauss.physics import material_response
from physgauss.study import write_json


def digest(path):
    return hashlib.sha256(Path(path).read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def verify_marginal_source(summary, source_ref):
    root = Path(__file__).resolve().parents[1]
    provenance = verify_source_compatibility(summary, source_ref)
    commit = provenance["study_source_commit"]
    # Comments/units documentation may be corrected after numerical execution;
    # executable numerical ASTs must still be identical to the recorded commit.
    modules = ("forces.py", "marginal_forces.py", "marginal_inference.py", "marginal_study.py")
    for module in modules:
        relative = "src/physgauss/"+module
        old = subprocess.run(["git", "-C", str(root), "show", commit+":"+relative],
                             capture_output=True, check=True).stdout.decode("utf-8")
        current = (root/relative).read_text(encoding="utf-8")
        if ast.dump(ast.parse(old)) != ast.dump(ast.parse(current)):
            raise ValueError("Numerical implementation changed: "+module)
    provenance["verified_additional_numerical_AST_modules"] = list(modules)
    return provenance


def publish(raw, destination, source_ref):
    raw, destination = Path(raw).resolve(), Path(destination).resolve()
    if destination.exists():
        raise FileExistsError("Choose a new publication directory; existing evidence is never overwritten.")
    summary = json.loads((raw/"summary.json").read_text())
    manifest = json.loads((raw/"manifest.json").read_text())
    for key, value in manifest.items():
        if summary[key] != value:
            raise ValueError("Manifest/summary mismatch: "+key)
    provenance = verify_marginal_source(summary, source_ref)
    expected = {(c, s, m) for c in summary["plan"]["conditions"] for s in summary["plan"]["seeds"] for m in summary["plan"]["methods"]}
    actual = [(r["condition"], r["seed"], r["method"]) for r in summary["cases"]]
    if len(actual) != len(expected) or set(actual) != expected:
        raise ValueError("Study has missing/duplicate cases.")
    destination.mkdir(parents=True)
    shutil.copyfile(raw/"summary.json", destination/"summary.json")
    shutil.copyfile(raw/"manifest.json", destination/"manifest.json")
    verified = []
    torch.set_num_threads(Config(**summary["plan"]["base"]).threads)
    seen_inputs = {}
    for r in summary["cases"]:
        pair = f"{r['condition']}__{r['seed']}"
        case = raw/"cases"/pair/r["method"]
        if json.loads((case/"summary.json").read_text()) != r:
            raise ValueError("Individual/aggregate record mismatch.")
        model = TrajectoryForceModel(Config(**r["config"]), ForceConfig(**summary["plan"]["force_prior"]),
                                     MarginalConfig(**summary["plan"]["trajectory_model"]))
        warm = torch.load(raw/"cases"/pair/"initialization.pt", weights_only=True)
        observed_hash = hashlib.sha256(warm["observed_rgb"].numpy().tobytes()).hexdigest()
        initial_hash = hashlib.sha256(warm["initial"].numpy().tobytes()).hexdigest()
        if observed_hash != r["observations_sha256"] or initial_hash != r["initial_state_sha256"]:
            raise ValueError("Input/initialization hash mismatch.")
        identity = (observed_hash, initial_hash)
        if pair in seen_inputs and seen_inputs[pair] != identity:
            raise ValueError("Solvers did not use identical images/initialization.")
        seen_inputs[pair] = identity
        if warm["source_fingerprint"] != summary["source_fingerprint_sha256"]:
            raise ValueError("Initialization source mismatch.")
        fitted = torch.load(case/"fit.pt", weights_only=True)
        if fitted["record"] != r["fit"]:
            raise ValueError("Fit checkpoint/record mismatch.")
        draws = torch.load(case/"draws.pt", weights_only=True)
        valid = draws["coordinates"][draws["valid_indices"]]
        fields = json.loads((case/"fields.json").read_text())
        with torch.no_grad():
            residual = model.residual(valid)
            moments = recover_force_moments(model.elimination, residual)
            torch.testing.assert_close(torch.tensor(fields["whitened_force_coefficient_mean"], dtype=torch.float64), moments["mean"], rtol=1e-12, atol=1e-12)
            torch.testing.assert_close(torch.tensor(fields["whitened_force_coefficient_covariance"], dtype=torch.float64), moments["covariance"], rtol=1e-12, atol=1e-12)
            queried = evaluate_force_posterior(fields, model.step_times)
            torch.testing.assert_close(queried["mean_total_N"], torch.tensor(fields["external_total_mean_N"], dtype=torch.float64), rtol=1e-12, atol=1e-12)
            torch.testing.assert_close(queried["total_draws_N"], model.force_prior.total_force(
                draws["force_coefficients_paired_with_valid_coordinates"], model.step_times), rtol=1e-12, atol=1e-12)
            covariance = queried["joint_total_covariance_N2"]
            torch.testing.assert_close(covariance, covariance.T, rtol=1e-12, atol=1e-12)
            if float(torch.linalg.eigvalsh(covariance).min()) < -1e-10:
                raise ValueError("Force covariance is not positive semidefinite.")
            full = model.step_trajectory(fitted["mean"][None])
            young, drag, offset, velocity = model.prior.decode(fitted["mean"][None])
            _, _, jac = material_response(full.x, model.mesh, young[:, None])
        target = destination/"cases"/pair/r["method"]
        target.mkdir(parents=True)
        shutil.copyfile(case/"fields.json", target/"fields.json")
        shutil.copyfile(case/"summary.json", target/"summary.json")
        point = {"config": r["config"], "trajectory_model": summary["plan"]["trajectory_model"],
            "force_prior": summary["plan"]["force_prior"], "state_coordinates": fitted["mean"].tolist(),
            "coordinate_definition": "8 standardized scene coordinates then all solver-step free-node velocities divided by velocity_coordinate_scale; node order is mesh order",
            "young_pa": float(young[0]), "drag_per_second": float(drag[0]),
            "initial_offset_m": offset[0].tolist(), "initial_velocity_m_per_second": velocity[0].tolist(),
            "conditional_force_coefficient_mean": fitted["conditional_force_mean"].tolist(),
            "minimum_solver_step_det_F": float(jac.min()),
            "source_fingerprint_sha256": r["source_fingerprint_sha256"],
            "qualification": "point estimate only; not a joint posterior sample"}
        write_json(target/"point_estimate.json", point)
        input_path = destination/"cases"/pair/"observations.json"
        if not input_path.exists():
            write_json(input_path, {"config": r["config"], "indices": warm["observed_indices"].tolist(),
                "rgb": warm["observed_rgb"].tolist(), "sigma": warm["observation_sigma"],
                "observations_sha256_float64_tensor_bytes": observed_hash,
                "initial_state_coordinates": warm["initial"].tolist(),
                "forward_warm_start_coordinates": warm["forward_coordinates"].tolist(),
                "source_fingerprint_sha256": warm["source_fingerprint"]})
        verified.append({"condition": r["condition"], "seed": r["seed"], "method": r["method"],
                         "raw_fit_sha256": hashlib.sha256((case/"fit.pt").read_bytes()).hexdigest(),
                         "raw_draws_sha256": hashlib.sha256((case/"draws.pt").read_bytes()).hexdigest(),
                         "verified_force_covariance_and_paired_draws": True})
    make_marginal_report(destination, summary)
    from physgauss.cli import source_fingerprint
    provenance.update({"numerical_source_fingerprint_sha256": summary["source_fingerprint_sha256"],
        "publication_source_fingerprint_sha256": source_fingerprint(), "cases": verified,
        "publication_note": "Numerical records retained unchanged; report regenerated with current labels. Numeric AST compatibility verified. Raw checkpoints remain local and reproducible via the fixed plan.",
        "files_sha256_normalized_lf": {p.relative_to(destination).as_posix(): digest(p)
            for p in sorted(destination.rglob("*")) if p.is_file()}})
    write_json(destination/"provenance.json", provenance)
    print(f"Verified and published {len(verified)} fits to {destination}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-ref", required=True)
    args = parser.parse_args()
    publish(args.raw, args.output, args.source_ref)
