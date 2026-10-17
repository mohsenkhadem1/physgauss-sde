"""Commands: physgauss check; physgauss demo; physgauss study."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
import time
import unittest

import numpy as np
import torch
from .benchmark import Config, SceneModel, make_benchmark
from .inference import Fit, fit, predict_samples, sample_coordinates
from .metrics import evaluate, physics_diagnostics
from .report import make_report
from .stochastic import NeuralDiffusion, train_diffusion


def source_fingerprint():
    root = Path(__file__).resolve().parents[2]
    digest = hashlib.sha256()
    for path in sorted((root / "src").rglob("*.py")):
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(path.read_bytes().replace(b"\r\n", b"\n"))
    return digest.hexdigest()


def run_demo(config: Config, output: Path, resume=False):
    config.validate()
    torch.set_num_threads(config.threads)
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    print("Creating isolated synthetic observations...", flush=True)
    benchmark = make_benchmark(config)
    fingerprint = source_fingerprint()
    record_path = output / "checkpoint.pt"
    checkpoint = None
    if resume:
        if not record_path.exists():
            raise FileNotFoundError("No checkpoint exists for --resume.")
        checkpoint = torch.load(record_path, weights_only=True)
        if checkpoint["config"] != asdict(config):
            raise ValueError("Checkpoint configuration differs; use a new output directory.")
        if checkpoint["source_fingerprint"] != fingerprint:
            raise ValueError("Source changed since checkpoint; rerun fitting in a new output directory.")
        diffusion = NeuralDiffusion()
        diffusion.load_state_dict(checkpoint["diffusion_weights"])
        for p in diffusion.parameters():
            p.requires_grad_(False)
        calibration = checkpoint["diffusion_calibration"]
    else:
        print("Calibrating neural diffusion on independent synthetic innovations...", flush=True)
        diffusion, calibration = train_diffusion(config.diffusion_seed, config.diffusion_steps)
    results = {}
    saved = {"config": asdict(config), "source_fingerprint": fingerprint,
             "diffusion_weights": diffusion.state_dict(), "diffusion_calibration": calibration,
             "fits": {}}
    summary = {
        "scope": "Synthetic known-rest-geometry, known-camera offline smoothing proof of concept",
        "posterior": "Joint standardized scene parameters and Brownian increments; local Gauss-Newton Laplace",
        "config": asdict(config), "source_fingerprint_sha256": fingerprint,
        "environment": {"python": platform.python_version(), "torch": str(torch.__version__),
                        "numpy": np.__version__, "platform": platform.platform(), "device": "cpu"},
        "diffusion_calibration": calibration,
        "truth_parameters_evaluation_only": benchmark.truth_parameters,
        "methods": {},
    }
    for method, stochastic in [("physical", False), ("physical_sde", True)]:
        print(f"Fitting {method}...", flush=True)
        model = SceneModel(config, diffusion if stochastic else None)
        if checkpoint and method in checkpoint["fits"]:
            fitted = Fit(**checkpoint["fits"][method])
        else:
            fitted = fit(model, benchmark.observations, progress=lambda s: print(s, flush=True))
        coordinates = sample_coordinates(fitted, config.posterior_samples, config.seed + 701)
        print(f"  simulating {len(coordinates)} physical posterior samples", flush=True)
        samples = predict_samples(model, coordinates)
        scores = evaluate(samples, benchmark, model.mesh)
        young, drag, _, _ = model.prior.decode(coordinates)
        diagnostics = physics_diagnostics(samples, model.mesh, young)
        results[method] = {"model": model, "fit": fitted, "samples": samples,
                           "metrics": scores, "coordinates": coordinates}
        summary["methods"][method] = {
            "latent_dimension": model.dimension, "fit_seconds": fitted.elapsed_seconds,
            "map_objective": fitted.objective, "map_gradient_norm": fitted.gradient_norm,
            "map_gauss_newton_decrement": fitted.newton_decrement,
            "posterior_samples": len(coordinates), "failed_samples": 0,
            "young_pa_q05_q50_q95": torch.quantile(young, young.new_tensor([.05, .5, .95])).tolist(),
            "drag_q05_q50_q95": torch.quantile(drag, drag.new_tensor([.05, .5, .95])).tolist(),
            "scores": scores, "physics": diagnostics,
        }
        saved["fits"][method] = {"mean": fitted.mean, "precision_cholesky": fitted.precision_cholesky,
                                 "history": fitted.history, "elapsed_seconds": fitted.elapsed_seconds,
                                 "objective": fitted.objective, "gradient_norm": fitted.gradient_norm,
                                 "newton_decrement": fitted.newton_decrement}
        # Save partial progress after each fit; --resume can finish the other fit.
        torch.save(saved, record_path)
        gap = scores["long_gap"]
        print(f"  gap RMSE={gap['position_rmse_m']*1000:.3f} mm; "
              f"coverage={gap['component_90_coverage']:.1%}; "
              f"width={gap['component_90_mean_width_m']*1000:.3f} mm", flush=True)
    summary["run_seconds_before_report"] = time.perf_counter() - start
    (output / "config.json").write_text(json.dumps(asdict(config), indent=2), encoding="utf-8")
    (output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    np.savez_compressed(output / "observations.npz", indices=benchmark.observations.indices.numpy(),
                        rgb=benchmark.observations.rgb.numpy(), times=results["physical"]["model"].times.numpy())
    np.savez_compressed(output / "posterior.npz",
                        **{f"{name}_positions": r["samples"]["x"].numpy() for name, r in results.items()})
    print("Rendering experiment report...", flush=True)
    make_report(output, benchmark, results, summary)
    print(f"Report: {output / 'report.html'}", flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("check", help="Run physical and numerical regression checks.")
    demo = sub.add_parser("demo", help="Run controlled monocular missing-frame experiment.")
    demo.add_argument("--config", type=Path)
    demo.add_argument("--output", type=Path, default=Path("runs/cpu-demo"))
    demo.add_argument("--quick", action="store_true", help="Small smoke test, not a research result.")
    demo.add_argument("--resume", action="store_true", help="Resume matching fits or rebuild samples/report from a checkpoint.")
    study = sub.add_parser("study", help="Matched noise-scale inference over seeds and observation schedules.")
    study.add_argument("--plan", type=Path)
    study.add_argument("--output", type=Path, default=Path("runs/scale-study"))
    study.add_argument("--quick", action="store_true", help="Two-case smoke study, not a calibration result.")
    study.add_argument("--resume", action="store_true")
    audit = sub.add_parser("audit", help="Compare local posteriors with the full nonlinear observed-image likelihood.")
    audit.add_argument("--study", type=Path, required=True)
    audit.add_argument("--source-ref", required=True, help="Committed source ref whose fingerprint matches the study.")
    audit.add_argument("--output", type=Path, default=Path("runs/likelihood-audit"))
    audit.add_argument("--samples", type=int, default=256)
    audit.add_argument("--seed", type=int, default=55001)
    audit.add_argument("--resume", action="store_true")
    nonlinear = sub.add_parser("sample", help="Metropolis sampling with the full nonlinear observed-image likelihood.")
    nonlinear.add_argument("--study", type=Path, required=True)
    nonlinear.add_argument("--source-ref", required=True)
    nonlinear.add_argument("--output", type=Path, default=Path("runs/nonlinear-sampling"))
    nonlinear.add_argument("--case", action="append", help="Exact condition__schedule__seed label; repeat to select cases.")
    nonlinear.add_argument("--method", choices=["physical", "fixed_sde", "adaptive_sde"], default="adaptive_sde")
    nonlinear.add_argument("--chains", type=int, default=4)
    nonlinear.add_argument("--warmup", type=int, default=1000)
    nonlinear.add_argument("--draws", type=int, default=2000)
    nonlinear.add_argument("--evaluation-samples", type=int, default=256)
    nonlinear.add_argument("--seed", type=int, default=67001)
    nonlinear.add_argument("--langevin", action="store_true", help="Use gradient-guided pCNL instead of pCN proposals.")
    nonlinear.add_argument("--resume", action="store_true", help="Resume at completed-case boundaries with unchanged settings.")
    forces = sub.add_parser("forces", help="Infer structured external force distributions and compare SDE controls.")
    forces.add_argument("--plan", type=Path)
    forces.add_argument("--quick", action="store_true")
    forces.add_argument("--output", type=Path, default=Path("runs/force-study"))
    forces.add_argument("--resume", action="store_true")
    marginal = sub.add_parser("marginalize", help="Compare alternating, joint and analytically marginalized force inference.")
    marginal.add_argument("--plan", type=Path)
    marginal.add_argument("--quick", action="store_true")
    marginal.add_argument("--output", type=Path, default=Path("runs/marginal-force-study"))
    marginal.add_argument("--historical-summary", type=Path, help="Optional v0.4 summary with exactly matching observations/configs.")
    marginal.add_argument("--resume", action="store_true")
    accurate = sub.add_parser("accurate", help="Compare corrected dynamics/visibility on independent RK4 reference inputs.")
    accurate.add_argument("--plan", type=Path)
    accurate.add_argument("--quick", action="store_true")
    accurate.add_argument("--output", type=Path, default=Path("runs/accurate-reference"))
    accurate.add_argument("--reference-root", type=Path, default=Path("examples/simulation-view-audit/high_accuracy_reference"))
    accurate.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.command == "check":
        tests = Path(__file__).resolve().parents[2] / "tests"
        if not tests.exists():
            parser.error("Run check from an editable installation or repository checkout containing tests/.")
        result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.discover(str(tests)))
        raise SystemExit(0 if result.wasSuccessful() else 1)
    if args.command == "audit":
        from .likelihood_audit import audit_study
        audit_study(args.study, args.output, args.source_ref, args.samples, args.seed, args.resume)
        return
    if args.command == "sample":
        from .mcmc import SamplerConfig
        from .nonlinear_study import run_nonlinear
        settings = SamplerConfig(chains=args.chains, warmup=args.warmup, draws=args.draws,
                                  seed=args.seed, langevin=args.langevin)
        run_nonlinear(args.study, args.output, args.source_ref, settings, args.case, args.method,
                      args.evaluation_samples, args.resume)
        return
    if args.command == "forces":
        from .force_study import quick_force_plan, run_force_study
        if args.quick and args.plan:
            parser.error("Choose --quick or --plan, not both.")
        plan = quick_force_plan() if args.quick else json.loads(
            (args.plan or Path("configs/force_study.json")).read_text(encoding="utf-8"))
        run_force_study(plan, args.output, args.resume)
        return
    if args.command == "marginalize":
        from .marginal_study import quick_marginal_plan, run_marginal_study
        if args.quick and args.plan:
            parser.error("Choose --quick or --plan, not both.")
        plan = quick_marginal_plan() if args.quick else json.loads(
            (args.plan or Path("configs/marginal_force_study.json")).read_text(encoding="utf-8"))
        run_marginal_study(plan, args.output, args.resume, args.historical_summary)
        return
    if args.command == "accurate":
        from .accurate_study import default_plan, run_accurate_study
        if args.quick and args.plan:
            parser.error("Choose --quick or --plan, not both.")
        plan = default_plan(args.quick) if args.plan is None else json.loads(args.plan.read_text(encoding="utf-8"))
        run_accurate_study(plan, args.output, args.reference_root, args.resume)
        return
    if args.command == "study":
        from .study import quick_plan, run_study
        if args.quick and args.plan:
            parser.error("Choose --quick or --plan, not both.")
        plan = quick_plan() if args.quick else json.loads(
            (args.plan or Path("configs/scale_study.json")).read_text(encoding="utf-8"))
        run_study(plan, args.output, args.resume)
        return
    if args.quick and args.config:
        parser.error("Choose --quick or --config, not both.")
    if args.quick:
        config = Config.quick()
    elif args.config:
        config = Config(**json.loads(args.config.read_text(encoding="utf-8")))
    else:
        config = Config()
    run_demo(config, args.output, args.resume)


if __name__ == "__main__":
    main()
