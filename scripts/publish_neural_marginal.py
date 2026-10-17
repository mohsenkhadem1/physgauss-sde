"""Verify and publish all 36 sparse neural-SDE/force-marginalization fits."""
import argparse
import base64
import csv
import hashlib
import html
import json
from pathlib import Path
import shutil
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from physgauss.cli import source_fingerprint
from physgauss.neural_marginal_study import default_plan, construct, calibrate, aggregate, CONDITIONS, MODES
from physgauss.accurate_study import digest_tensor
from physgauss.benchmark import Observations
from physgauss.forces import evaluate_force_posterior
from physgauss.physics import edge_features
from physgauss.study import write_json
from publish_accurate_study import canonical_hash, source_commit, savefig, table


ROOT = Path(__file__).resolve().parents[1]
COLORS = {"floor": "#788b99", "constant": "#c9782a", "neural": "#007c91"}
LABELS = {"floor": "Current floor", "constant": "Constant internal", "neural": "Neural internal"}
TITLES = {"no_load": "Deterministic / no push", "external_pulse": "Deterministic / push",
          "stochastic": "Stochastic / no push", "pulse_and_stochastic": "Stochastic / push"}


def lookup(summary, condition, mode, seed=None):
    rows = summary["aggregates"] if seed is None else summary["cases"]
    return next(r for r in rows if r["condition"] == condition and r["mode"] == mode and (seed is None or r["seed"] == seed))


@torch.no_grad()
def verify(raw, summary):
    plan = summary["plan"]
    if plan != default_plan() or summary["source_fingerprint_sha256"] != source_fingerprint():
        raise ValueError("Need the declared full plan and unchanged inference source.")
    expected = {f"{c}__{s}__{m}" for c in plan["conditions"] for s in plan["seeds"] for m in plan["modes"]}
    if len(summary["cases"]) != 36 or {r["case"] for r in summary["cases"]} != expected:
        raise ValueError("Incomplete/duplicated model comparison.")
    if aggregate(summary["cases"]) != summary["aggregates"]:
        raise ValueError("Aggregate measurements differ from individual seeds.")
    for condition, expected_hash in summary["deterministic_reference_hashes"].items():
        source = ROOT/"examples/simulation-view-audit/high_accuracy_reference"/f"{condition}.json"
        if canonical_hash(source) != expected_hash:
            raise ValueError("Original deterministic reference changed.")
    network, calibration = calibrate(plan, raw/"calibration.json")
    if hashlib.sha256(json.dumps(calibration, sort_keys=True).encode()).hexdigest() != summary["calibration_sha256_canonical_json"]:
        raise ValueError("Calibration weights/statistics changed.")
    common = construct(plan, network, calibration, "floor")
    initial = common.rollout(common.times.new_zeros(1, 8), common.times.new_zeros(1, 15))[0]
    initial_hash = digest_tensor(initial)
    for condition in plan["conditions"]:
        for seed in plan["seeds"]:
            folder = raw/"datasets"/f"{condition}__{seed}"
            data = json.loads((folder/"inputs.json").read_text())
            reference = torch.load(folder/"reference.pt", weights_only=True)
            torch.testing.assert_close(common.times.new_tensor(data["positions_m_evaluation_only"]), reference["positions"], rtol=0, atol=0)
            if not CONDITIONS[condition][1] and data["reference_audit"]["source_sha256_normalized_lf"] != summary["deterministic_reference_hashes"][condition]:
                raise ValueError("Dataset uses a different deterministic reference.")
            if CONDITIONS[condition][1]:
                audit = data["reference_audit"]
                if audit["successive_coordinate_rmse_changes_m"][-1] > plan["reference_last_change_tolerance_m"]:
                    raise ValueError("Stochastic reference failed coupled refinement.")
                production = audit["production_staggered_refinement"]["rows"]
                if production[0]["coordinate_rmse_vs_fine_reference_m"] > plan["production_stochastic_rmse_tolerance_m"]:
                    raise ValueError("Production stochastic integration accuracy failed.")
                for row in audit["grids"]:
                    if row["minimum_det_F"] <= 0 or row["support_position_error_m"] != 0 or max(
                        row["maximum_internal_net_impulse_Ns"], row["maximum_internal_impulse_torque_Nms"], row["maximum_free_momentum_residual_Ns"]) > 1e-12:
                        raise ValueError("Reference physical/constraint audit failed.")
            observed = common.times.new_tensor(data["observed_rgb"])
            if digest_tensor(observed) != data["observations_sha256"]:
                raise ValueError("Saved image hash differs.")
            if data["reused_exact_v06_six_images"]:
                historical = json.loads((ROOT/"examples/accurate-reference"/f"inputs_{condition}.json").read_text())
                torch.testing.assert_close(observed, common.times.new_tensor(historical["noisy_rgb"]["0"])[common.view_frames], rtol=0, atol=0)
            else:
                clean = common.times.new_tensor(data["clean_rgb"]["0"])[common.view_frames]
                noise = .015*torch.randn(observed.shape, dtype=observed.dtype, generator=torch.Generator().manual_seed(seed+100000))
                torch.testing.assert_close(observed, clean+noise, rtol=0, atol=0)
            for mode in plan["modes"]:
                row = lookup(summary, condition, mode, seed)
                if row["observations_sha256"] != data["observations_sha256"] or row["initial_state_sha256"] != initial_hash:
                    raise ValueError("Models did not share RGB and initialization.")
                if row["source_fingerprint_sha256"] != summary["source_fingerprint_sha256"]:
                    raise ValueError("Mixed numerical source.")
                case = raw/"cases"/row["case"]
                if json.loads((case/"summary.json").read_text()) != row:
                    raise ValueError("Aggregate and case records differ.")
                model = construct(plan, network, calibration, mode)
                fitted = torch.load(case/"fit.pt", weights_only=True)
                if fitted["record"] != row["fit"]:
                    raise ValueError("Checkpoint and fit record differ.")
                value = float(model.objective(fitted["mean"][None], Observations(model.view_frames, observed, .015))[0])
                if abs(value-row["fit"]["marginal_objective"]) > 1e-8 or row["fit"]["profile_marginal_identity_abs_error"] > 1e-8:
                    raise ValueError("Saved normalized density failed reproduction.")
                point = json.loads((case/"point.json").read_text())
                positions = model.trajectory(fitted["mean"][None]).x[0]
                torch.testing.assert_close(positions, common.times.new_tensor(point["positions_m"]), rtol=0, atol=0)
                gap = (model.times > .3+1e-8)&(model.times < .7-1e-8)
                error = positions[:, model.free]-reference["positions"][:, model.free]
                actual_rmse = float(error[gap].square().mean().sqrt())
                if abs(actual_rmse-row["scores"]["original_gap"]["position_rmse_m"]) > 1e-12:
                    raise ValueError("Point reconstruction score differs from the independent reference.")
                mapping = model.position_map.flatten(0, 1)
                projected = torch.linalg.solve_triangular(fitted["precision_cholesky"], mapping.T, upper=False)
                sd = projected.square().sum(0).sqrt().reshape(model.config.frames, -1, 3)
                torch.testing.assert_close(sd, common.times.new_tensor(point["free_position_sd_m"]), rtol=1e-12, atol=1e-12)
                fields = json.loads((case/"fields.json").read_text())
                covariance = common.times.new_tensor(fields["whitened_force_coefficient_covariance"])
                within = common.times.new_tensor(fields["conditional_coefficient_covariance"])
                between = common.times.new_tensor(fields["between_trajectory_coefficient_covariance"])
                torch.testing.assert_close(covariance, within+between, rtol=1e-12, atol=1e-12)
                if float(torch.linalg.eigvalsh(covariance).min()) <= 0:
                    raise ValueError("Invalid force covariance.")
                portable = evaluate_force_posterior(fields, model.times)
                torch.testing.assert_close(portable["mean_total_N"], common.times.new_tensor(fields["external_total_mean_N"]), rtol=1e-12, atol=1e-12)
                torch.testing.assert_close(portable["mean_nodal_N"].sum(1), portable["mean_total_N"], rtol=1e-12, atol=1e-12)


def grouped(ax, summary, key, scale=1., ylabel="", title=""):
    x = np.arange(4)
    for index, mode in enumerate(MODES):
        values = [lookup(summary, c, mode)[key]*scale for c in CONDITIONS]
        ax.bar(x+(index-1)*.24, values, width=.23, color=COLORS[mode], label=LABELS[mode])
    ax.set_xticks(x, ["No push", "Push", "Random", "Push + random"], fontsize=8)
    ax.set(ylabel=ylabel, title=title)


def figures(raw, output, summary):
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    for ax, condition in zip(axes.flat, CONDITIONS):
        for index, mode in enumerate(MODES):
            values = [1000*lookup(summary, condition, mode, seed)["scores"]["original_gap"]["position_rmse_m"] for seed in summary["plan"]["seeds"]]
            center = np.mean(values)
            ax.bar(index, center, color=COLORS[mode], width=.65)
            ax.scatter(index+np.linspace(-.10, .10, len(values)), values, color="black", s=18, zorder=3)
            passing = lookup(summary, condition, mode)["stationary_count"]
            ax.text(index, max(values), f"{center:.2f} mm\n{passing}/3 stationary", ha="center", va="bottom", fontsize=8)
        ax.set_xticks(range(3), [LABELS[m] for m in MODES], fontsize=8)
        ax.set(title=TITLES[condition], ylabel="Missing-gap coordinate RMSE [mm]")
        ax.margins(y=.3)
    fig.suptitle("Same six images: does internal diffusion improve reconstruction?\nBars: mean across three seeds; black points: individual seeds")
    fig.tight_layout(); savefig(fig, output, "sparse_comparison.png")

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    grouped(axes[0], summary, "mean_gap_component_crps_m", 1000, "Gaussian component CRPS [mm]", "Lower is better")
    grouped(axes[1], summary, "mean_gap_component_90_coverage", 100, "Component coverage [%]", "Local nominal 90% intervals")
    axes[1].axhline(90, color="black", ls="--", lw=1)
    grouped(axes[2], summary, "mean_gap_component_90_width_m", 1000, "Mean interval width [mm]", "Width matters alongside coverage")
    axes[0].legend(fontsize=8)
    fig.suptitle("Local Gaussian uncertainty diagnostics; three seeds do not establish calibration")
    fig.tight_layout(); savefig(fig, output, "uncertainty.png")

    fig, ax = plt.subplots(figsize=(9, 4))
    grouped(ax, summary, "mean_force_rmse_N", 1., "External total-force RMSE [N]", "Does internal noise reduce incorrect external-force attribution?")
    ax.legend(fontsize=9); fig.tight_layout(); savefig(fig, output, "force_comparison.png")

    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True)
    for i, condition in enumerate(("no_load", "pulse_and_stochastic")):
        for mode in ("floor", "neural"):
            row = lookup(summary, condition, mode, 4101)
            fields = json.loads((raw/"cases"/row["case"]/"fields.json").read_text())
            t = np.array(fields["times_s"])
            mean, lo, hi, truth = [np.array(fields[k]) for k in ("external_total_mean_N", "external_total_q05_N", "external_total_q95_N", "truth_external_total_N_evaluation_only")]
            for col, dim in enumerate((0, 2)):
                ax = axes[i, col]
                ax.plot(t, mean[:, dim], color=COLORS[mode], label=LABELS[mode])
                ax.fill_between(t, lo[:, dim], hi[:, dim], color=COLORS[mode], alpha=.15)
                if mode == "neural": ax.plot(t, truth[:, dim], "--", color="black", label="True external force")
                ax.axvspan(.3, .7, color="gray", alpha=.08)
                ax.set(title=TITLES[condition]+" / "+"xyz"[dim], xlabel="Time [s]", ylabel="Total force [N]")
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("Example force distributions, seed 4101; local 90% bands, not calibrated")
    fig.tight_layout(); savefig(fig, output, "force_distributions.png")

    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    for ax, condition in zip(axes.flat, CONDITIONS):
        for mode in MODES:
            history = lookup(summary, condition, mode, 4101)["fit"]["history"]
            ax.semilogy(range(1, len(history)+1), [r["step_decrement"] for r in history], "o-", ms=3, color=COLORS[mode], label=LABELS[mode])
        ax.axhline(.05, color="gray", ls="--"); ax.set(title=TITLES[condition], xlabel="Iteration", ylabel="Gaussian-metric decrement")
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("Optimization at seed 4101; all seeds and stop flags remain in the records")
    fig.tight_layout(); savefig(fig, output, "convergence.png")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for condition, marker in (("stochastic", "o"), ("pulse_and_stochastic", "s")):
        for seed in summary["plan"]["seeds"]:
            data = json.loads((raw/"datasets"/f"{condition}__{seed}"/"inputs.json").read_text())
            audit = data["reference_audit"]
            axes[0].loglog([192, 384], np.array(audit["successive_coordinate_rmse_changes_m"])*1000, marker+"-", alpha=.6)
            production = audit["production_staggered_refinement"]["rows"]
            axes[1].loglog([r["substeps"] for r in production], [1000*r["coordinate_rmse_vs_fine_reference_m"] for r in production], marker+"-", alpha=.6)
    axes[0].set(title="Coupled reference refinement", xlabel="Coarser substeps in successive pair", ylabel="Pairwise coordinate change [mm]")
    axes[1].set(title="Production integrator, oracle-amplitude audit", xlabel="Substeps per frame", ylabel="Coordinate RMSE vs fine reference [mm]")
    fig.suptitle("Six stochastic sequences; these are numerical consistency checks, not absolute error bounds")
    fig.tight_layout(); savefig(fig, output, "reference_accuracy.png")

    condition, seed = "pulse_and_stochastic", 4101
    data = json.loads((raw/"datasets"/f"{condition}__{seed}"/"inputs.json").read_text())
    images = {m: torch.load(raw/"cases"/lookup(summary, condition, m, seed)["case"]/"images.pt", weights_only=True) for m in MODES}
    times = [0, 3, 6, 10, 14, 17, 20]; observed = {f: n for n, f in enumerate(data["observed_indices"])}
    tiles = [None, np.array(data["clean_rgb"]["0"]), images["floor"][0].numpy(), images["constant"][0].numpy(), images["neural"][0].numpy(),
             np.array(data["clean_rgb"]["200"]), images["floor"][200].numpy(), images["neural"][200].numpy()]
    labels = ["Sparse input", "Reference", "Current floor", "Constant internal", "Neural internal", "Unseen reference", "Unseen floor", "Unseen neural"]
    fig, axes = plt.subplots(8, 7, figsize=(12, 12))
    for i in range(8):
        for j, frame in enumerate(times):
            ax = axes[i, j]
            if i == 0 and frame not in observed:
                ax.set_facecolor("#e4e9eb"); ax.text(.5, .5, "Missing", ha="center", va="center", transform=ax.transAxes)
            else:
                image = np.array(data["observed_rgb"])[observed[frame]] if i == 0 else tiles[i][frame]
                ax.imshow(np.clip(image, 0, 1), interpolation="nearest")
            ax.set_xticks([]); ax.set_yticks([])
            for spine in ax.spines.values(): spine.set_visible(False)
            if j == 0: ax.set_ylabel(labels[i], fontsize=9)
            if i == 0: ax.set_title(f"t={frame*.05:.2f} s")
    fig.suptitle("Actual 24 x 24 images: pulse + stochastic motion, seed 4101\nSame six inputs for every model; unseen +200-degree camera never fitted")
    fig.tight_layout(); savefig(fig, output, "image_comparison.png")


def report(output, summary):
    def embedded(name, caption):
        content = base64.b64encode((output/name).read_bytes()).decode()
        return f'<figure><img src="data:image/png;base64,{content}" alt="{html.escape(caption)}"><figcaption>{html.escape(caption)}</figcaption></figure>'
    aggregate_rows = [[r["condition"], LABELS[r["mode"]], f"{1000*r['mean_gap_position_rmse_m']:.3f}",
        f"{1000*r['mean_gap_component_crps_m']:.3f}", f"{r['mean_gap_component_90_coverage']:.1%}",
        f"{1000*r['mean_gap_component_90_width_m']:.2f}", f"{r['mean_force_rmse_N']:.5f}",
        f"{r['stationary_count']}/3", f"{r['importance_screen_count']}/3"] for r in summary["aggregates"]]
    individual = [[r["condition"], r["seed"], r["mode"], f"{1000*r['scores']['original_gap']['position_rmse_m']:.3f}",
        f"{r['fit']['marginal_gaussian_metric_decrement']:.3g}", r["fit"]["stationarity_screen_pass"],
        f"{r['local_gaussian_audit']['relative_weight_ess']:.3f}", r["fit"]["stop_reason"]] for r in summary["cases"]]
    stationary = sum(r["fit"]["stationarity_screen_pass"] for r in summary["cases"])
    importance = sum(r["local_gaussian_audit"]["screen_pass"] for r in summary["cases"])
    document = f'''<!doctype html><html lang="en"><meta charset="utf-8"><title>Neural SDE with marginalized forces</title>
<style>body{{font:16px/1.5 system-ui;max-width:1250px;margin:32px auto;padding:0 24px;color:#20313b;background:#f9fafb}}img{{width:100%;height:auto}}figure{{margin:24px 0}}figcaption{{color:#536371;font-size:14px}}table{{width:100%;border-collapse:collapse;font-size:12px;background:white}}td,th{{padding:8px;border-bottom:1px solid #ddd;text-align:left}}th{{background:#e4edf1}}.notice{{background:#e5f2f4;border-left:5px solid #007c91;padding:18px}}h1,h2{{line-height:1.2}}</style>
<h1>Does neural internal noise help sparse-view force reconstruction?</h1>
<div class="notice">36 fixed fits: four synthetic conditions, three seeds and three noise models. Every model receives the same six moving-camera images, force prior and initialization. Both constant and neural balanced internal diffusion are compared with the current full-rank floor. All use exact conditional Gaussian force elimination. Read accuracy, uncertainty width, CRPS and inference flags together.</div>
<h2>What the experiment tests</h2>
<p>The current model has independent free-node velocity discrepancy. The constant and neural models add equal/opposite central internal impulses with the same rank-three spatial basis. Neural amplitudes depend on inferred strain and preceding-velocity speed; the constant amplitude matches RMS neural output on separate calibration features. The 33-parameter neural network is frozen after independent synthetic training, not learned from the sparse test video. All retain the same fixed discrepancy floor.</p>
<p>This is a noise-model comparison with force marginalization held fixed. It does not independently measure marginalization versus joint fitting; that was a separate v0.6 comparison. Adding internal diffusion may help stochastic motion but can introduce unnecessary flexibility on deterministic motion. Three seeds are exploratory.</p>
{embedded('sparse_comparison.png', 'Missing-gap reconstruction under identical six-view inputs. No rows or seeds are selected away; black markers show individual outcomes.')}
{table(['Truth', 'Model', 'Gap mm', 'CRPS mm', 'Local 90% coverage', 'Width mm', 'Force RMSE N', 'Stationary', 'Weight screen'], aggregate_rows)}
<h2>Uncertainty and force attribution</h2>
<p>Local Gaussian intervals use a metric accounting for both residual-mean and covariance derivatives; this is not the exact posterior Hessian or established calibration. Coverage counts correlated free-node components. CRPS rewards accuracy and appropriate width together. {stationary}/36 fits pass stationarity and {importance}/36 pass the independent 128-proposal weight-concentration screen; individual flags follow below. Passing is not proof of adequate posterior exploration.</p>
{embedded('uncertainty.png', 'Compare width and CRPS alongside coverage. Wider intervals can increase coverage without improving useful uncertainty.')}
{embedded('force_comparison.png', 'External-force error; the stochastic-only truth has no external push. Noise can otherwise be attributed incorrectly to the smooth external-force model.')}
{embedded('force_distributions.png', 'Selected force functions at a predeclared seed, with the complete multi-seed comparison in the table. Bands are local approximations.')}
<h2>What marginalization contributes</h2>
<p>For an inferred path z, mechanical residual r(z)=G a+eta has Gaussian force coefficients a and state-dependent disturbance covariance Q(z). Integrating a gives C(z)=Q(z)+G G^T. The exact conditional force covariance is S(z)=[I+G^T Q(z)^(-1)G]^(-1). Both log determinants and derivatives of Q are retained.</p>
<p>This removes the separate force-update loop and preserves the full conditional force distribution. The trajectory/material problem still requires nonlinear optimization. With fixed Q/G, marginal and best-force joint trajectory optima coincide; no universal speed/accuracy gain follows. With state-dependent Q, marginal_NLL=min_force joint_NLL+0.5 logdet[I+G^T Q^(-1)G]+constant. That varying force-volume correction distinguishes marginalization from choosing only the best force.</p>
<p>Force covariance is E[S(z)]+Cov[m(z)] over local trajectory draws; unlike v0.6, the first term varies and is averaged. Exact force elimination does not make the remaining trajectory posterior exact. Unknown geometry, missing views and force/noise ambiguities remain.</p>
<h2>Actual images and independent simulation checks</h2>
{embedded('image_comparison.png', 'Native images are 24 x 24 and enlarged without inventing detail. The middle input is missing. Display clips noisy pixels only; fitting uses unclipped RGB.')}
<p>Deterministic reference motion is RK4-96. Stochastic references use an independently implemented fine Euler-Maruyama integrator with analytic state-dependent edge amplitudes, coupled Brownian paths at 192/384/768 substeps per frame, audited impulses/supports, and a final-grid consistency tolerance. An evaluation-only oracle-amplitude audit separately checks the production staggered solver at six/twelve substeps. It does not initialize or fit any reconstruction.</p>
{embedded('reference_accuracy.png', 'Reference and production stochastic time refinement. The same constitutive law is used; this is numerical verification rather than experimental or mesh validation.')}
{embedded('convergence.png', 'Optimization histories at seed 4101. Normalized density, shared input/initialization hashes and covariance identities were verified for every published fit.')}
{table(['Truth', 'Seed', 'Model', 'Gap mm', 'Final decrement', 'Stationary', 'Relative weight ESS', 'Stop'], individual)}
<h2>Limits and handover</h2>
<p>One known 8-node/6-tetrahedron mesh, 30 Gaussians, known cameras/appearance/density/supports, one six-frame schedule and three seeds. Fixed-profile temporal forces, frozen simulator-supervised diffusion, and a retained full-rank floor restrict the conclusions. There is no real-video test, geometry/camera inference, arbitrary force localization, mesh convergence, or proof of calibrated nonlinear uncertainty. The neural network is not identified from the test video, and uncertainty in its calibrated weights is not propagated.</p>
<p>Exact inputs, reference audits/positions, network weights, all fitted solver coordinates, force distributions, per-seed measurements, CSV, plan and source hashes accompany this offline report. Full local precision factors remain in raw checkpoints or are regenerated using the fixed committed plan. Historical reports are preserved.</p></html>'''
    (output/"report.html").write_text(document, encoding="utf-8")


@torch.no_grad()
def publish(raw, output, source_ref):
    torch.set_num_threads(2)
    raw, output = Path(raw).resolve(), Path(output).resolve()
    if output.exists(): raise FileExistsError("Choose an unused publication directory.")
    summary = json.loads((raw/"summary.json").read_text())
    commit = source_commit(source_ref)
    verify(raw, summary)
    output.mkdir(parents=True)
    for path in raw.rglob("*.json"):
        target = output/path.relative_to(raw); target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
    network, calibration = calibrate(summary["plan"], raw/"calibration.json")
    model = construct(summary["plan"], network, calibration, "neural")
    write_json(output/"model_constants.json", {
        "rest_nodes_m": model.mesh.rest.tolist(), "tetrahedra": model.mesh.cells.tolist(),
        "nodal_mass_kg": model.mesh.mass.tolist(), "pinned_nodes": model.mesh.pinned.tolist(),
        "edges": model.mesh.edges.tolist(), "internal_spatial_basis": model.internal_basis.tolist(),
        "internal_edge_incidence": model.edge_incidence.tolist(),
        "force_spatial_nodal_weights": model.force_prior.spatial_weights.tolist(),
        "force_inducing_times_s": model.force_prior.knots.tolist(),
        "step_times_s": model.step_times.tolist(), "drift_step_sizes_s": model.step_sizes.tolist(),
        "kick_sizes_s": model.kick_sizes.tolist(),
        "floor_velocity_diffusion": model.marginal_config.velocity_diffusion,
        "qualification": "actual numerical basis/sign convention and scene constants used in the run; not fitted truth"})
    for folder in (raw/"datasets").iterdir():
        reference = torch.load(folder/"reference.pt", weights_only=True)
        record = {"positions_m": reference["positions"].tolist(), "velocities_m_per_s": reference["velocities"].tolist(), "audit": reference["audit"]}
        if "brownian_finest_increments" in reference:
            record["brownian_finest_increments_sha256_float64_bytes"] = digest_tensor(reference["brownian_finest_increments"])
        write_json(output/"datasets"/folder.name/"reference.json", record)
    for row in summary["cases"]:
        fitted = torch.load(raw/"cases"/row["case"]/"fit.pt", weights_only=True)
        write_json(output/"cases"/row["case"]/"solver_state.json", {"coordinates": fitted["mean"].tolist(),
            "conditional_force_mean_at_MAP": fitted["conditional_force_mean"].tolist(),
            "qualification": "point coordinates only; full precision regenerated by the committed plan"})
        if row["mode"] == "neural":
            full = model.step_trajectory(fitted["mean"][None])
            features = edge_features(full.x[:, :-1], full.v[:, :-1], model.mesh)[0]
            amplitude = network(features)
            cov = model.noise_covariance(fitted["mean"][None])
            baseline_trace = model.free_dof*model.marginal_config.velocity_diffusion**2*model.kick_sizes
            ratio = cov[0].diagonal(dim1=-2, dim2=-1).sum(-1)/baseline_trace-1
            write_json(output/"cases"/row["case"]/"internal_noise_diagnostics.json", {
                "neural_edge_amplitude_mean": float(amplitude.mean()),
                "neural_edge_amplitude_min_max": [float(amplitude.min()), float(amplitude.max())],
                "neural_edge_amplitude_coefficient_of_variation": float(amplitude.std()/amplitude.mean()),
                "constant_calibration_amplitude": calibration["constant_amplitude"],
                "mean_internal_to_floor_velocity_variance_trace_ratio": float(ratio.mean()),
                "qualification": "post-fit diagnostic on MAP states, not used to tune network/noise hyperparameters"})
    with (output/"comparison.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["condition", "seed", "mode", "gap_rmse_mm", "gap_crps_mm", "gap_coverage", "gap_width_mm", "force_rmse_N", "stationary", "weight_relative_ess", "fit_seconds"])
        for row in summary["cases"]:
            gap = row["scores"]["original_gap"]
            writer.writerow([row["condition"], row["seed"], row["mode"], 1000*gap["position_rmse_m"], 1000*gap["component_gaussian_crps_m"],
                gap["component_90_coverage"], 1000*gap["component_90_mean_width_m"], row["force_scores"]["rmse_N"],
                row["fit"]["stationarity_screen_pass"], row["local_gaussian_audit"]["relative_weight_ess"], row["fit"]["seconds_including_final_covariance"]])
    figures(raw, output, summary); report(output, summary)
    write_json(output/"publication.json", {"implementation_source_commit": commit,
        "numerical_source_fingerprint_sha256": source_fingerprint(), "publisher_sha256_normalized_lf": canonical_hash(Path(__file__)),
        "verification": "all 36 fits; matching plan/source/calibration; all shared RGB and initial states; source reference and stochastic balance/refinement gates; reproduced objectives/point paths; force covariance identities and portable queries; aggregate identities",
        "files_sha256_text_normalized_lf_binary_raw": {p.relative_to(output).as_posix(): canonical_hash(p) for p in sorted(output.rglob("*")) if p.is_file()}})
    print("Verified and published:", output, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-ref", required=True)
    args = parser.parse_args(); publish(args.raw, args.output, args.source_ref)
