"""Verify and publish calibrated sparse-view comparison and uncertainty checks."""
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
from physgauss.accurate_study import digest_tensor
from physgauss.benchmark import Observations
from physgauss.calibrated_study import default_plan, check_splits, case_condition, evaluation_record
from physgauss.neural_marginal_study import calibrate, construct
from physgauss.physics import edge_features
from physgauss.stochastic import calibration_law, NeuralDiffusion
from physgauss.uncertainty_validation import apply_conformal
from physgauss.cli import source_fingerprint
from physgauss.likelihood_audit import _environment
from physgauss.study import write_json
from publish_accurate_study import canonical_hash, source_commit, table, savefig
from publish_neural_marginal import publish as publish_comparison


LABELS = {"floor": "Independent floor", "constant": "Calibrated constant", "neural": "Calibrated neural"}
COLORS = {"floor": "#788b99", "constant": "#c9782a", "neural": "#007c91"}


@torch.no_grad()
def verify_validation(raw, summary, plan):
    if summary["plan"] != plan or summary["source_fingerprint_sha256"] != source_fingerprint():
        raise ValueError("Validation plan/source mismatch.")
    check_splits(plan)
    if apply_conformal(summary["cases"], plan["interval_calibration_seeds"], plan["interval_test_seeds"], plan["interval_alpha"]) != summary["interval_validation"]:
        raise ValueError("Interval metrics do not reproduce.")
    network, calibration = calibrate(plan, raw/"calibration.json")
    if hashlib.sha256(json.dumps(calibration, sort_keys=True).encode()).hexdigest() != summary["calibration_sha256"]:
        raise ValueError("Validation calibration changed.")
    frozen = json.loads((raw/"frozen_intervals.json").read_text())
    for row in summary["interval_validation"]:
        original = next(r for r in frozen["quantiles"] if r["mode"] == row["mode"])
        if original["sd_multiplier"] != row["sd_multiplier"]: raise ValueError("Test-tuned conformal quantile.")
    expected = {(seed, mode) for seed in plan["interval_calibration_seeds"]+plan["interval_test_seeds"] for mode in plan["interval_modes"]}
    if len(summary["cases"]) != len(expected) or {(r["seed"], r["mode"]) for r in summary["cases"]} != expected:
        raise ValueError("Incomplete/duplicated uncertainty validation.")
    shared = construct(plan, network, calibration, "floor")
    initial = shared.rollout(shared.times.new_zeros(1, 8), shared.times.new_zeros(1, 15))[0]
    observed_hashes = {}
    # OS/BLAS runtimes can differ in their floating-point bit patterns.
    # Worker hashes must still agree across every model and seed.
    initial_hash = digest_tensor(initial) if summary["environment"] == _environment() else summary["cases"][0]["initial_state_sha256"]
    for row in summary["cases"]:
        seed, mode, condition = row["seed"], row["mode"], row["condition"]
        if case_condition(seed) != condition: raise ValueError("Condition selection changed.")
        label = f"{condition}__{seed}"
        inputs = json.loads((raw/"datasets"/label/"inputs.json").read_text())
        reference = torch.load(raw/"datasets"/label/"reference.pt", weights_only=True)
        audit = reference["audit"]
        if audit != inputs["reference_audit"] or audit["successive_coordinate_rmse_changes_m"][-1] > plan["reference_last_change_tolerance_m"]:
            raise ValueError("Independent stochastic reference changed/failed refinement.")
        if audit["production_staggered_refinement"]["rows"][0]["coordinate_rmse_vs_fine_reference_m"] > plan["production_stochastic_rmse_tolerance_m"]:
            raise ValueError("Production reference gate failed.")
        for grid in audit["grids"]:
            if grid["minimum_det_F"] <= 0 or grid["support_position_error_m"] != 0 or max(grid["maximum_internal_net_impulse_Ns"],
                grid["maximum_internal_impulse_torque_Nms"], grid["maximum_free_momentum_residual_Ns"]) > 1e-12:
                raise ValueError("Reference balance/supports failed.")
        model = construct(plan, network, calibration, mode)
        rgb = model.times.new_tensor(inputs["observed_rgb"])
        if digest_tensor(rgb) != row["observations_sha256"]: raise ValueError("RGB differs.")
        clean = model.times.new_tensor(inputs["clean_rgb"]["0"])[model.view_frames]
        expected_noise = .015*torch.randn(rgb.shape, dtype=rgb.dtype, generator=torch.Generator().manual_seed(seed+100000))
        torch.testing.assert_close(rgb, clean+expected_noise, rtol=0, atol=1e-12)
        if seed in observed_hashes and observed_hashes[seed] != row["observations_sha256"]: raise ValueError("Unmatched model inputs.")
        observed_hashes[seed] = row["observations_sha256"]
        if initial_hash is not None and row["initial_state_sha256"] != initial_hash: raise ValueError("Initialization differs.")
        initial_hash = row["initial_state_sha256"]
        case = raw/"cases"/f"{label}__{mode}"
        if json.loads((case/"summary.json").read_text()) != row: raise ValueError("Case record mismatch.")
        fitted = torch.load(case/"fit.pt", weights_only=True)
        if fitted["record"] != row["fit"]: raise ValueError("Validation fit record differs from checkpoint.")
        reproduced = evaluation_record(model, fitted, {"x": reference["positions"]}, condition, seed, mode)
        for key, value in reproduced.items():
            if isinstance(value, (float, list)):
                torch.testing.assert_close(torch.tensor(value, dtype=torch.float64), torch.tensor(row[key], dtype=torch.float64), rtol=1e-9, atol=2e-10)
            elif value != row[key]: raise ValueError("Position intervals/score changed: "+key)
        objective = float(model.objective(fitted["mean"][None], Observations(model.view_frames, rgb, .015))[0])
        if abs(objective-row["fit"]["marginal_objective"]) > 1e-7: raise ValueError("Normalized objective changed.")
        fields = json.loads((case/"fields.json").read_text())
        covariance, within, between = [model.times.new_tensor(fields[key]) for key in
            ("whitened_force_coefficient_covariance", "conditional_coefficient_covariance", "between_trajectory_coefficient_covariance")]
        torch.testing.assert_close(covariance, within+between, rtol=1e-12, atol=1e-12)
    return network, calibration


def compact_validation(raw, output, summary):
    output.mkdir(parents=True, exist_ok=True)
    for name in ("manifest.json", "calibration.json", "summary.json", "frozen_intervals.json"):
        shutil.copyfile(raw/name, output/name)
    for path in list(raw.glob("execution_*.json"))+list(raw.glob("artifact_download.json")):
        shutil.copyfile(path, output/path.name)
    for folder in (raw/"datasets").iterdir():
        data = json.loads((folder/"inputs.json").read_text())
        # Keep exact observed pixels; evaluation clean views are reconstructible
        # from saved states and the renderer and are never needed for inference.
        compact = {k: v for k, v in data.items() if k != "clean_rgb"}
        compact["raw_inputs_sha256_normalized_lf"] = canonical_hash(folder/"inputs.json")
        target = output/"datasets"/folder.name; target.mkdir(parents=True, exist_ok=True)
        (target/"inputs.json").write_text(json.dumps(compact, separators=(",", ":")), encoding="utf-8")
        ref = torch.load(folder/"reference.pt", weights_only=True)
        write_json(target/"reference.json", {"positions_m": ref["positions"].tolist(), "velocities_m_per_s": ref["velocities"].tolist(),
            "audit": ref["audit"], "brownian_finest_increments_sha256_float64_bytes": digest_tensor(ref["brownian_finest_increments"])})
    for folder in (raw/"cases").iterdir():
        target = output/"cases"/folder.name; target.mkdir(parents=True, exist_ok=True)
        for name in ("summary.json", "fields.json"): shutil.copyfile(folder/name, target/name)
        fit = torch.load(folder/"fit.pt", weights_only=True)
        write_json(target/"solver_state.json", {"coordinates": fit["mean"].tolist(),
            "conditional_force_mean_at_MAP": fit["conditional_force_mean"].tolist(),
            "qualification": "point coordinates; regenerate Fisher precision from observations and recorded source using recover_neural_posterior.py"})
    with (output/"validation.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream); writer.writerow(["split", "condition", "seed", "mode", "gap_rmse_mm", "max_standardized_error", "component_coverage", "stationary", "weight_relative_ess"])
        for row in summary["cases"]:
            writer.writerow([row["split"], row["condition"], row["seed"], row["mode"], 1000*row["gap_rmse_m"], row["gap_max_standardized_error"],
                row["raw_component_90_coverage"], row["stationary"], row["local_gaussian_audit"]["relative_weight_ess"]])


@torch.no_grad()
def calibration_audit(raw_comparison, old, output, network):
    old_record = json.loads((old/"calibration.json").read_text()); old_net = NeuralDiffusion()
    old_net.load_state_dict({k: torch.tensor(v, dtype=torch.float64) for k, v in old_record["state_dict"].items()})
    summary = json.loads((raw_comparison/"summary.json").read_text())
    model = construct(summary["plan"], network, json.loads((raw_comparison/"calibration.json").read_text()), "neural")
    rows = []; truth_values = []; old_values = []; new_values = []
    for condition in ("stochastic", "pulse_and_stochastic"):
        for seed in summary["plan"]["seeds"]:
            ref = torch.load(raw_comparison/"datasets"/f"{condition}__{seed}"/"reference.pt", weights_only=True)
            features = edge_features(ref["positions"][None], ref["velocities"][None], model.mesh)[0]
            truth = calibration_law(features); previous = old_net(features); new = network(features)
            rows.append({"condition": condition, "seed": seed, "old_sigma_relative_rmse": float((previous/truth-1).square().mean().sqrt()),
                "new_sigma_relative_rmse": float((new/truth-1).square().mean().sqrt()), "true_cv": float(truth.std()/truth.mean()),
                "new_cv": float(new.std()/new.mean())})
            truth_values.append(truth.flatten()); old_values.append(previous.flatten()); new_values.append(new.flatten())
    truth, previous, new = [torch.cat(v) for v in (truth_values, old_values, new_values)]
    result = {"cases": rows, "old_pooled_relative_sigma_rmse": float((previous/truth-1).square().mean().sqrt()),
        "new_pooled_relative_sigma_rmse": float((new/truth-1).square().mean().sqrt()),
        "qualification": "evaluation-only true frame states, same audit as v0.7; no model/interval tuning; fresh trajectories are tested separately"}
    write_json(output/"calibration_audit.json", result)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].scatter(truth.numpy(), previous.numpy(), s=7, alpha=.15, label="Previous neural", color="#9b5963")
    axes[0].scatter(truth.numpy(), new.numpy(), s=7, alpha=.18, label="Recalibrated neural", color=COLORS["neural"])
    limits = [float(min(truth.min(), new.min(), previous.min())), float(max(truth.max(), new.max(), previous.max()))]
    axes[0].plot(limits, limits, "k--", lw=1); axes[0].legend(fontsize=8)
    axes[0].set(xlabel="True amplitude", ylabel="Network amplitude", title="Same true benchmark states (evaluation only)")
    axes[1].bar([0, 1], [100*result["old_pooled_relative_sigma_rmse"], 100*result["new_pooled_relative_sigma_rmse"]], color=["#9b5963", COLORS["neural"]])
    axes[1].set_xticks([0, 1], ["Previous neural", "Recalibrated neural"]); axes[1].set(ylabel="Relative amplitude RMSE [%]", title="Calibration, not reconstruction error")
    fig.tight_layout(); savefig(fig, output, "calibration.png")
    return result


def figures(output, comparison, previous, validation, posterior):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    conditions = ["no_load", "external_pulse", "stochastic", "pulse_and_stochastic"]
    names = ["No push", "Push", "Random", "Push + random"]
    for ax, indices in zip(axes, ((0, 1), (2, 3))):
        x = np.arange(2)
        for number, (study, mode, label, color) in enumerate(((previous, "floor", "Independent floor", COLORS["floor"]),
                (previous, "neural", "Previous neural", "#9b5963"), (comparison, "constant", "Calibrated constant", COLORS["constant"]),
                (comparison, "neural", "Calibrated neural", COLORS["neural"]))):
            values = [1000*next(r for r in study["aggregates"] if r["condition"] == conditions[i] and r["mode"] == mode)["mean_gap_position_rmse_m"] for i in indices]
            bars = ax.bar(x+(number-1.5)*.20, values, width=.19, label=label, color=color)
            ax.bar_label(bars, fmt="%.2f", fontsize=8)
        ax.set_xticks(x, [names[i] for i in indices]); ax.set(ylabel="Gap coordinate RMSE [mm]"); ax.margins(y=.2)
    axes[0].set_title("Deterministic truth"); axes[1].set_title("Stochastic truth")
    axes[0].legend(fontsize=8); fig.suptitle("Same exact six images and references; three seeds per condition")
    fig.tight_layout(); savefig(fig, output, "paired_comparison.png")

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for index, row in enumerate(validation["interval_validation"]):
        mode = row["mode"]; color = COLORS[mode]
        rates = [row["raw_simultaneous_covered"]/20, row["conformal_trajectory_coverage"]]
        xs = np.array([0, 1])+(index-.5)*.3
        axes[0].bar(xs, np.array(rates)*100, width=.28, color=color, label=LABELS[mode])
        axes[0].errorbar(xs[1], 100*rates[1], yerr=[[100*(rates[1]-row["coverage_95_wilson_interval"][0])],
            [100*(row["coverage_95_wilson_interval"][1]-rates[1])]], fmt="none", ecolor="black", capsize=4)
        axes[1].bar(xs, [1000*row["raw_mean_width_m"], 1000*row["conformal_mean_width_m"]], width=.28, color=color)
        tests = [r for r in validation["cases"] if r["mode"] == mode and r["split"] == "test"]
        axes[2].plot(range(1, 21), [r["gap_max_standardized_error"] for r in tests], "o-", color=color, label=LABELS[mode], alpha=.8)
        axes[2].axhline(row["sd_multiplier"], color=color, ls="--")
    axes[0].axhline(90, color="black", ls=":"); axes[0].set_xticks([0, 1], ["Raw local bands", "Conformal sets"])
    axes[0].set(ylabel="Entire missing trajectory covered [%]", ylim=(0, 110), title="20 independent held-out motions")
    axes[0].legend(fontsize=8); axes[1].set_xticks([0, 1], ["Raw local bands", "Conformal sets"])
    axes[1].set(ylabel="Mean component interval width [mm]", title="Coverage and width together")
    axes[2].set(xlabel="Held-out trajectory", ylabel="Maximum |error| / local SD", title="Frozen thresholds; all test cases retained")
    fig.suptitle("Whole-gap coverage is stricter than coordinate coverage; error bars: 95% Wilson intervals")
    fig.tight_layout(); savefig(fig, output, "interval_validation.png")

    # First held-out motion is fixed by the protocol, not selected for its image/error.
    seed = validation["plan"]["interval_test_seeds"][0]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharex=True)
    for row in validation["cases"]:
        if row["seed"] != seed: continue
        mode = row["mode"]
        inputs = json.loads((output/"validation/datasets"/f"{row['condition']}__{seed}"/"inputs.json").read_text())
        truth = np.array(inputs["positions_m_evaluation_only"])[7:14, 2]
        point = np.array(row["positions_m"])[7:14, 2]
        # First free vertex is global vertex2, and component0 in the score array.
        sd = np.array(row["gap_standard_deviations_m"])[:, 0]
        quantile = next(r["sd_multiplier"] for r in validation["interval_validation"] if r["mode"] == mode)
        for dimension, ax in enumerate(axes):
            t = np.arange(7, 14)*.05; error = (point[:, dimension]-truth[:, dimension])*1000
            ax.plot(t, error, "o-", color=COLORS[mode], label=LABELS[mode])
            ax.fill_between(t, error-quantile*sd[:, dimension]*1000, error+quantile*sd[:, dimension]*1000, color=COLORS[mode], alpha=.16)
    for dimension, ax in enumerate(axes): ax.axhline(0., color="black", ls="--"); ax.set(xlabel="Time [s]", ylabel="Point error and conformal band [mm]", title="xyz"[dimension])
    axes[0].legend(fontsize=8); fig.suptitle(f"First held-out motion, seed {seed}: first free vertex, frozen whole-gap prediction sets")
    fig.tight_layout(); savefig(fig, output, "heldout_intervals.png")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for chain, trace in enumerate(posterior["log_target_traces"]): axes[0].plot(trace, lw=.8, label=f"Chain {chain+1}")
    axes[0].set(xlabel="Saved retained transition (decimated)", ylabel="Log marginal target", title="Exact-target MCMC, no truth in transitions")
    axes[0].legend(fontsize=8)
    diagnostics = posterior["diagnostics"]["quantities"]
    rhats = [v["rank_folded_split_rhat"] for v in diagnostics.values() if v["rank_folded_split_rhat"] is not None]
    axes[1].hist(rhats, bins=35, color=COLORS["neural"]); axes[1].axvline(1.01, color="black", ls="--")
    axes[1].set(xlabel="Rank/folded split Rhat", ylabel="Monitored quantities", title=f"{len(posterior['diagnostics']['failed_quantities'])}/{len(diagnostics)} fail complete screen")
    fig.suptitle("Posterior exploration audit is separate from conformal prediction coverage")
    fig.tight_layout(); savefig(fig, output, "posterior_audit.png")


def report(output, comparison, previous, validation, posterior, calibration, audit):
    def embed(name, caption):
        data = base64.b64encode((output/name).read_bytes()).decode()
        return f'<figure><img src="data:image/png;base64,{data}" alt="{html.escape(caption)}"><figcaption>{html.escape(caption)}</figcaption></figure>'
    def pooled(study, mode): return 1000*np.mean([r["mean_gap_position_rmse_m"] for r in study["aggregates"] if r["mode"] == mode and r["condition"] in ("stochastic", "pulse_and_stochastic")])
    rows = [[r["condition"], LABELS[r["mode"]], f"{1000*r['mean_gap_position_rmse_m']:.3f}", f"{1000*r['mean_gap_component_crps_m']:.3f}",
             f"{r['mean_gap_component_90_coverage']:.1%}", f"{1000*r['mean_gap_component_90_width_m']:.2f}", f"{r['stationary_count']}/3", f"{r['importance_screen_count']}/3"] for r in comparison["aggregates"]]
    intervals = [[LABELS[r["mode"]], f"{r['sd_multiplier']:.3f}", f"{r['raw_simultaneous_covered']}/20", f"{r['conformal_simultaneous_covered']}/20",
        f"{r['coverage_95_wilson_interval'][0]:.1%}–{r['coverage_95_wilson_interval'][1]:.1%}", f"{1000*r['conformal_mean_width_m']:.2f}",
        f"{1000*r['mean_gap_rmse_m']:.3f}"] for r in validation["interval_validation"]]
    test_rows = [[r["seed"], r["condition"], LABELS[r["mode"]], f"{1000*r['gap_rmse_m']:.3f}", f"{r['gap_max_standardized_error']:.3f}",
        r["stationary"], f"{r['local_gaussian_audit']['relative_weight_ess']:.3f}"] for r in validation["cases"] if r["split"] == "test"]
    stationary = sum(r["fit"]["stationarity_screen_pass"] for r in comparison["cases"])
    validation_stationary = sum(r["stationary"] for r in validation["cases"])
    passed = sum(r["local_gaussian_audit"]["screen_pass"] for r in comparison["cases"])
    full = posterior["diagnostics"]
    force_lines = []
    heldout_calibration = [r["evaluation_only_amplitude_calibration"]["relative_sigma_rmse"] for r in validation["cases"] if r["split"] == "test" and r["mode"] == "neural"]
    heldout_sigma_error = float(np.sqrt(np.mean(np.square(heldout_calibration))))
    for mode in ("constant", "neural"):
        group = [r for r in validation["cases"] if r["split"] == "test" and r["mode"] == mode]
        force_lines.append(f"{LABELS[mode]} mean force-component coverage {np.mean([r['local_force_scores']['component_90_coverage'] for r in group]):.1%}, width {np.mean([r['local_force_scores']['component_90_mean_width_N'] for r in group]):.4f} N")
    document = f'''<!doctype html><html lang="en"><meta charset="utf-8"><title>Calibrated neural noise and sparse-view uncertainty</title>
<style>body{{font:16px/1.5 system-ui;max-width:1200px;margin:30px auto;padding:0 22px;color:#20313b;background:#fafbfc}}img{{width:100%;height:auto}}figure{{margin:25px 0}}figcaption{{font-size:14px;color:#52636d}}table{{width:100%;border-collapse:collapse;font-size:12px;background:white}}td,th{{padding:8px;border-bottom:1px solid #ddd;text-align:left}}th{{background:#e4edf1}}.notice{{background:#e5f2f4;border-left:5px solid #007c91;padding:18px}}h1,h2{{line-height:1.2}}</style>
<h1>Better neural noise calibration: what improved, and which uncertainty is validated?</h1>
<div class="notice">114 sparse-view fits: 36 paired reruns plus 78 fits on 39 new independent motions. All models use marginalized external forces and the same six-view moving-camera schedule. Network calibration, prediction-set calibration, held-out evaluation and nonlinear posterior exploration are separate checks.</div>
<p><strong>Paired stochastic-case gap errors:</strong> independent floor {pooled(comparison,'floor'):.3f} mm; previous neural {pooled(previous,'neural'):.3f} mm; recalibrated constant {pooled(comparison,'constant'):.3f} mm; recalibrated neural {pooled(comparison,'neural'):.3f} mm. Three underlying seeds are shared across conditions. These averages do not prove a neural-specific advantage.</p>
<h2>1. Calibration now matches the operating regime</h2>
<p>The same 33-parameter network uses scaled smooth magnitudes of strain and relative speed, encoding the declared sign symmetry. Twelve independent training motions vary mechanics and initial conditions; four other motions validate it. Operating-region and broad feature states receive equal mass. Training minimizes Gaussian innovation NLL from 128 independent synthetic innovations per state; analytic amplitudes are evaluation labels, not regression targets. The constant control is fitted directly to training innovations, avoiding inherited network bias. Weights are frozen before every reconstruction.</p>
<p>On independent calibration-validation motions: relative amplitude RMSE <strong>{100*calibration['validation']['relative_sigma_rmse']:.3f}%</strong>, standardized innovation variance {calibration['validation']['standardized_innovation_variance']:.3f}, and nominal-90% innovation coverage {calibration['validation']['innovation_90_coverage']:.2%}. On the unchanged benchmark states, amplitude error fell from {100*audit['old_pooled_relative_sigma_rmse']:.2f}% to {100*audit['new_pooled_relative_sigma_rmse']:.3f}%. On the 20 fresh held-out motion sequences, pooled frame-state amplitude error is {100*heldout_sigma_error:.3f}%. Synthetic simulator supervision is still required; weights are not learned from test RGB and weight uncertainty is not propagated.</p>
{embed('calibration.png', 'Evaluation-only calibration on unchanged true frame states. Independent new-motion checks are also published per case.')}
<h2>2. Sparse-view reconstruction under matched inputs</h2>
{embed('paired_comparison.png', 'All methods receive identical six observed images and the same accurate reference motion. Old results remain archived; new constant and neural calibrations both change.')}
{table(['Truth','Model','Gap mm','Local CRPS mm','Component 90%','Width mm','Stationary','Weight screen'], rows)}
<p>{stationary}/36 comparison fits and {validation_stationary}/78 independent-validation fits pass stationarity. {passed}/36 comparison fits pass the local nonlinear importance-weight screen. Optimizer stationarity is separate from posterior exploration. No failed case is removed.</p>
{embed('comparison/image_comparison.png', 'Actual 24×24 sparse inputs, missing frame, reconstructed views and unseen camera; no generated or enhanced image content.')}
<h2>3. Held-out validation of whole-gap prediction sets</h2>
<p>Nineteen independent stochastic motions calibrate a maximum standardized position-error score; twenty new motions test it. Each case independently draws pulse/no-pulse with probability 1/2. A complete seven-frame gap (84 xyz components) supplies one score. Frozen model-specific multipliers use the 18th of 19 calibration scores. Calibration completes before any test fitting; test truths never adjust the network or interval multiplier.</p>
<p>Raw local 90% intervals are component-wise. Conformal sets target <strong>simultaneous 90% coverage of the entire missing-motion gap</strong>, a stricter event. Their finite-sample marginal property assumes exchangeability of calibration and new motions and averages over calibration-set randomness. It is not a guarantee for every individual video, a Bayesian posterior correction, or evidence for another scene/view schedule. Twenty trials leave substantial statistical uncertainty.</p>
{embed('interval_validation.png', 'All 20 independent held-out motions; 95% Wilson intervals use whole trajectories as trials. Wider or narrower prediction sets are reported explicitly.')}
{table(['Model','SD multiplier','Raw whole-gap','Conformal whole-gap','95% Wilson interval','Set width mm','Test gap mm'], intervals)}
{embed('heldout_intervals.png', 'First held-out motion fixed by seed order; prediction bands use only the frozen calibration multipliers.')}
<h2>4. Nonlinear posterior uncertainty remains a separate question</h2>
<p>The neural pulse-plus-stochastic seed-4101 mode was refined, then checked with observed Hessians at two finite-difference scales. Relative curvature change is {posterior['curvature_relative_refinement']:.3g}. Importance screens compare the Fisher Gaussian with the observed-Hessian Gaussian using 512 draws each. Four Metropolis-corrected chains target the full nonlinear marginalized-force density, with 1,000 warmup and 2,000 retained steps each. All 1,448 state coordinates, 84 missing-position components and target density are monitored.</p>
<p><strong>Posterior status: {html.escape(posterior['posterior_status'])}.</strong> {len(full['failed_quantities'])}/{len(full['quantities'])} quantities fail the complete Rhat/ESS/frozen-chain screen. Even a finite diagnostic pass cannot exclude unvisited modes. Conformal prediction coverage does not validate the Bayesian posterior, material intervals, or force intervals.</p>
{embed('posterior_audit.png', 'Full nonlinear sampling audit. Failed diagnostics are retained and reported rather than hidden by predictive coverage.')}
<p>Approximate local force intervals are separately scored on all test motions: {'; '.join(force_lines)}. These are correlated component averages, not independent force-trajectory coverage guarantees. They remain approximate.</p>
<h2>5. Numerical checks, reproducibility and limits</h2>
<p>All stochastic evaluation motions use independent coupled 192/384/768-grid references, balance/support checks, and an oracle-amplitude production 6/12-step consistency audit. Deterministic motion retains the converged RK4-96 reference. Exact observed RGB is unchanged for the 36 paired fits. These are numerical consistency checks, not absolute SDE/mesh/experimental error bounds.</p>
<p>Known eight-node/six-tetrahedron mesh, 30 Gaussians, known appearance/cameras/density/supports, restricted external-force profile, retained independent discrepancy floor, and one sparse schedule. There is no real-video, unknown-geometry/camera or distribution-shift validation. Neither calibrated innovations nor finite-case predictive coverage establishes general physical realism.</p>
<p>The publication includes exact observed pixels, reference states/audits, every fitted solver coordinate, network weights, frozen quantiles, per-case force distributions/scores, all inference flags, CSV and source/artifact hashes. Large precision matrices and full raw chains can be regenerated from the committed source, inputs and seeds. Read docs/CALIBRATED_NOISE_PROTOCOL.md and docs/CALIBRATED_NOISE_RESULTS.md for equations, assumptions and commands.</p>
{table(['Seed','Truth','Model','Gap mm','Maximum standardized error','Stationary','Relative weight ESS'], test_rows)}
<p>Methods: <a href="https://arxiv.org/abs/2107.07511">Angelopoulos and Bates, split conformal prediction</a>; <a href="https://arxiv.org/abs/1903.08008">Vehtari et al., Rhat and ESS diagnostics</a>.</p></html>'''
    (output/"report.html").write_text(document, encoding="utf-8")


def publish(raw, old, posterior_raw, output, source_ref):
    raw, old, posterior_raw, output = map(Path, (raw, old, posterior_raw, output))
    if output.exists(): raise FileExistsError("Choose a new publication directory.")
    torch.set_num_threads(2); commit = source_commit(source_ref)
    comparison = json.loads((raw/"comparison/summary.json").read_text())
    validation = json.loads((raw/"validation/summary.json").read_text())
    previous = json.loads((old/"summary.json").read_text())
    posterior = json.loads((posterior_raw/"summary.json").read_text())
    plan = default_plan()
    network, calibration = verify_validation(raw/"validation", validation, plan)
    if comparison["calibration_sha256_canonical_json"] != validation["calibration_sha256"]: raise ValueError("Different frozen networks.")
    if posterior["settings"]["source_fingerprint_sha256"] != source_fingerprint(): raise ValueError("Posterior audit source changed.")
    audited_case = posterior["settings"]["case"]
    audited_fit = torch.load(raw/"comparison/cases"/audited_case/"fit.pt", weights_only=True)
    audited_record = next(r for r in comparison["cases"] if r["case"] == audited_case)
    if (digest_tensor(audited_fit["mean"]) != posterior["settings"]["original_fit_mean_sha256"]
            or audited_record["observations_sha256"] != posterior["settings"]["observations_sha256"]):
        raise ValueError("Nonlinear audit did not use the identical comparison fit/input.")
    for row in comparison["cases"]:
        original = next(r for r in previous["cases"] if r["case"] == row["case"])
        if row["observations_sha256"] != original["observations_sha256"] or row["initial_state_sha256"] != original["initial_state_sha256"]:
            raise ValueError("Unmatched old/new comparison.")
        if row["mode"] == "floor" and abs(row["scores"]["original_gap"]["position_rmse_m"]-original["scores"]["original_gap"]["position_rmse_m"]) > 1e-10:
            raise ValueError("Unchanged floor baseline failed reproduction.")
    output.mkdir(parents=True)
    publish_comparison(raw/"comparison", output/"comparison", source_ref, expected_plan=plan, write_report=False)
    compact_validation(raw/"validation", output/"validation", validation)
    (output/"posterior").mkdir()
    for name in ("summary.json", "settings.json", "curvature.json"): shutil.copyfile(posterior_raw/name, output/"posterior"/name)
    mode = torch.load(posterior_raw/"mode.pt", weights_only=True)
    write_json(output/"posterior/refined_mode.json", {"coordinates": mode["mean"].tolist(), "fit": mode["record"]})
    write_json(output/"previous_comparison.json", {"summary_sha256_normalized_lf": canonical_hash(old/"summary.json"),
        "source_fingerprint_sha256": previous["source_fingerprint_sha256"], "aggregates": previous["aggregates"],
        "qualification": "archived v0.7 results; exact RGB/initial-state matching verified for all 36 cases"})
    audit = calibration_audit(raw/"comparison", old, output, network)
    figures(output, comparison, previous, validation, posterior)
    report(output, comparison, previous, validation, posterior, calibration, audit)
    write_json(output/"execution.json", {"implementation_source_commit": commit,
        "comparison_command": "physgauss calibrated-noise --plan configs/calibrated_noise_study.json --paired-source runs/neural-marginal-final --output runs/calibrated-noise-final --stage comparison",
        "validation_command": "Frozen sparse-view uncertainty study GitHub workflow, four calibration partitions -> frozen quantiles -> four test partitions -> verified aggregate",
        "reference_helper": "local pilot only: python scripts/prepare_validation_references.py --study runs/calibrated-noise-final/validation",
        "execution": "paired comparison on local CPU; primary validation on CPU CI partitions under recorded environments; exact same source/weights/settings. Local calibration pilot stopped before any test fitting; never mixed into primary cases.",
        "cross_runtime_verification": "reconstructed normalized targets and Fisher factors from exact stored RGB/solver states without optimization; position/SD/score rtol=1e-9 atol=2e-10; normalized NLL absolute tolerance 1e-7",
        "validation_workflow": json.loads((raw/"validation/artifact_download.json").read_text()) if (raw/"validation/artifact_download.json").exists() else None})
    write_json(output/"publication.json", {"implementation_source_commit": commit,
        "numerical_source_fingerprint_sha256": source_fingerprint(), "publisher_sha256_normalized_lf": canonical_hash(Path(__file__)),
        "verification": "all 114 fits; 36 exact old/new matched inputs and initial states; unchanged floor reproduction; calibration/source/plan splits; normalized objectives, all uncertainty scores and frozen quantiles; physical/reference refinement; force total-covariance identities",
        "files_sha256_text_normalized_lf_binary_raw": {p.relative_to(output).as_posix(): canonical_hash(p) for p in sorted(output.rglob("*")) if p.is_file()}})
    print("Verified calibrated publication:", output, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, required=True); parser.add_argument("--old", type=Path, default=Path("examples/neural-marginal"))
    parser.add_argument("--posterior", type=Path, required=True); parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-ref", required=True)
    args = parser.parse_args(); publish(args.raw, args.old, args.posterior, args.output, args.source_ref)
