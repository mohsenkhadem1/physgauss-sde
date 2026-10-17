"""Verify and publish the complete independent-reference comparison offline."""
import argparse
import base64
import hashlib
import html
import json
from pathlib import Path
import shutil
import subprocess
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from physgauss.accurate_study import (construct, default_plan, digest_tensor, initial_state,
                                      nested_initial, inputs, observations)
from physgauss.cli import source_fingerprint
from physgauss.study import write_json
from physgauss.benchmark import Config
from physgauss.forces import ForceConfig, evaluate_force_posterior
from physgauss.marginal_forces import MarginalConfig, TrajectoryForceModel
from physgauss.continuous_render import ContinuousGaussianRenderer


ROOT = Path(__file__).resolve().parents[1]
CONDITIONS = ("no_load", "external_pulse")
METHODS = ("marginalized", "joint", "alternating")
COLORS = {"marginalized": "#007c91", "joint": "#bd5a14", "alternating": "#8656a5"}


def canonical_hash(path):
    data = path.read_bytes()
    if path.suffix in (".json", ".html", ".md", ".py"):
        data = data.replace(b"\r\n", b"\n")
    return hashlib.sha256(data).hexdigest()


def source_commit(ref):
    commit = subprocess.run(["git", "rev-parse", "--verify", "--end-of-options", ref+"^{commit}"],
                             cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
    for path in sorted((ROOT/"src").rglob("*.py")):
        relative = path.relative_to(ROOT).as_posix()
        saved = subprocess.run(["git", "show", commit+":"+relative], cwd=ROOT,
                               capture_output=True, check=True).stdout
        if saved.replace(b"\r\n", b"\n") != path.read_bytes().replace(b"\r\n", b"\n"):
            raise ValueError("Numerical source differs from committed implementation: "+relative)
    return commit


def verify(raw, summary):
    plan = summary["plan"]
    if plan != default_plan():
        raise ValueError("Publication requires the declared full comparison plan.")
    if summary["source_fingerprint_sha256"] != source_fingerprint():
        raise ValueError("Numerical source differs from the experiment.")
    jobs = [(v, "six_single", m) for v in plan["variants"]
            for m in (plan["methods"] if v == "corrected" else ["marginalized"])]
    jobs += [("corrected", s, "marginalized") for s in plan["schemes"] if s != "six_single"]
    expected = {f"{c}__{v}__{s}__{m}" for c in CONDITIONS for v, s, m in jobs}
    if len(summary["cases"]) != len(expected) or {r["case"] for r in summary["cases"]} != expected:
        raise ValueError("Incomplete or duplicated comparison cases.")
    common = initial_state(plan)
    common_model = construct(plan, "corrected", "six_single")
    shared_x = common_model.trajectory(common[None]).x.detach()
    references = ROOT/"examples/simulation-view-audit/high_accuracy_reference"
    for condition in CONDITIONS:
        data = inputs(plan, references/f"{condition}.json", common_model)
        saved = json.loads((raw/f"inputs_{condition}.json").read_text())
        if summary["reference_hashes_normalized_lf"][condition] != data["reference_sha256_normalized_lf"]:
            raise ValueError("Independent RK4 reference changed.")
        if saved["original_noise_sha256"] != data["original_noise_sha256"]:
            raise ValueError("Original noise residuals changed.")
        for kind in ("clean", "noisy"):
            for offset in (0, 100, 200):
                torch.testing.assert_close(torch.tensor(saved[kind+"_rgb"][str(offset)], dtype=torch.float64),
                                           data[kind][offset], atol=0., rtol=0.)
        for row in (r for r in summary["cases"] if r["condition"] == condition):
            destination = raw/"cases"/row["case"]
            if json.loads((destination/"summary.json").read_text()) != row:
                raise ValueError("Aggregate and fit summaries differ.")
            if row["source_fingerprint_sha256"] != summary["source_fingerprint_sha256"]:
                raise ValueError("Mixed numerical sources in case records.")
            model = construct(plan, row["variant"], row["scheme"])
            observed = observations(model, data)
            start = nested_initial(model, common, plan)
            if digest_tensor(observed.rgb) != row["observations_sha256"] or digest_tensor(start) != row["initial_state_sha256"]:
                raise ValueError("Observation or initialization mismatch.")
            if digest_tensor(shared_x) != row["initial_frame_positions_sha256"]:
                raise ValueError("Shared initial frame poses differ.")
            fit = torch.load(destination/"fit.pt", weights_only=True)
            if fit["record"] != row["fit"]:
                raise ValueError("Checkpoint and numerical record differ.")
            torch.testing.assert_close(model.trajectory(fit["mean"][None]).x[0],
                torch.tensor(json.loads((destination/"point.json").read_text())["positions_m"], dtype=torch.float64),
                atol=0., rtol=0.)
            if row["fit"]["schur_precision_relative_max_error"] > 1e-10 or row["fit"]["profile_marginal_identity_abs_error"] > 1e-8:
                raise ValueError("Force elimination identities failed.")
            value = float(model.objective(fit["mean"][None], observed)[0])
            if abs(value-row["fit"]["marginal_objective"]) > 1e-8:
                raise ValueError("Saved fit does not reproduce the density.")
            force_path = destination/"fields.json"
            if force_path.exists():
                fields = json.loads(force_path.read_text())
                cov = torch.tensor(fields["whitened_force_coefficient_covariance"], dtype=torch.float64)
                conditional = torch.tensor(fields["conditional_coefficient_covariance"], dtype=torch.float64)
                between = torch.tensor(fields["between_trajectory_coefficient_covariance"], dtype=torch.float64)
                torch.testing.assert_close(cov, conditional+between, atol=1e-12, rtol=1e-12)
                torch.testing.assert_close(conditional, model.elimination.conditional_covariance, atol=1e-12, rtol=1e-12)
                if float(torch.linalg.eigvalsh(cov).min()) <= 0:
                    raise ValueError("Indefinite force posterior covariance.")
                query = evaluate_force_posterior(fields, model.times)
                torch.testing.assert_close(query["mean_total_N"],
                    torch.tensor(fields["external_total_mean_N"], dtype=torch.float64), atol=1e-12, rtol=1e-12)
                torch.testing.assert_close(query["mean_nodal_N"].sum(1), query["mean_total_N"], atol=1e-12, rtol=1e-12)
    audit = summary["numerical_audit"]
    for row in audit["integration"]:
        if row["substeps"] == 6 and row["position_rmse_m"] > .00015:
            raise ValueError("Integration accuracy gate failed.")
    for row in audit["rendering"]:
        limit = 3e-5 if row["bins"] == 128 else 2e-6
        if row["rgb_rmse"] > limit:
            raise ValueError("Depth quadrature accuracy gate failed.")


def select(summary, condition, variant="corrected", scheme="six_single", method="marginalized"):
    return next(r for r in summary["cases"] if (r["condition"], r["variant"], r["scheme"], r["method"]) ==
                (condition, variant, scheme, method))


def savefig(fig, output, name):
    fig.savefig(output/name, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def depth_probe(output):
    """Post-fit continuity probe at a published historical stalled state.

    This diagnostic does not select any setting, initialization, or fit. Hard
    sorting is rounding-sensitive; its exact jump pattern may vary by platform.
    """
    path = ROOT/"examples/marginal-force-study/cases/external_pulse__4101/marginalized/point_estimate.json"
    record = json.loads(path.read_text())
    model = TrajectoryForceModel(Config(**record["config"]), ForceConfig(**record["force_prior"]),
                                 MarginalConfig(**record["trajectory_model"]))
    state = torch.tensor(record["state_coordinates"], dtype=torch.float64)
    direction = torch.randn(state.shape, dtype=state.dtype, generator=torch.Generator().manual_seed(6610))
    direction /= direction.norm()
    indices = torch.tensor([0, 3, 6, 14, 17, 20])
    rows = []
    for name, renderer in (("hard", model.renderer), ("continuous", ContinuousGaussianRenderer(model.mesh, 24))):
        for step in (1e-2, 1e-4, 1e-6, 1e-8, 1e-10):
            plus = renderer(model.trajectory((state+step*direction)[None]).x[:, indices], model.eye[indices], model.rotation[indices])
            minus = renderer(model.trajectory((state-step*direction)[None]).x[:, indices], model.eye[indices], model.rotation[indices])
            difference = float((plus-minus).square().mean().sqrt())
            rows.append({"renderer": name, "coordinate_step": step, "symmetric_rgb_rmse": difference})
            if name == "continuous" and difference/step > .01:
                raise ValueError("Continuous renderer failed the declared continuity probe bound.")
    write_json(output/"depth_continuity.json", {
        "historical_point_sha256_normalized_lf": canonical_hash(path), "direction_seed": 6610,
        "scope": "post-fit renderer-only numerical probe; no retuning; exact hard-sort jumps depend on rounding/platform",
        "rows": rows})
    fig, ax = plt.subplots(figsize=(7, 4))
    for name, color in (("hard", "#bd5a14"), ("continuous", "#007c91")):
        selected = [r for r in rows if r["renderer"] == name]
        ax.loglog([r["coordinate_step"] for r in selected], [r["symmetric_rgb_rmse"] for r in selected],
                  "o-", label=name, color=color)
    ax.set(xlabel="Symmetric coordinate perturbation h", ylabel="RGB RMSE between +h and -h",
           title="Historical stalled pose: does image change vanish with step size?")
    ax.legend(); ax.grid(alpha=.15); fig.tight_layout(); savefig(fig, output, "depth_continuity.png")


def force_basis_audit(output, summary):
    model = construct(summary["plan"], "corrected", "six_single")
    times = torch.linspace(0., 1., 1001, dtype=torch.float64)
    design = model.force_prior.config.total_force_std_n*model.force_prior.features(times)
    rows = []
    for condition in CONDITIONS:
        strength = torch.exp(-.5*((times-.5)/.08).square()) if condition == "external_pulse" else torch.zeros_like(times)
        truth = strength[:, None]*model.mesh.mass[model.free].sum()*times.new_tensor([.45, 0., -.30])
        coefficient = torch.linalg.lstsq(design, truth).solution
        projection = design@coefficient
        rows.append({"condition": condition, "total_force_rms_N": float(truth.square().mean().sqrt()),
                     "oracle_basis_projection_rmse_N": float((truth-projection).square().mean().sqrt()),
                     "evaluation_grid_points": len(times)})
    write_json(output/"force_basis_audit.json", {"rows": rows,
        "scope": "evaluation-only least-squares oracle on a dense time grid; no truth coefficients or basis tuning enter inference; RMSE not directly identical to the 21-frame fitted score"})


def figures(raw, output, summary):
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    audit = summary["numerical_audit"]
    historical = json.loads((ROOT/"examples/simulation-view-audit/physics_audit.json").read_text())
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for condition, color in zip(CONDITIONS, ("#007c91", "#bd5a14")):
        new = [r for r in audit["integration"] if r["condition"] == condition]
        old = [r for r in historical["deterministic_refinement"] if r["condition"] == condition]
        axes[0].loglog([r["substeps"] for r in old], [r["position_rmse_m_vs_RK4_96"]*1000 for r in old],
                       "o--", color=color, alpha=.6, label=condition+" Euler")
        axes[0].loglog([r["substeps"] for r in new], [r["position_rmse_m"]*1000 for r in new],
                       "s-", color=color, label=condition+" staggered")
    axes[0].axvline(6, color="gray", lw=.8); axes[0].set(xlabel="Substeps per frame", ylabel="Coordinate RMSE vs RK4-96 [mm]",
                                                       title="Deterministic time accuracy")
    axes[0].legend(fontsize=8)
    rows = [r for r in audit["rendering"] if r["condition"] == "external_pulse" and r["bins"] < 1024]
    axes[1].semilogy([r["bins"] for r in rows], [r["rgb_rmse"] for r in rows], "o-", color="#007c91")
    axes[1].axhline(.015, ls="--", color="#bd5a14", label="Observation noise SD")
    axes[1].set(xlabel="Depth slabs", ylabel="RGB RMSE vs 1024 slabs", title="Depth quadrature (pulse reference)")
    axes[1].legend(); fig.tight_layout(); savefig(fig, output, "numerical_accuracy.png")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, condition in zip(axes, CONDITIONS):
        rows = [select(summary, condition, variant=v) for v in ("legacy", "integrator_only", "renderer_only", "corrected")]
        values = [r["scores"]["original_gap"]["position_rmse_m"]*1000 for r in rows]
        ax.bar(range(4), values, color=["#a1adb5", "#7799a9", "#e5a26b", "#007c91"])
        ax.set_xticks(range(4), ["Legacy", "Integrator\nonly", "Renderer\nonly", "Both"])
        for n, (row, value) in enumerate(zip(rows, values)):
            ax.text(n, value, f"{value:.2f}\n"+("pass" if row["fit"]["stationarity_screen_pass"] else "FAIL"), ha="center", va="bottom", fontsize=8)
        ax.set(ylabel="Original-gap position RMSE [mm]", title=condition, ylim=(0, max(values)*1.27))
    fig.suptitle("Component controls: marginalized solver, identical reference images")
    fig.tight_layout(); savefig(fig, output, "component_comparison.png")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, condition in zip(axes, CONDITIONS):
        for method in METHODS:
            row = select(summary, condition, method=method)
            history = row["fit"]["history"]
            ax.semilogy(range(1, len(history)+1), [r["step_decrement"] for r in history], "o-", ms=3,
                        color=COLORS[method], label=method)
        ax.axhline(.05, color="gray", ls="--", label="Step threshold")
        ax.set(xlabel="Iteration", ylabel="Method step decrement", title=condition)
        ax.legend(fontsize=8)
    fig.suptitle("Solver convergence (final common marginal screen is reported separately)")
    fig.tight_layout(); savefig(fig, output, "solver_convergence.png")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    schemes = summary["plan"]["schemes"]
    labels = ["6 times\n1 camera", "12 times\n1 camera", "6 times\n2 cameras", "21 times\n1 camera", "21 times\n2 cameras"]
    for ax, condition in zip(axes, CONDITIONS):
        rows = [select(summary, condition, scheme=s) for s in schemes]
        values = [r["scores"]["original_gap"]["position_rmse_m"]*1000 for r in rows]
        ax.bar(range(5), values, color="#007c91")
        for n, (row, value) in enumerate(zip(rows, values)):
            ax.text(n, value, f"{value:.2f}\n"+("pass" if row["fit"]["stationarity_screen_pass"] else "FAIL"), ha="center", va="bottom", fontsize=8)
        ax.set_xticks(range(5), labels, fontsize=8)
        ax.set(ylabel="Original-gap position RMSE [mm]", title=condition, ylim=(0, max(values)*1.3))
    fig.suptitle("Adding views to the corrected marginalized model; dense schedules observe the original gap")
    fig.tight_layout(); savefig(fig, output, "view_comparison.png")

    fig, axes = plt.subplots(2, 3, figsize=(12, 6), sharex=True)
    for row_index, condition in enumerate(CONDITIONS):
        row = select(summary, condition)
        fields = json.loads((raw/"cases"/row["case"]/"fields.json").read_text())
        t = np.array(fields["times_s"])
        mean, lo, hi, truth = [np.array(fields[k]) for k in ("external_total_mean_N", "external_total_q05_N", "external_total_q95_N", "truth_external_total_N_evaluation_only")]
        for dim, ax in enumerate(axes[row_index]):
            ax.fill_between(t, lo[:, dim], hi[:, dim], color="#007c91", alpha=.2)
            ax.plot(t, mean[:, dim], color="#007c91", label="Inferred local mean")
            ax.plot(t, truth[:, dim], "--", color="black", label="True force")
            ax.axvspan(.3, .7, color="gray", alpha=.1)
            ax.set(title=condition+" / "+"xyz"[dim], xlabel="Time [s]", ylabel="Total force [N]")
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("Marginalized force distribution; local 90% bands, not calibrated")
    fig.tight_layout(); savefig(fig, output, "force_distribution.png")

    condition = "external_pulse"
    data = json.loads((raw/f"inputs_{condition}.json").read_text())
    old = select(summary, condition, variant="legacy")
    new = select(summary, condition)
    old_images = torch.load(raw/"cases"/old["case"]/"images.pt", weights_only=True)
    new_images = torch.load(raw/"cases"/new["case"]/"images.pt", weights_only=True)
    times = [0, 3, 6, 10, 14, 17, 20]
    tiles = [np.array(data["noisy_rgb"]["0"]), np.array(data["clean_rgb"]["0"]),
             old_images[0].numpy(), new_images[0].numpy(), np.array(data["clean_rgb"]["200"]),
             old_images[200].numpy(), new_images[200].numpy()]
    names = ["Sparse input", "RK4 reference", "Legacy fit", "Corrected fit", "Unseen reference", "Unseen legacy", "Unseen corrected"]
    fig, axes = plt.subplots(7, 7, figsize=(12, 11))
    for i in range(7):
        for j, frame in enumerate(times):
            ax = axes[i, j]
            if i == 0 and frame == 10:
                ax.set_facecolor("#e4e9eb"); ax.text(.5, .5, "Missing", ha="center", va="center", transform=ax.transAxes)
            else:
                ax.imshow(np.clip(tiles[i][frame], 0, 1), interpolation="nearest")
            ax.set_xticks([]); ax.set_yticks([])
            for spine in ax.spines.values(): spine.set_visible(False)
            if j == 0: ax.set_ylabel(names[i], fontsize=9)
            if i == 0: ax.set_title(f"t={frame*.05:.2f} s")
    fig.suptitle("Actual 24 x 24 images (enlarged without adding detail); pulse sequence\nUnseen camera +200 degrees was never fitted")
    fig.tight_layout(); savefig(fig, output, "image_comparison.png")


def table(headers, rows):
    def cells(tag, values): return "".join(f"<{tag}>{html.escape(str(v))}</{tag}>" for v in values)
    return "<table><thead><tr>"+cells("th", headers)+"</tr></thead><tbody>"+"".join("<tr>"+cells("td", r)+"</tr>" for r in rows)+"</tbody></table>"


def report(output, summary):
    primary = [r for r in summary["cases"] if r["variant"] == "corrected" and r["scheme"] == "six_single"]
    stationary = sum(r["fit"]["stationarity_screen_pass"] for r in primary)
    audit_pass = sum(r["local_gaussian_audit"]["screen_pass"] for r in primary)
    headers = ["Truth", "Method", "Gap mm", "Withheld mm", "Force RMSE N", "Final decrement", "Stationary", "Weight rel. ESS", "Local 90% coverage (withheld)"]
    rows = [[r["condition"], r["method"], f"{1000*r['scores']['original_gap']['position_rmse_m']:.3f}",
        f"{1000*r['scores']['withheld']['position_rmse_m']:.3f}", f"{r['force_scores']['rmse_N']:.5f}",
        f"{r['fit']['marginal_gauss_newton_decrement']:.4g}", r["fit"]["stationarity_screen_pass"],
        f"{r['local_gaussian_audit']['relative_weight_ess']:.3f}", f"{r['scores']['withheld']['component_90_coverage']:.1%}"] for r in primary]
    component_rows = [[r["condition"], r["variant"], f"{1000*r['scores']['original_gap']['position_rmse_m']:.3f}",
        f"{r['fit']['marginal_gauss_newton_decrement']:.3g}", r["fit"]["stationarity_screen_pass"]]
        for r in summary["cases"] if r["method"] == "marginalized" and r["scheme"] == "six_single"]
    view_rows = [[r["condition"], r["scheme"], r["observed_slots"], f"{1000*r['scores']['original_gap']['position_rmse_m']:.3f}",
        f"{1000*r['scores']['original_gap']['component_90_mean_width_m']:.2f}",
        f"{r['scores']['original_gap']['component_90_coverage']:.1%}",
        "none" if r["scores"]["withheld"] is None else f"{1000*r['scores']['withheld']['position_rmse_m']:.3f}",
        f"{r['scores']['camera_200_clean_psnr_db']:.2f}", r["fit"]["stationarity_screen_pass"]]
        for r in summary["cases"] if r["variant"] == "corrected" and r["method"] == "marginalized"]
    def embedded(name, caption):
        content = base64.b64encode((output/name).read_bytes()).decode()
        return f'<figure><img src="data:image/png;base64,{content}" alt="{html.escape(caption)}"><figcaption>{html.escape(caption)}</figcaption></figure>'
    document = f'''<!doctype html><html lang="en"><meta charset="utf-8"><title>Accurate reference comparison</title>
<style>body{{font:16px/1.5 system-ui;max-width:1250px;margin:32px auto;padding:0 24px;color:#20313b;background:#f9fafb}}img{{width:100%;height:auto}}figure{{margin:24px 0}}figcaption{{color:#536371;font-size:14px}}table{{width:100%;border-collapse:collapse;font-size:12px;background:white}}td,th{{padding:8px;border-bottom:1px solid #ddd;text-align:left}}th{{background:#e4edf1}}.notice{{background:#e5f2f4;border-left:5px solid #007c91;padding:18px}}h1,h2{{line-height:1.2}}</style>
<h1>Accurate dynamics, continuous visibility, and sparse views</h1>
<div class="notice">The corrected six-substep integrator has about <strong>0.130 mm coordinate RMSE</strong> against independent RK4-96, versus about 3.3 mm for historical Euler-3. Smooth overlapping depth integration replaces whole-splat depth sorting. This is numerical validation on two known-scene toy sequences, not real-world physical validation.</div>
<h2>Numerical checks</h2>
<p>Halving the staggered time step reduces deterministic position error by about four. Its parameter-dependent density determinant is included so force marginalization remains normalized. Production 128-slab rendering differs from 1024 slabs by about 2.15e-5 RGB RMSE, and the 512-slab image reference by about 1.0e-6; observation noise SD is 0.015.</p>
{embedded('numerical_accuracy.png', 'Independent time refinement and depth quadrature; integration error refers to exact truth parameters and analytic true loading, not fitted reconstruction error.')}
{embedded('depth_continuity.png', 'Post-fit probe at the published historical stalled pulse pose. Continuous image differences decrease with perturbation size; hard depth sorting produces rounding-sensitive jumps. No fitting settings were chosen from this probe.')}
<h2>Actual images</h2>
<p>Images are 24 x 24 pixels. Native tiles are enlarged without inventing detail. Display clips noisy RGB to [0,1]; fitting uses unclipped values. Reference images are re-rendered from RK4-96 positions using the new continuous renderer. Original six-view noise residuals are reused. All controls see the same new images.</p>
{embedded('image_comparison.png', 'Sparse observed input, independent reference motion, and legacy/corrected reconstructions. The t=0.5 s input is absent; the +200-degree camera is evaluated only.')}
<h2>Three solvers under the corrected model</h2>
<p>All solvers share the same initialization, force prior, discrepancy model, corrected discretization and renderer. The budget is 20 iterations. {stationary}/6 primary corrected fits pass the common marginal stationarity screen; {audit_pass}/6 pass the independent local-Gaussian importance-weight screen. A passed screen is not posterior calibration. Force elimination is exact conditional on trajectory; trajectory inference remains a local approximation.</p>
{table(headers, rows)}
{embedded('solver_convergence.png', 'Different solver step metrics are shown by iteration. The table applies one common final marginal metric to all three methods.')}
{embedded('force_distribution.png', 'Fixed-profile external force posteriors; local uncertainty combines conditional and between-trajectory covariance. Bands do not establish calibrated nonlinear uncertainty.')}
<h2>Which numerical fix helps?</h2>
<p>These controls use marginalized inference and identical continuous-reference images. Legacy and integrator-only rows retain the hard renderer and therefore deliberately mismatch image formation. Absolute density values across different time grids or renderers are not comparable evidence; compare reconstruction and within-model optimization diagnostics.</p>
{table(['Truth', 'Control', 'Gap mm', 'Final decrement', 'Stationary'], component_rows)}
{embedded('component_comparison.png', 'Component controls; FAIL means the local optimizer stationarity screen did not pass.')}
<h2>Do more observations help?</h2>
<p>All view schedules use the corrected marginalized model and the same prior-mean rollout initialization. Added views have known cameras. Twelve single-camera timestamps and six two-camera pairs both use twelve images. Dense schedules observe the former missing gap: their gap error is a common-window score, not a withheld-image claim.</p>
{table(['Truth', 'Schedule', 'Images', 'Original gap mm', 'Local band width mm', 'Local coverage (gap)', 'Withheld mm', 'Unseen-camera PSNR dB', 'Stationary'], view_rows)}
{embedded('view_comparison.png', 'Sparse spatial/temporal views versus dense observations, on independent accurate motion.')}
<h2>Scope and handover</h2>
<p>Two deterministic sequences, one tiny 8-node/6-tetrahedron mesh, fixed force profile and one noise realization. Geometry, appearance, density, supports and camera poses are known. No real video, unknown-geometry completion, arbitrary force location, mesh convergence, contact, or calibrated uncertainty is established. The stochastic reference remains uncertified. The renderer is a continuous projected-footprint volume surrogate, not exact 3D Gaussian ray tracing. Stored staggered velocities produce finite-difference integer-time estimates rather than exact SDE velocities.</p>
<p>The force prior is a finite five-knot Matérn inducing projection, not an unrestricted GP. Its temporal basis need not reproduce the analytic narrow pulse exactly. An evaluation-only dense-grid oracle projection is saved in force_basis_audit.json; it never initializes or tunes a fit.</p>
<p>Compact inputs, fitted solver coordinates, point paths, force distributions, records and source/environment provenance accompany this offline report. Full raw precisions remain in ignored local checkpoints and can be regenerated from the committed plan. Historical results have not been overwritten.</p></html>'''
    (output/"report.html").write_text(document, encoding="utf-8")


@torch.no_grad()
def publish(raw, output, source_ref):
    torch.set_num_threads(2)
    raw, output = Path(raw).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError("Choose a new publication directory.")
    summary = json.loads((raw/"summary.json").read_text())
    commit = source_commit(source_ref)
    verify(raw, summary)
    output.mkdir(parents=True)
    for path in raw.rglob("*.json"):
        destination = output/path.relative_to(raw)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, destination)
    for row in summary["cases"]:
        fit = torch.load(raw/"cases"/row["case"]/"fit.pt", weights_only=True)
        write_json(output/"cases"/row["case"]/"solver_state.json", {
            "coordinates": fit["mean"].tolist(), "state_dimension": len(fit["mean"]),
            "conditional_force_coefficient_mean": fit["conditional_force_mean"].tolist(),
            "qualification": "point coordinates only; full raw precision is regenerated by the fixed plan"})
    depth_probe(output)
    force_basis_audit(output, summary)
    figures(raw, output, summary)
    report(output, summary)
    write_json(output/"publication.json", {"implementation_source_commit": commit,
        "numerical_source_fingerprint_sha256": source_fingerprint(),
        "publication_script_sha256_normalized_lf": canonical_hash(Path(__file__)),
        "verification": "20 cases; exact regenerated RGB/noise hashes; common prior-only initialization; saved objective and trajectory reproduction; Schur/profile identities; force total covariance and positivity; integration/render quadrature gates",
        "files_sha256_text_normalized_lf_binary_raw": {p.relative_to(output).as_posix(): canonical_hash(p) for p in sorted(output.rglob("*")) if p.is_file()}})
    print("Verified and published:", output, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-ref", required=True)
    args = parser.parse_args()
    publish(args.raw, args.output, args.source_ref)
