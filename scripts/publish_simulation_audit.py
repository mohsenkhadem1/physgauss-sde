"""Verify and package the simulation/view audit into an offline HTML report."""
import argparse
import base64
import hashlib
import html
import json
from pathlib import Path
import shutil
import subprocess
import torch
from physgauss.cli import source_fingerprint
from physgauss.study import write_json


def canonical_hash(path):
    path = Path(path)
    content = path.read_bytes()
    if path.suffix in (".json", ".html", ".csv", ".md", ".py"):
        content = content.replace(b"\r\n", b"\n")
    return hashlib.sha256(content).hexdigest()


def publish(raw, output, source_ref):
    root = Path(__file__).resolve().parents[1]
    raw, output = Path(raw).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError("Choose a new publication directory.")
    manifest = json.loads((raw/"manifest.json").read_text())
    if manifest["mode"] != "full" or manifest["source_fingerprint_sha256"] != source_fingerprint():
        raise ValueError("Need a completed full audit from the same numerical source.")
    script = root/"scripts/audit_simulation_views.py"
    if hashlib.sha256(script.read_bytes()).hexdigest() != manifest["audit_script_sha256"]:
        raise ValueError("Audit script changed after execution.")
    commit = subprocess.run(["git", "-C", str(root), "rev-parse", "--verify", "--end-of-options", source_ref+"^{commit}"],
                            capture_output=True, check=True, text=True).stdout.strip()
    original = subprocess.run(["git", "-C", str(root), "show", commit+":scripts/audit_simulation_views.py"],
                              capture_output=True, check=True).stdout
    if original.replace(b"\r\n", b"\n") != script.read_bytes().replace(b"\r\n", b"\n"):
        raise ValueError("Source commit does not contain the executed audit script.")
    physics = json.loads((raw/"physics_audit.json").read_text())
    camera = json.loads((raw/"camera_audit.json").read_text())
    views = json.loads((raw/"view_audit.json").read_text())
    if physics["force_negative_independent_energy_gradient_max_error_N"] > 1e-10:
        raise ValueError("Independent constitutive force check failed.")
    for r in physics["balances"]:
        if max(r["max_free_node_momentum_residual_Ns"], r["max_total_angular_balance_residual_kg_m2_per_s"],
               r["max_internal_impulse_net_Ns"], r["max_internal_impulse_torque_Nms"]) > 1e-12:
            raise ValueError("Momentum/torque balance failed.")
        if r["minimum_det_F_all_steps"] <= 0 or r["max_support_position_error_m"] != 0:
            raise ValueError("Invalid mesh/support state.")
    if camera["rotation_orthogonality_max_error"] > 1e-12 or camera["look_at_target_principal_point_max_error_pixels"] > 1e-10:
        raise ValueError("Camera check failed.")
    for condition in ("no_load", "external_pulse"):
        rows = [r for r in physics["deterministic_refinement"] if r["condition"] == condition]
        if rows[-1]["position_rmse_m_vs_RK4_96"] > .00025 or rows[-1]["RK4_48_vs_96_position_rmse_m"] > 1e-8:
            raise ValueError("High-accuracy reference/refinement tolerance failed.")
        if any(a["position_rmse_m_vs_RK4_96"] <= b["position_rmse_m_vs_RK4_96"] for a, b in zip(rows[:-1], rows[1:])):
            raise ValueError("Deterministic refinement did not reduce error.")
    expected = {(condition, scheme) for condition in ("external_pulse", "pulse_and_stochastic")
                for scheme in views["protocol"]["schemes"]}
    keys = [(r["condition"], r["scheme"]) for r in views["cases"]]
    if len(keys) != 10 or set(keys) != expected:
        raise ValueError("Incomplete view comparison.")
    for condition in ("external_pulse", "pulse_and_stochastic"):
        rows = [r for r in views["cases"] if r["condition"] == condition]
        if len(set(r["shared_initial_state_sha256"] for r in rows)) != 1:
            raise ValueError("Ablation initial states differ.")
        data = json.loads((raw/f"view_inputs_{condition}.json").read_text())
        primary = torch.tensor(data["primary_rgb_all_times"], dtype=torch.float64)
        secondary = torch.tensor(data["secondary_rgb_all_times"], dtype=torch.float64)
        original_six = primary[data["original_six_indices"]]
        if hashlib.sha256(original_six.numpy().tobytes()).hexdigest() != data["original_input_hash"]:
            raise ValueError("Original shared noisy images changed.")
        baseline_trace = next(r["fixed_reference_information"]["force_coefficient_covariance_trace"] for r in rows if r["scheme"] == "six_single")
        for r in rows:
            images = torch.stack([img[frame] for offset in r["camera_offsets_degrees"]
                                  for frame in r["observed_timestamps"] for img in [primary if offset == 0 else secondary]])
            if hashlib.sha256(images.numpy().tobytes()).hexdigest() != r["observations_sha256"]:
                raise ValueError("Ablation observation hash mismatch.")
            if r["fixed_reference_information"]["force_coefficient_covariance_trace"] > baseline_trace+1e-8:
                raise ValueError("Extra data increased fixed-state force covariance.")
            saved = json.loads((raw/"fits"/f"{condition}__{r['scheme']}"/"summary.json").read_text())
            if saved != r:
                raise ValueError("Fit/aggregate summary mismatch.")
    output.mkdir(parents=True)
    for path in raw.rglob("*"):
        if path.is_file() and path.suffix in (".json", ".png", ".csv"):
            destination = output/path.relative_to(raw)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, destination)
    def table(headers, rows):
        return "<table><thead><tr>"+"".join("<th>"+html.escape(str(h))+"</th>" for h in headers)+"</tr></thead><tbody>"+"".join(
            "<tr>"+"".join("<td>"+html.escape(str(v))+"</td>" for v in row)+"</tr>" for row in rows)+"</tbody></table>"
    refinements = table(["substeps per frame", "dt [ms]", "no-load error [mm]", "pulse error [mm]"], [
        [n, f"{50/n:.4f}"]+[f"{1000*next(r['position_rmse_m_vs_RK4_96'] for r in physics['deterministic_refinement'] if r['condition'] == c and r['substeps'] == n):.4f}"
                            for c in ("no_load", "external_pulse")] for n in (3, 6, 12, 24, 48)])
    viewtable = table(["truth", "observations", "images", "gap pose mm", "force RMSE N", "stationary", "local image rank", "local force covariance trace"], [
        [r["condition"], r["scheme"], r["input_image_count"], f"{1000*r['original_gap_position_rmse_m']:.3f}",
         f"{r['conditional_MAP_force_rmse_N']:.5f}", r["fit"]["stationarity_screen_pass"],
         r["fixed_reference_information"]["image_only_local_rank"], f"{r['fixed_reference_information']['force_coefficient_covariance_trace']:.3f}"] for r in views["cases"]])
    def embedded(name, caption):
        encoded = base64.b64encode((output/name).read_bytes()).decode()
        return f'<figure><img src="data:image/png;base64,{encoded}" alt="{html.escape(caption)}"><figcaption>{html.escape(caption)}</figcaption></figure>'
    force_error = physics["force_negative_independent_energy_gradient_max_error_N"]
    document = f"""<!doctype html><html lang="en"><meta charset="utf-8"><title>Simulation and view coverage audit</title>
<style>body{{font:16px/1.5 system-ui,sans-serif;max-width:1250px;margin:32px auto;padding:0 24px;color:#202b35;background:#f9fafb}}h1,h2{{line-height:1.2}}img{{width:100%;height:auto}}figure{{margin:24px 0}}figcaption{{font-size:14px;color:#536371}}table{{width:100%;border-collapse:collapse;background:white;font-size:13px}}td,th{{padding:8px;border-bottom:1px solid #ddd;text-align:left}}th{{background:#e4edf1}}.notice{{padding:18px;background:#fff0d5;border-left:5px solid #b97721}}</style>
<h1>What do the images and simulation actually support?</h1>
<div class="notice"><strong>The mechanics formulas and discrete balances pass, but the historical integration step is too coarse for millimetre-level physical claims.</strong>
Its position error against a converged independent RK4 solution is about 3.3 mm (up to 11.5 mm at a node/time). More views address a different limitation. The renderer still has depth-order discontinuities.</div>
<h2>Actual images and reconstruction</h2>
<p>Native images are 24×24 pixels from a coloured 30-Gaussian volumetric proxy. Tiles are enlarged by nearest-neighbor display; no detail was invented. Input noise is clipped only for display, not inference. All geometry, appearance, density and cameras are known in these tests.</p>
{embedded('input_reconstruction.png', 'Original sparse input, noiseless simulation truth, fitted MAP and absolute RGB error. The central timestamp has no input image.')}
{embedded('unseen_views.png', 'The same missing physical timestamp from four camera offsets. These novel-view truth images were not supplied to the original fit.')}
{embedded('mesh_camera_coverage.png', 'Actual tetrahedral geometry and camera coverage. One camera moves through a 100° azimuth arc at a single elevation; horizontal field of view is 33.72°.')}
<h2>Independent simulation audit</h2>
<p>Force agrees with an independently differentiated scalar strain energy to {force_error:.2g} N. Free-node momentum, internal impulse torque and total angular balance including support reactions agree at floating-point precision. Supports stay fixed and all audited determinants remain positive. The RK4 reference uses the same FEM force law but an independent fourth-order integration routine.</p>
{refinements}
<p>RK4 at 48 versus 96 substeps per frame differs by about 2.1×10⁻¹⁰ m in coordinate RMSE. The benchmark step is below the linearized stability limit, but numerical stability is not numerical accuracy. Coupled Brownian refinements also reduce error. This is not mesh convergence, experimental validation or a thermal-noise validation.</p>
{embedded('time_refinement.png', 'Semi-implicit Euler time-step convergence against independent RK4 and deterministic work–energy defects.')}
<p>Two new deterministic high-accuracy reference inputs and trajectories are included under high_accuracy_reference/. They retain the old six-image noise realization while replacing coarse simulated motion with converged RK4. These new inputs were not used in the view ablation or historical fits.</p>
<h2>Does adding views help?</h2>
<p>Two original sequences are re-fitted under five observation schemes with a common six-image-only initialization, unchanged force prior and unchanged discrepancy process. Original six noisy images are retained exactly. Twelve frames from one camera and six simultaneous two-camera pairs both use twelve images. Additional frames can observe the former gap, so its pose error is called original-gap error, not held-out error.</p>
{viewtable}
{embedded('view_ablation.png', 'Finite nonlinear fit errors versus local information at an identical reference state. Failed stationarity flags limit performance interpretation.')}
<p>The image-only rank concerns the exposed trajectory coordinates with known geometry/appearance. Six times provide at most 63 local pose combinations, versus 728 state coordinates; mechanics/prior information is essential. A second camera improves conditioning but cannot increase the pose dimension at a fixed time. More times also constrain additional temporal states. Fixed-state covariance reduction is a local linear result and does not prove reliable nonlinear identification.</p>
<h2>Audit conclusion</h2>
<p>There are three separate issues: sparse observations and known-scene assumptions; insufficient time-integration accuracy in the historical toy setup; and nonsmooth visibility in the renderer. Increasing image count cannot validate the simulator or repair its numerical visibility objective. Preserve historical results as discretized-model experiments, use converged reference data, then address integration/visibility before scientific physical-identification or calibration claims.</p>
<p>Raw JSON, exact ablation inputs, fitted paths and source provenance accompany this offline report. Full raw Gaussian precisions remain in ignored local fit checkpoints. No public/real-world dataset or unknown-geometry reconstruction was evaluated.</p></html>"""
    (output/"report.html").write_text(document, encoding="utf-8")
    write_json(output/"publication.json", {"audit_source_commit": commit,
        "numerical_source_fingerprint_sha256": manifest["source_fingerprint_sha256"],
        "audit_script_sha256": manifest["audit_script_sha256"],
        "verification": "constitutive/constraint/balance gates; time-refinement accuracy; exact shared RGB hashes and initial states; local added-information covariance; all 10 fit summaries",
        "files_sha256_text_normalized_lf_binary_raw": {p.relative_to(output).as_posix(): canonical_hash(p) for p in sorted(output.rglob("*")) if p.is_file()}})
    print("Verified and published audit:", output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-ref", required=True)
    args = parser.parse_args()
    publish(args.raw, args.output, args.source_ref)
