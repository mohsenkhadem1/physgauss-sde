"""Self-contained comparison report; all uncertainty qualified explicitly."""
import base64
import csv
import html
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


COLORS = {"alternating": "#b66a16", "joint": "#8064ad", "marginalized": "#167b8c"}


def make_marginal_report(output, summary):
    output = Path(output)
    pairs = list(dict.fromkeys((r["condition"], r["seed"]) for r in summary["cases"]))
    fig, axes = plt.subplots(len(pairs), 2, figsize=(12, 3.1*len(pairs)), squeeze=False, constrained_layout=True)
    for row, (condition, seed) in enumerate(pairs):
        records = [r for r in summary["cases"] if (r["condition"], r["seed"]) == (condition, seed)]
        for r in records:
            method = r["method"]
            fields = json.loads((output/"cases"/f"{condition}__{seed}"/method/"fields.json").read_text())
            t = np.asarray(fields["times_s"])
            mean = np.asarray(fields["external_total_mean_N"])
            lo, hi = np.asarray(fields["external_total_q05_N"]), np.asarray(fields["external_total_q95_N"])
            truth = np.asarray(fields["truth_external_total_N_evaluation_only"])
            for col, component in enumerate((0, 2)):
                ax = axes[row, col]
                if r is records[0]:
                    ax.plot(t, truth[:, component], color="#222222", linewidth=2, label="truth")
                ax.plot(t, mean[:, component], color=COLORS[method], label=method, linewidth=1.4)
                ax.fill_between(t, lo[:, component], hi[:, component], color=COLORS[method], alpha=.10)
                observed = r["config"]["observed_frames"]
                left, right = max(zip(observed[:-1], observed[1:]), key=lambda p: p[1]-p[0])
                duration = r["config"]["duration"]
                denominator = r["config"]["frames"]-1
                if r is records[0]:
                    ax.axvspan(left/denominator*duration, right/denominator*duration, color="grey", alpha=.12)
                ax.set_title(f"{condition}, seed {seed} · {'x' if component == 0 else 'z'} force")
                ax.set_xlabel("time [s]"); ax.set_ylabel("unknown smooth load [N]")
                ax.grid(alpha=.2)
                if row == 0 and col == 0:
                    ax.legend(fontsize=8, ncol=2)
    fig.savefig(output/"comparison.png", dpi=150); plt.close(fig)
    columns = ["condition", "seed", "method", "point_rmse_mm", "force_rmse_N", "decrement", "stationary",
               "relative_weight_ess", "optimization_seconds", "schur_error"]
    rows = []
    for r in summary["cases"]:
        rows.append({"condition": r["condition"], "seed": r["seed"], "method": r["method"],
            "point_rmse_mm": 1000*r["point_withheld_position_rmse_m"], "force_rmse_N": r["force_scores"]["rmse_N"],
            "decrement": r["fit"]["marginal_gauss_newton_decrement"], "stationary": r["fit"]["stationarity_screen_pass"],
            "relative_weight_ess": r["local_gaussian_audit"]["relative_weight_ess"],
            "optimization_seconds": r["fit"]["optimization_seconds"],
            "schur_error": r["fit"]["schur_precision_relative_max_error"]})
    with (output/"scores.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns); writer.writeheader(); writer.writerows(rows)
    def table(headers, contents):
        return "<table><thead><tr>"+"".join("<th>"+html.escape(str(x))+"</th>" for x in headers)+"</tr></thead><tbody>"+"".join(
            "<tr>"+"".join("<td>"+html.escape(str(x))+"</td>" for x in row)+"</tr>" for row in contents)+"</tbody></table>"
    measured = table(["condition", "solver", "withheld MAP mm", "force RMSE N", "GN decrement", "stationary", "relative weight ESS", "fit seconds"],
        [[r["condition"], r["method"], f"{r['point_rmse_mm']:.3f}", f"{r['force_rmse_N']:.5f}", f"{r['decrement']:.3g}",
          r["stationary"], f"{r['relative_weight_ess']:.3g}", f"{r['optimization_seconds']:.2f}"] for r in rows])
    old = summary["historical_reference"]
    historical = table(["condition", "old model", "withheld posterior-mean mm", "force RMSE N", "MCMC checks pass"],
        [[r["condition"], r["method"], f"{1000*r['withheld_position_rmse_m']:.3f}", f"{r['force_rmse_N']:.5f}", r["all_mcmc_checks_pass"]]
         for r in old["cases"]]) if old["cases"] else "<p>No historical reference supplied for this run.</p>"
    embedded = base64.b64encode((output/"comparison.png").read_bytes()).decode()
    n_stationary = sum(r["stationary"] for r in rows)
    n_audit = sum(r["local_gaussian_audit"]["screen_pass"] for r in summary["cases"])
    document = f"""<!doctype html><html lang="en"><meta charset="utf-8"><title>Force marginalization comparison</title>
<style>body{{font:16px/1.5 system-ui,sans-serif;margin:32px auto;padding:0 24px;max-width:1250px;color:#202b35;background:#f9fafb}}h1,h2{{line-height:1.2}}table{{border-collapse:collapse;width:100%;font-size:13px;background:white;margin:18px 0}}td,th{{padding:8px;border-bottom:1px solid #ddd;text-align:left}}th{{background:#e5eef1}}.notice{{padding:18px;background:#fff0d5;border-left:5px solid #bb7919}}img{{width:100%;height:auto}}code{{overflow-wrap:anywhere}}</style>
<h1>Eliminating unknown forces analytically</h1>
<p>Same irregular monocular observations, force prior, geometry, mechanics and observation-only warm start for alternating, joint and marginalized trajectory solvers.</p>
<div class="notice"><strong>Exact force elimination; approximate trajectory uncertainty.</strong>
{n_stationary}/{len(rows)} fits pass the local stationarity screen; {n_audit}/{len(rows)} pass the independent importance-weight concentration screen.
Neither screen establishes calibrated uncertainty, unique mechanics/force recovery or global convergence. Shaded force bands are provisional local-Gaussian/conditional estimates.</div>
<h2>What is compared</h2><p>The new model adds fixed full-rank independent free-node velocity diffusion of
{summary['plan']['trajectory_model']['velocity_diffusion']} m/√s. This makes a proper trajectory density. All three new solvers use exactly this model.
It differs from the historical model's absent or balanced low-rank neural internal noise. Historical rows use identical input images but cannot isolate a marginalization effect.</p>
<p>Alternating solves the Gaussian force conditional then updates mechanics/trajectory with force fixed. Joint optimizes force and trajectory together.
Marginalized integrates all 15 force coefficients and retains their conditional covariance. All optimize the same state/velocity coordinate density; fixed covariance makes joint-profile and marginalized trajectory objectives differ by a constant.</p>
<h2>Matched solver measurements</h2>{measured}
<p>Withheld MAP error is separate from the posterior-mean metric in the historical table. Timing excludes the shared existing-simulator initialization and final covariance/evaluation; see JSON for complete times. Same iteration budgets are not equal computational budgets. Importance ESS is a proposal-weight concentration diagnostic, not MCMC ESS.</p>
<img src="data:image/png;base64,{embedded}" alt="Force x and z histories, truth and provisional intervals for each condition. Grey denotes the longest missing interval.">
<h2>Historical simulator reference</h2>{historical}
<p>The historical MCMC fits fail their convergence checks. Their point estimates are context, not reliable posterior ground truth or a matched inference comparison.</p>
<h2>Reproducibility</h2><p>Source SHA-256: <code>{summary['source_fingerprint_sha256']}</code>.</p>
<p>Manifest, summary, scores and portable force posterior files accompany this report. Raw local checkpoints retain joint trajectory/material draws paired with conditional force draws; do not combine independent material and force marginals for scene generation.</p></html>"""
    (output/"report.html").write_text(document, encoding="utf-8")
