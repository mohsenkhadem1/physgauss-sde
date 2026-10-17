"""Self-contained external-force posterior and impulse report."""
import base64
import csv
import html
import json
import os
from pathlib import Path
import numpy as np

COLORS = {"force_only": "#2563eb", "sde_only": "#dc2626", "force_and_sde": "#059669"}


def make_force_report(output, summary):
    output = Path(output)
    os.environ.setdefault("MPLCONFIGDIR", str(output/".matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    pairs = list(dict.fromkeys((c["condition"], c["seed"]) for c in summary["cases"]))
    fig, axes = plt.subplots(len(pairs), 3, figsize=(13, 3*len(pairs)), squeeze=False, layout="constrained")
    rows, diagnostic_rows = [], []
    for row, (condition, seed) in enumerate(pairs):
        cases = [c for c in summary["cases"] if c["condition"] == condition and c["seed"] == seed]
        for case_index, case in enumerate(cases):
            field = json.loads((output/"cases"/case["label"]/"fields.json").read_text())
            t = np.array(field["times_s"])
            mean = np.array(field["external_total_mean_N"])
            lo, hi = np.array(field["external_total_q05_N"]), np.array(field["external_total_q95_N"])
            target = np.array(field["truth_external_total_N_evaluation_only"])
            for direction, axis in enumerate(axes[row]):
                if case_index == 0:
                    axis.plot(t, target[:, direction], color="#111827", linestyle="--", linewidth=1.5, label="True external load")
                    config = case["config"]
                    left, right = max(zip(config["observed_frames"][:-1], config["observed_frames"][1:]), key=lambda p: p[1]-p[0])
                    duration = config["duration"]
                    axis.axvspan(left*duration/(config["frames"]-1), right*duration/(config["frames"]-1), color="#e2e8f0", alpha=.5)
                color = COLORS[case["method"]]
                axis.plot(t, mean[:, direction], color=color, label=case["method"].replace("_", " "), linewidth=1)
                axis.fill_between(t, lo[:, direction], hi[:, direction], color=color, alpha=.13)
                axis.set(title=f"{condition}, seed {seed} | {'xyz'[direction]} load", xlabel="Time (s)", ylabel="Total external force (N)")
                axis.grid(alpha=.15)
        axes[row, 0].legend(fontsize=7)
    fig.suptitle("Conditional external-force posteriors | Bands: 5-95%; grey: largest observation gap")
    fig.savefig(output/"forces.png", dpi=140); plt.close(fig)
    for case in summary["cases"]:
        q = list(case["diagnostics"]["quantities"].values())
        max_rhat = max((v["rank_folded_split_rhat"] for v in q if v["rank_folded_split_rhat"] is not None), default=None)
        min_bulk = min((v["bulk_ess"] for v in q if v["bulk_ess"] is not None), default=None)
        f = case["force_scores"]["external_total"]
        impulse = case["force_scores"]["external_impulse"]
        position = case["trajectory_scores"]["withheld"]["position_rmse_m"]
        rows.append([case["condition"], str(case["seed"]), case["method"],
            str(case["diagnostics"]["all_monitored_quantities_pass"]),
            "undefined" if max_rhat is None else f"{max_rhat:.3f}",
            "undefined" if min_bulk is None else f"{min_bulk:.0f}",
            f"{f['rmse_N']:.5f}", f"{f['component_90_coverage']:.1%}",
            f"{f['component_90_mean_width_N']:.5f}", f"{impulse['rmse_Ns']:.5f}", f"{1000*position:.3f}"])
        for name, value in case["diagnostics"]["quantities"].items():
            diagnostic_rows.append([case["label"], name, value["rank_folded_split_rhat"], value["bulk_ess"], value["tail_ess"], value["diagnostic_pass"]])
    headers = ["Condition", "Seed", "Model", "All chain checks pass", "Max Rhat", "Min bulk ESS", "Force RMSE N", "90% coverage", "Width N", "Impulse RMSE Ns", "Position RMSE mm"]
    for name, head, body in [("scores.csv", headers, rows), ("diagnostics.csv", ["case", "quantity", "Rhat", "bulk_ess", "tail_ess", "pass"], diagnostic_rows)]:
        with (output/name).open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream); writer.writerow(head); writer.writerows(body)
    table = "<tr>"+"".join("<th>"+h+"</th>" for h in headers)+"</tr>"
    table += "".join("<tr>"+"".join("<td>"+html.escape(v)+"</td>" for v in row)+"</tr>" for row in rows)
    image = base64.b64encode((output/"forces.png").read_bytes()).decode("ascii")
    document = """<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Force-field posterior experiment</title>
<style>body{font:15px/1.6 system-ui;background:#f6f8fc;color:#172033;margin:0}main{max-width:1400px;margin:36px auto;padding:0 20px}.notice{background:#fff7ed;border-left:4px solid #b91c1c;padding:16px}img{width:100%;background:white;margin:20px 0}table{font-size:12px;background:white;width:100%;border-collapse:collapse}td,th{text-align:left;padding:8px;border-bottom:1px solid #ddd}.scroll{overflow:auto}pre{white-space:pre-wrap;font-size:11px}</style><main>
<h1>Force distributions from sparse moving-camera observations</h1>
<p>A 15-coordinate projected Matérn prior describes a smooth external load on a known material-attached spatial profile. The three models infer that load, fixed-amplitude internal SDE impulses, or both. All fit the same observed images per condition. The SDE-only model excludes external force: its zero external-force band is conditional on that exclusion, not evidence that the load is absent.</p>
<div class="notice"><strong>Inspect convergence before using bands.</strong> A nonlinear fitted proposal and Metropolis correction target the full image likelihood and standardized priors. Rank/folded Rhat must be below 1.01, both bulk and tail ESS at least 400, and no chain frozen. Failed checks leave scores and intervals provisional. This is one known geometry, with selected synthetic force conditions; no real-world calibration or inferred model probabilities are claimed.</div>
<p>Forces are measured in N and impulses in Ns. White-noise forcing is recorded as finite-step impulses; combined forcing is averaged over explicit solver windows. The spatial profile and temporal bandwidth are restricted by design. Camera, rest geometry, density and supports remain known. Full covariance and paired force/impulse samples are saved in raw run files.</p>
<img alt="External force posterior means, intervals and synthetic truth along three axes" src="data:image/png;base64,{{IMAGE}}"><div class="scroll"><table>{{TABLE}}</table></div>
<p>Every retained state enters latent diagnostics, external-force moments, intervals and scores; the force-impulse functionals have separate diagnostics in JSON. An equally spaced subset per chain controls rendering and combined force/impulse evaluation cost and is not assumed independent. Compact coefficient moments and joint function draws allow querying the external posterior at other times inside the recording. Truth is used only after fitting for evaluation, including the force-basis representation error.</p>
<details><summary>Full measurements and provenance</summary><pre>{{JSON}}</pre></details></main></html>"""
    document = document.replace("{{IMAGE}}", image).replace("{{TABLE}}", table).replace("{{JSON}}", html.escape(json.dumps(summary, indent=2)))
    (output/"report.html").write_text(document, encoding="utf-8")
