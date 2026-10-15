"""Static scientific plots and self-contained report for the scale study."""
import base64
import csv
import html
import io
import json
import os
from pathlib import Path
import numpy as np


LABELS = {"physical": "No process noise", "fixed_sde": "Fixed neural SDE", "adaptive_sde": "Adaptive noise mixture"}
COLORS = {"physical": "#2563eb", "fixed_sde": "#d97706", "adaptive_sde": "#059669"}


def make_study_report(output, summary):
    output = Path(output)
    os.environ.setdefault("MPLCONFIGDIR", str(output / ".matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    conditions = summary["plan"]["conditions"]
    schedules = list(summary["plan"]["schedules"])
    fig, axes = plt.subplots(3, len(conditions), figsize=(5 * len(conditions), 10),
                             squeeze=False, layout="constrained")
    for column, condition in enumerate(conditions):
        for method in LABELS:
            records = [g for g in summary["groups"] if g["condition"] == condition and g["method"] == method]
            ordered = [next(g for g in records if g["schedule"] == schedule) for schedule in schedules]
            for row, metric, factor in [(0, "position_rmse_m", 1000), (1, "component_90_mean_width_m", 1000)]:
                means = np.array([g["metrics"][metric]["mean"] * factor for g in ordered])
                deviations = [g["metrics"][metric]["seed_std"] for g in ordered]
                errors = np.array([(d if d is not None else 0) * factor for d in deviations])
                axes[row, column].errorbar(np.arange(len(schedules)), means, yerr=errors, fmt="o-",
                                           color=COLORS[method], capsize=4, label=LABELS[method])
            levels = [.5, .8, .9, .95]
            matching = [c for c in summary["cases"] if c["condition"] == condition]
            coverage = [[c["methods"][method]["coverage_curve_withheld"][str(level)]["coverage"]
                         for c in matching] for level in levels]
            axes[2, column].plot(levels, np.mean(coverage, axis=1), "o-", color=COLORS[method],
                                 label=LABELS[method])
        axes[0, column].set_title(condition.replace("_", " "))
        for row in range(2):
            axes[row, column].set_xticks(np.arange(len(schedules)), schedules)
        axes[2, column].plot([.45, 1], [.45, 1], "--", color="#64748b")
        axes[2, column].set(xlim=(.45, 1), ylim=(0, 1.03), xlabel="Nominal component coverage")
        for ax in axes[:, column]:
            ax.grid(alpha=.18)
    axes[0, 0].set_ylabel("Withheld coordinate RMSE (mm)")
    axes[1, 0].set_ylabel("Mean 90% interval width (mm)")
    axes[2, 0].set_ylabel("Empirical coverage")
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("All withheld times | Bars: standard deviation across seeds, not confidence intervals", fontsize=13)
    fig.savefig(output / "study.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), layout="constrained")
    group_names = [(c, s) for c in conditions for s in schedules]
    for i, (condition, schedule) in enumerate(group_names):
        cases = [c for c in summary["cases"] if c["condition"] == condition and c["schedule"] == schedule]
        axes[0].scatter([i] * len(cases), [c["scale_posterior"]["probability_zero_process_noise"] for c in cases],
                         s=35, alpha=.8)
        axes[1].scatter([i] * len(cases), [c["scale_posterior"]["posterior_mean_scale"] for c in cases], s=35, alpha=.8)
    ticks = [c.replace("_", " ") + "\n" + s for c, s in group_names]
    for ax in axes:
        ax.set_xticks(range(len(ticks)), ticks, fontsize=7)
        ax.grid(alpha=.18)
    axes[0].set(title="Evidence for the zero-noise component", ylabel="Posterior probability", ylim=(-.03, 1.03))
    axes[1].set(title="Inferred effective process-noise scale", ylabel="Posterior mean scale")
    fig.savefig(output / "noise_scale.png", dpi=150)
    plt.close(fig)

    rows = []
    csv_rows = []
    for group in summary["groups"]:
        m = group["metrics"]
        values = [group["condition"], group["schedule"], LABELS[group["method"]],
                  str(group["independent_realizations"]),
                  f"{m['position_rmse_m']['mean']*1000:.3f}",
                  f"{m['position_crps_m']['mean']*1000:.3f}",
                  f"{m['component_90_coverage']['mean']:.1%}",
                  f"{m['component_90_mean_width_m']['mean']*1000:.3f}",
                  f"{m['normalized_ause']['mean']:.3f}",
                  f"{m['linearization_rmse_sigma']['mean']:.3f}"]
        rows.append("<tr>" + "".join("<td>" + html.escape(v) + "</td>" for v in values) + "</tr>")
        csv_rows.append(values)
    headers = ["Truth", "Schedule", "Method", "Seeds", "RMSE mm", "CRPS mm", "90% coverage", "Width mm", "AUSE", "Lin. error sigma"]
    with (output / "groups.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(csv_rows)
    figures = "".join('<img alt="' + label + '" src="data:image/png;base64,'
                      + base64.b64encode((output / filename).read_bytes()).decode("ascii") + '">'
                      for filename, label in [("study.png", "Accuracy, interval width and coverage"),
                                               ("noise_scale.png", "Posterior process-noise scale")])
    document = """<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>PhysGauss-SDE adaptive noise study</title><style>body{font:15px/1.6 system-ui,sans-serif;background:#f6f8fc;color:#172033;margin:0}main{max-width:1220px;margin:40px auto;padding:0 22px}h1{font-size:34px;line-height:1.2}.notice{background:#fff7ed;border-left:4px solid #d97706;padding:16px 20px}img{width:100%;margin-top:20px;border:1px solid #e2e8f0;border-radius:10px;background:white}.scroll{overflow:auto}table{width:100%;border-collapse:collapse;background:white;font-size:13px}td,th{padding:10px;text-align:left;border-bottom:1px solid #e2e8f0}pre{font-size:12px;white-space:pre-wrap;background:#e8edf5;padding:15px}small{color:#475569}</style><main>
<p><small>PHYSICALLY CONSTRAINED GAUSSIAN RECONSTRUCTION / MILESTONE 2</small></p><h1>Learning when stochastic motion is needed</h1>
<p>Matched deterministic, fixed-noise and adaptive-mixture inference across independent motion/noise realizations and observation schedules. All methods use the same image linearization, scene prior, material model and input images. Noise levels are compared by integrating latent Gaussian disturbances, using observed images only.</p>
<div class="notice"><strong>Scope:</strong> one known rest geometry and camera path. These are independent realizations, not independent scene geometries. Scale evidence is exact for a local linearized likelihood and approximate for the nonlinear renderer. The deterministic pulse condition contains an external force absent from inference: selected noise can reflect model mismatch. This is not a demonstrated physical-noise detector or across-scene calibration result.</div>
<p>All tables score every withheld timestamp. Schedules have identical image budgets. Error bars show seed standard deviations; they are not confidence intervals. The coverage plot averages per-realization diagnostics across the listed schedules; repeated schedules for one seed are correlated.</p>
{{FIGURES}}<h2>Per-condition measurements</h2><div class="scroll"><table><tr>{{HEADERS}}</tr>{{ROWS}}</table></div>
<p>AUSE assesses error ranking, not probability calibration. Coverage must be read with interval width and CRPS. The last column measures nonlinear render disagreement with the fitted local likelihood, in units of known image-noise sigma; a large value weakens confidence in the approximate evidence. Noise-grid and spike-prior choices are declared before evaluation; no held-out trajectory or image is used to select them. The complete record includes sensitivity to alternative predeclared zero-noise prior masses.</p>
<details><summary>Full reproducibility record and per-case results</summary><pre>{{RECORD}}</pre></details></main></html>"""
    document = document.replace("{{FIGURES}}", figures).replace("{{ROWS}}", "".join(rows))
    document = document.replace("{{HEADERS}}", "".join("<th>" + h + "</th>" for h in headers))
    document = document.replace("{{RECORD}}", html.escape(json.dumps(summary, indent=2)))
    (output / "report.html").write_text(document, encoding="utf-8")
