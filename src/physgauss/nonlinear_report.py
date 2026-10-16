"""Self-contained sampler diagnostics and explicitly provisional predictions."""
import base64
import csv
import html
import json
import os
from pathlib import Path
import torch


def make_nonlinear_report(output, summary):
    output = Path(output)
    os.environ.setdefault("MPLCONFIGDIR", str(output/".matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(len(summary["cases"]), 3, figsize=(14, 3.3*len(summary["cases"])),
                             squeeze=False, layout="constrained")
    rows, table = [], []
    for row, case in enumerate(summary["cases"]):
        chain = torch.load(output/"cases"/case["label"]/"chains.pt", weights_only=True)
        for c in range(chain["coordinates"].shape[0]):
            axes[row, 0].plot(chain["coordinates"][c, :, 0], linewidth=.5, alpha=.8, label=f"Chain {c+1}")
            axes[row, 1].plot(chain["scales"][c], linewidth=.5, alpha=.8)
            axes[row, 2].plot(chain["log_likelihood"][c], linewidth=.5, alpha=.8)
        for ax, title in zip(axes[row], ["Standardized log stiffness", "Process scale", "Observed log likelihood"]):
            ax.set(title=case["label"]+"\n"+title, xlabel="Retained transition")
            ax.grid(alpha=.15)
        for name, record in case["diagnostics"]["quantities"].items():
            rows.append([case["label"], name, record["rank_folded_split_rhat"], record["bulk_ess"],
                         record["tail_ess"], record["frozen_chains"], record["diagnostic_pass"]])
        qs = list(case["diagnostics"]["quantities"].values())
        maximum = max((q["rank_folded_split_rhat"] for q in qs if q["rank_folded_split_rhat"] is not None), default=None)
        minimum = min((q["bulk_ess"] for q in qs if q["bulk_ess"] is not None), default=None)
        for name, evaluation in case["evaluations"].items():
            s = evaluation["scores"]["withheld"]
            table.append([case["label"], name, str(case["diagnostics"]["all_monitored_quantities_pass"]),
                "unavailable" if maximum is None else f"{maximum:.3f}",
                "unavailable" if minimum is None else f"{minimum:.0f}",
                f"{1000*s['position_rmse_m']:.3f}", f"{1000*s['position_crps_m']:.3f}",
                f"{s['component_90_coverage']:.1%}", f"{1000*s['component_90_mean_width_m']:.3f}"])
    axes[0, 0].legend(fontsize=7)
    fig.savefig(output/"chains.png", dpi=140)
    plt.close(fig)
    with (output/"diagnostics.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["case", "quantity", "rank_folded_split_rhat", "bulk_ess", "tail_ess", "frozen_chains", "pass"])
        writer.writerows(rows)
    headers = ["Case", "Inference", "MCMC checks pass", "Max Rhat", "Min bulk ESS", "RMSE mm", "CRPS mm", "90% coverage", "Width mm"]
    rendered_table = "<tr>"+"".join("<th>"+h+"</th>" for h in headers)+"</tr>"
    rendered_table += "".join("<tr>"+"".join("<td>"+html.escape(s)+"</td>" for s in r)+"</tr>" for r in table)
    image = base64.b64encode((output/"chains.png").read_bytes()).decode("ascii")
    document = """<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Nonlinear posterior sampling</title>
<style>body{font:15px/1.6 system-ui;background:#f6f8fc;color:#172033;margin:0}main{max-width:1300px;margin:36px auto;padding:0 20px}.notice{padding:16px;background:#fff7ed;border-left:4px solid #b91c1c}img{width:100%;background:white;margin:20px 0}table{width:100%;border-collapse:collapse;background:white;font-size:12px}td,th{padding:8px;text-align:left;border-bottom:1px solid #ddd}.scroll{overflow:auto}pre{white-space:pre-wrap;font-size:11px}</style><main>
<h1>Nonlinear posterior sampling from sparse moving-camera images</h1>
<p>A mixture of preconditioned Crank-Nicolson (optionally gradient-guided pCNL), independent, and component-transport Metropolis transitions uses the full nonlinear observed-image likelihood. A local Gaussian mixture preconditions the proposals. Warmup is discarded and adaptation stops before retained draws. Ground truth is used only after sampling for evaluation.</p>
<div class="notice"><strong>Read convergence before prediction scores.</strong> Every latent coordinate, observed log likelihood, and adaptive process scale is monitored using rank-normalized/folded split Rhat and bulk/tail ESS. Rhat must be below 1.01, both ESS at least 400, and no chain frozen. Failed checks mean the displayed intervals remain provisional. Passing checks cannot exclude unseen modes or establish physical calibration.</div>
<p>Prediction scores use an equal, evenly spaced rendering subset per chain to control memory. All retained states enter the diagnostics. Correlated states are not independent experiments. The cases use one known geometry and camera path.</p>
<img alt="Four-chain stiffness, scale and log likelihood traces" src="data:image/png;base64,{{IMAGE}}"><div class="scroll"><table>{{TABLE}}</table></div>
<p><a href="https://arxiv.org/abs/1202.0709">pCN reference</a>; <a href="https://arxiv.org/abs/1903.08008">rank, folding and ESS reference</a>. No continuous-scale inference, inferred camera, or geometry completion is claimed.</p>
<details><summary>Full measurements and provenance</summary><pre>{{JSON}}</pre></details></main></html>"""
    document = document.replace("{{IMAGE}}", image).replace("{{TABLE}}", rendered_table)
    document = document.replace("{{JSON}}", html.escape(json.dumps(summary, indent=2)))
    (output/"report.html").write_text(document, encoding="utf-8")
