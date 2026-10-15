"""Self-contained nonlinear likelihood-audit report."""
import base64
import csv
import html
import json
import os
from pathlib import Path
import numpy as np
from .study_report import COLORS, LABELS


def make_audit_report(output, summary):
    output = Path(output)
    os.environ.setdefault("MPLCONFIGDIR", str(output / ".matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    names = list(dict.fromkeys(c["condition"] for c in summary["cases"]))
    fig, axes = plt.subplots(1, len(names), figsize=(5*len(names), 4.3), squeeze=False, layout="constrained")
    for col, condition in enumerate(names):
        matching = [c for c in summary["cases"] if c["condition"] == condition]
        labels = [c["schedule"] + "\n" + str(c["seed"]) for c in matching]
        ax = axes[0, col]
        for i, method in enumerate(LABELS):
            values = [c["methods"][method]["relative_ess"] for c in matching]
            ax.scatter(np.arange(len(matching)) + (i-1)*.13, values, color=COLORS[method], label=LABELS[method], s=30)
        ax.axhline(.1, color="#b91c1c", linestyle="--", linewidth=1)
        ax.set(title=condition.replace("_", " "), ylim=(-.02, 1.03), ylabel="Relative importance ESS")
        ax.set_xticks(range(len(matching)), labels, fontsize=7)
        ax.grid(alpha=.15)
    axes[0, 0].legend(fontsize=7)
    fig.suptitle("Local posterior overlap with the nonlinear likelihood | Dashed: diagnostic threshold 0.1")
    fig.savefig(output / "likelihood_audit.png", dpi=150)
    plt.close(fig)
    rows = []
    for case in summary["cases"]:
        for method, value in case["methods"].items():
            rows.append([case["condition"], case["schedule"], str(case["seed"]), LABELS[method],
                         f"{value['relative_ess']:.3f}", f"{value['importance_ess']:.1f}",
                         f"{value['maximum_normalized_weight']:.3f}", str(value["invalid_physical_proposals"])])
    headers = ["Condition", "Schedule", "Seed", "Proposal", "Relative ESS", "ESS", "Max weight", "Invalid"]
    with (output / "diagnostics.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f); writer.writerow(headers); writer.writerows(rows)
    table = "<tr>" + "".join("<th>"+h+"</th>" for h in headers) + "</tr>"
    table += "".join("<tr>"+"".join("<td>"+html.escape(v)+"</td>" for v in row)+"</tr>" for row in rows)
    image = base64.b64encode((output / "likelihood_audit.png").read_bytes()).decode("ascii")
    document = """<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Nonlinear likelihood audit</title>
<style>body{font:15px/1.6 system-ui,sans-serif;background:#f6f8fc;color:#172033;margin:0}main{max-width:1200px;margin:40px auto;padding:0 22px}.notice{background:#fff7ed;border-left:4px solid #b91c1c;padding:16px}img{width:100%;margin:20px 0;background:white;border:1px solid #e2e8f0}table{border-collapse:collapse;background:white;width:100%;font-size:13px}td,th{padding:8px;text-align:left;border-bottom:1px solid #e2e8f0}.scroll{overflow:auto}pre{font-size:12px;white-space:pre-wrap;background:#e8edf5;padding:15px}</style><main>
<h1>Does the local posterior match the nonlinear likelihood?</h1><p>Independent proposals from each linearized posterior are evaluated using the full physical simulation and observed-image likelihood. Priors are unchanged, so the importance ratio is the nonlinear likelihood divided by the local linearized likelihood. No withheld ground truth is used.</p>
<div class="notice"><strong>Diagnostic, not a posterior correction:</strong> low effective sample size indicates concentrated weights and fragile local-posterior overlap. A high finite-sample ESS cannot exclude unseen modes or establish calibration. We do not report reweighted uncertainty intervals as reliable results.</div>
<p>Each case/method uses {{COUNT}} independent proposals. The threshold 0.1 was declared before this audit. Exact historical source fingerprints and unchanged model modules are verified; old checkpoints without saved observations require the matching numerical environment for regeneration.</p>
<img alt="Relative importance effective sample size across conditions and proposals" src="data:image/png;base64,{{IMAGE}}"><div class="scroll"><table>{{TABLE}}</table></div>
<details><summary>Full record and source compatibility verification</summary><pre>{{RECORD}}</pre></details></main></html>"""
    document = document.replace("{{COUNT}}", str(summary["samples_per_method_case"]))
    document = document.replace("{{IMAGE}}", image).replace("{{TABLE}}", table)
    document = document.replace("{{RECORD}}", html.escape(json.dumps(summary, indent=2)))
    (output / "report.html").write_text(document, encoding="utf-8")
