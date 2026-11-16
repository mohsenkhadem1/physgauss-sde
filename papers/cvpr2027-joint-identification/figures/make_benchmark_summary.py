"""Render the paper's benchmark summary from sealed, already-scored physical predictions.

Only the scored result JSON is read. No inference, fitting, or held-out reference
is performed here. Output: papers/cvpr2027-joint-identification/figures/benchmark-summary.pdf
"""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
METHODS = ("ours (RGB)", "M-PhyGs", "PhysTwin", "GIC", "PAC-NeRF")
COLORS = ("#237b9e", "#c08d38", "#aa719f", "#767f87", "#b95d52")
PANELS = (
    ("S1", "rms_mm", "Cantilever: oblique push", "Displacement error (mm)", False),
    ("S2", "rms_mm", "Block: off-centre press", "Displacement error (mm)", False),
    ("S3", "rms_mm", "Cantilever: collision", "Displacement error (mm)", False),
    ("S3", "impulse_err_Ns", "Free-striker rebound", "Impulse error (N s)", True),
)

def make(repo):
    data = json.loads((repo / "reports/fig1-v01/fig1-scores.json").read_text())["rows"]
    plt.rcParams.update({"font.family": "DejaVu Sans", "pdf.fonttype": 42,
                         "font.size": 8.5, "axes.spines.top": False,
                         "axes.spines.right": False, "figure.facecolor": "white"})
    fig, axes = plt.subplots(1, 4, figsize=(14.7, 3.65))
    fig.subplots_adjust(left=.125, right=.987, top=.81, bottom=.22, wspace=.60)
    for ax, (scenario, metric, title, xlabel, logarithmic) in zip(axes, PANELS):
        values = []
        for method in METHODS:
            rows = [r[metric] for r in data if r["scenario"] == scenario
                    and r["method"] == method and r["evidence"] == "after poke"]
            if len(rows) != 3:
                raise ValueError("Expected exactly three sealed scores: " + scenario + " " + method)
            values.append(sum(rows)/3)
        for y, (v, color) in enumerate(zip(values, COLORS)):
            ax.barh(y, v, color=color, height=.62)
            label = f"{v:.3f}" if logarithmic else f"{v:.2f}"
            ax.annotate(label, (v, y), xytext=(4, 0), textcoords="offset points",
                        va="center", fontsize=7.4)
        ax.set_title(title, loc="left", fontweight="bold", fontsize=9.0)
        ax.set_yticks(range(5))
        ax.set_yticklabels(METHODS if ax is axes[0] else [""] * 5)
        ax.tick_params(axis="y", length=0)
        ax.invert_yaxis()
        ax.set_xlabel(xlabel, fontsize=7.5, labelpad=6)
        if logarithmic:
            ax.set_xscale("log")
            ax.set_xlim(.001, 4)
        else:
            ax.set_xlim(0, max(values)*1.35)
        ax.grid(axis="x", alpha=.15)
        ax.set_axisbelow(True)
    fig.suptitle("Response to a new action after one calibration interaction",
                 x=.125, y=.98, ha="left", fontsize=11, fontweight="bold")
    dest = HERE / "benchmark-summary.pdf"
    fig.savefig(dest, bbox_inches="tight", metadata={"Creator":"Scored-physical-benchmark",
                                                     "CreationDate":None,"ModDate":None})
    plt.close(fig)
    print(dest)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=HERE.parents[2])
    make(parser.parse_args().repo)
