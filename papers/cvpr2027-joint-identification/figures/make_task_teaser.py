"""Build the paper teaser directly from the two archived stem experiments.

Usage: python figures/make_task_teaser.py [--repo PATH_TO_REPOSITORY]
Dependencies: matplotlib, numpy, Pillow, PyMuPDF. Reference image from the
first (original reference) panel of the archived held-camera figure.
"""
import argparse
import io
import json
from pathlib import Path

import fitz
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
INK, MUTED, GRID = "#162a36", "#455966", "#d6dfe0"
TEAL, ORANGE = "#006b70", "#a95123"


def passive_rgb(path):
    with fitz.open(path) as doc:
        images = doc[0].get_image_info(xrefs=True)
        first_row = min(item["bbox"][1] for item in images)
        reference = min((image for image in images if image["bbox"][1] < first_row + 1),
                        key=lambda item: item["bbox"][0])
        data = doc.extract_image(reference["xref"])["image"]
        return np.asarray(Image.open(io.BytesIO(data)).convert("RGB"))


def section(fig, x, letter, title):
    fig.text(x, .93, letter, fontsize=8, fontweight="bold", ha="left", va="center",
             color="white", bbox={"boxstyle": "round,pad=.21", "fc": INK, "ec": "none"})
    fig.text(x+.031, .929, title, fontsize=9.2, fontweight="bold", va="center", color=INK)


def draw(repo):
    with (repo / "reports/stem-trial-v01/scores.json").open() as fp:
        rows = json.load(fp)["rows"]
    with (repo / "reports/gap-first-v01/stem/summary.json").open() as fp:
        cohorts = json.load(fp)["conditions"]
    worlds = ("W500", "W1000", "W2000")
    held = [next(r for r in rows if r["world"] == w and r["method"] == "GIC"
                 and r["evidence"] == "D0 only") for w in worlds]
    truth = np.array([r["peak_z_075L_true_mm"] for r in held])
    baseline = np.array([r["peak_z_075L_pred_mm"] for r in held])
    assert np.allclose(baseline, baseline[0])

    mpl.rcParams.update({"font.family": "DejaVu Sans", "pdf.fonttype": 42,
                         "axes.linewidth": .6, "xtick.labelsize": 7.0,
                         "ytick.labelsize": 7.2, "text.color": INK})
    fig = plt.figure(figsize=(7.1, 2.12), facecolor="white")
    section(fig, .012, "a", "Same passive sag")
    section(fig, .239, "b", "Response to a 20 N push")
    section(fig, .625, "c", "Which evidence to collect?")

    rgb = passive_rgb(repo / "output/stem-trial-v01/stem-render-physics.pdf")
    ax = fig.add_axes([.012, .25, .199, .50])
    ax.imshow(rgb, interpolation="none")
    ax.axis("off")
    fig.text(.012, .79, "Held-out camera, frame 24", fontsize=7.2, color=MUTED)
    fig.text(.012, .194, "One observed motion", fontsize=7.8, fontweight="bold")
    fig.text(.012, .115, "for three material pairs.", fontsize=6.9, color=MUTED)

    ax = fig.add_axes([.292, .265, .285, .43])
    ax.set_xlim(0, 37)
    ax.set_ylim(2.55, -.55)
    ax.set_yticks(range(3), ["500", "1,000", "2,000"])
    ax.set_xticks([0, 10, 20, 30])
    ax.set_axisbelow(True)
    ax.grid(axis="x", color=GRID, linewidth=.5)
    ax.tick_params(colors=MUTED, length=0, pad=2)
    ax.spines[["top", "left", "right"]].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.set_xlabel("Peak lateral displacement (mm)", fontsize=7.7, color=INK, labelpad=2.5)
    ax.axvline(baseline[0], color=ORANGE, lw=1.1, linestyle=(0, (2.5, 2.0)))
    for y, x in enumerate(truth):
        ax.plot([0, x], [y, y], color=TEAL, lw=1.3, solid_capstyle="round")
        ax.scatter(x, y, s=22, color=TEAL, zorder=3)
        ax.annotate(f"{x:.1f}", (x, y), xytext=(4, 0), textcoords="offset points",
                    fontsize=7.4, va="center", color=TEAL, fontweight="bold")
    fig.text(.239, .785, "Density (kg/m³)", fontsize=7.2, color=MUTED)
    fig.text(.239, .028, f"GIC predicts {baseline[0]:.1f} mm for all three stems (dashed).",
             fontsize=6.6, color=ORANGE)

    ax = fig.add_axes([.615, .18, .382, .565])
    ax.set_xlim(0, 3.55)
    ax.set_ylim(3.3, -.55)
    ax.axis("off")
    col_x = [1.65, 2.40, 3.15]
    for x, label in zip(col_x, ["Proceed", "Look", "Probe"]):
        ax.text(x, -.25, label, ha="center", va="center", fontsize=7.5, color=MUTED)
    conditions = [("repeat_tip", "Same load"),
                  ("visible_base", "New load,\nvisible base"),
                  ("concealed_base", "New load,\nhidden base")]
    for i, (key, name) in enumerate(conditions):
        result = cohorts[key]["risk_policy"]
        actions = result["actions"]
        counts = [actions.get("proceed", 0), actions.get("look", 0),
                  sum(n for a, n in actions.items() if a.startswith("push_"))]
        assert sum(counts) == result["n"] == 96
        y = i + .48
        ax.text(.03, y, name, va="center", fontsize=7.35, linespacing=1.03)
        for x, count in zip(col_x, counts):
            foreground, background = (("white", TEAL) if count == max(counts)
                                      else (MUTED, "#eef2f3"))
            ax.add_patch(plt.Rectangle((x-.34, y-.41), .68, .82,
                                       facecolor=background, edgecolor="none"))
            ax.text(x, y, str(count), ha="center", va="center", color=foreground,
                    fontsize=8.1, fontweight="bold" if count == max(counts) else "normal")
    fig.text(.625, .072, "Action counts over 96 beams per condition", fontsize=6.9, color=MUTED)
    for x in (.224, .602):
        fig.add_artist(mpl.lines.Line2D([x, x], [.12, .81], transform=fig.transFigure,
                                        color=GRID, linewidth=.65))
    fig.savefig(HERE / "task-teaser.pdf", metadata={"Creator": "Figure generator",
                                                     "CreationDate": None, "ModDate": None})
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=HERE.parents[2])
    draw(parser.parse_args().repo)
