"""Create a nomenclature-corrected copy of the archived vector prediction figure.

Only the figure title is changed. Simulation pixels, traces, and recorded numerical
annotations remain those of the archived source file.
Run: python papers/cvpr2027-joint-identification/figures/normalize_predicted_shapes.py --repo .
"""
import argparse
import re
from pathlib import Path
import fitz

HERE = Path(__file__).resolve().parent
REPLACEMENT = "Predicted cantilever response to a held-out 20 N push at 0.75 L (top view)"


def main(repo):
    original = repo / "output/paper-figures/fig4-predicted-shapes.pdf"
    target = HERE / "fig4-predicted-shapes.pdf"
    doc = fitz.open(original)
    replacements = 0
    for page in doc:
        spans = [
            span
            for block in page.get_text("dict")["blocks"]
            for line in block.get("lines", [])
            for span in line.get("spans", [])
            if span["text"].startswith("Predicted stem shape under the held-out 20 N push")
        ]
        for span in spans:
            rect = fitz.Rect(span["bbox"])
            rect.x0 -= 1
            rect.x1 = min(page.rect.width - 1, rect.x1 + 20)
            rect.y0 -= 1
            rect.y1 += 1
            page.add_redact_annot(rect, fill=(1, 1, 1))
            page.apply_redactions(images=0, graphics=0, text=0)
            page.insert_text(span["origin"], REPLACEMENT,
                             fontsize=span["size"], fontname="hebo", color=(0, 0, 0))
            replacements += 1
    if replacements != 1:
        raise RuntimeError(f"Expected one archived figure title; found {replacements}")
    if any(page.search_for("Predicted stem shape") for page in doc):
        raise RuntimeError("Remaining legacy figure terminology")
    doc.save(target, garbage=4, deflate=True, no_new_id=True)
    doc.close()
    print(f"Saved revised vector figure: {target}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", type=Path, default=HERE.parents[2])
    main(ap.parse_args().repo)
