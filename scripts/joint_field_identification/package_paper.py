"""Verify the numerical inputs and package the anonymous paper for a standalone build."""
import hashlib
import json
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PAPER = ROOT / "papers/cvpr2027-joint-identification"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    assert "Working draft" not in (PAPER / "sections/results-macros.tex").read_text()
    for name in ("results-provenance.json", "interaction-provenance.json"):
        for source, expected in json.loads((PAPER / name).read_text()).items():
            assert digest(ROOT / source) == expected, source

    sources = {name: PAPER / name for name in (
        "main.tex", "supplement.tex", "cvpr.sty", "ieeenat_fullname.bst", "references.bib",
        "README.txt", "build.sh", "results-provenance.json", "interaction-provenance.json",
        "figures/make_task_teaser.py",
    )}
    for directory, glob in (("sections", "*.tex"), ("figures", "*.pdf")):
        for path in (PAPER / directory).glob(glob):
            if not path.name.startswith("gap-first-"):
                sources[path.relative_to(PAPER).as_posix()] = path
    # These panels are already tracked with their experiments. Bundle them in figures/
    # so both PDFs build from the source ZIP without a repository checkout.
    for source in ("output/paper-figures/fig4-predicted-shapes.pdf",
                   "output/stem-trial-v01/stem-render-physics.pdf"):
        sources["figures/" + Path(source).name] = ROOT / source

    release = PAPER / "release"
    release.mkdir(exist_ok=True)
    manifest = release / "source-manifest.json"
    manifest.write_text(json.dumps({n: digest(p) for n, p in sorted(sources.items())}, indent=2) + "\n")
    sources[manifest.name] = manifest
    with zipfile.ZipFile(release / "latex-source.zip", "w") as archive:
        for name, source in sorted(sources.items()):
            info = zipfile.ZipInfo(name, date_time=(2026, 11, 9, 0, 0, 0))
            info.external_attr = (0o100755 if name == "build.sh" else 0o100644) << 16
            archive.writestr(info, source.read_bytes(), compress_type=zipfile.ZIP_DEFLATED,
                             compresslevel=9)
    print(f"Packaged {len(sources)} files and verified both study manifests.")


if __name__ == "__main__":
    main()
