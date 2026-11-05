BEYOND VISUAL AGREEMENT
Compositional Mechanics for Position-Driven Deformable Twins
Latest cohesive CVPR-format manuscript, 5 November 2026.

OPEN THESE FILES
main.pdf: eight content pages plus one references/LLM-disclosure page.
supplement.pdf: complete controlled and external protocols, per-case tables,
reference sensitivity, failures and later flow diagnostics.
release/latex-source.zip: self-contained anonymous LaTeX source with PDF figures.

WHAT IS CONSOLIDATED
Six simulation interaction families (24 worlds / 48 queries), including
supports/release, curved contact, drops/obstacles, rigid impact, two soft bodies
and heterogeneous material; the strongest prospective40-case SoftVT result;
all8 real PhysTwin continuations; and the2 geometry-qualified real transfer
pairs. Static observability and additional-observation experiments explain
why visual agreement and physical response need separate evaluation.
Older dynamic, geometry-transfer and inverse experiments remain in the supplement.
The latest conditional-flow expansion is a diagnostic, not a replacement for
the stronger established models. Every adapter and metric is labeled.

DELIVERABLE CHANGES
One coherent method and evaluation narrative replaces two disconnected drafts.
Seven new scientific figures include matched error-map zooms, three physical
coordinate roles, exact scale ambiguity, per-world scenario results and real
capture comparisons. These use saved evidence or explicitly labeled schematic
geometry, not image-generated experimental results. Old papers and predictions
remain unchanged. No new prediction run was necessary: the independent audits
identified sufficient existing evidence for the claims actually made.

BUILD (from this directory)
  tectonic main.tex
  tectonic supplement.tex
Or use pdflatex/bibtex/pdflatex/pdflatex for each document.
The official CVPR2027 style and bibliography style are unchanged. T1 font
encoding ensures the required Times-compatible fonts instead of a silent XeTeX
fallback. Raw data are not required to rebuild from supplied PDF figures.

AUDIT (from repository root)
  python3 papers/cvpr2027-physical-response/tools/audit_evidence.py
Checks156 frozen-source, arithmetic, coverage and numerical-claim conditions.
evidence/current-audit.json binds the current source. This is not an independent
physical-ground-truth replication. Separate independent scientific, literature,
source-level adapter and page-layout reviews are in reports/cvpr-cohesive-v01/.

FIGURES (requires the local saved experiment assets)
  MPLCONFIGDIR=/tmp/cvpr-mpl runs/phystwin-env/bin/python \
    papers/cvpr2027-physical-response/tools/build_cohesive_figures.py
  MPLCONFIGDIR=/tmp/cvpr-mpl runs/phystwin-env/bin/python \
    papers/cvpr2027-physical-response/tools/build_family_results.py
Do not rebuild while TeX is reading the figure files. Provenance manifests bind
source arrays, selection rules, crops and final figure hashes.

CLAIMS AND STATUS
This is an anonymous manuscript prepared for author review, not a submission or
a guarantee of acceptance. General SOTA, calibrated external physical inference,
superior active probing and image-only reconstruction are not established.
Strong analytic/native/simple controls and mixed real outcomes are retained.
Registration, authorship, conflicts and final author approval remain outstanding;
no author identity or submission ID has been fabricated. The mandatory LLM usage
statement is included. Official author guidance was checked5 November2026.

Raw recordings, checkpoints and large trajectories remain local. PokeFlex is
unavailable under approved access; Deform360 is deferred at the user's request.
