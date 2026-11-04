POSITION-DRIVEN DEFORMABLE TWINS
Separating Material Response from Interaction Conditions

Deliverables
- main.pdf: eight content pages plus references and the LLM usage statement.
- supplement.pdf: five pages of implementation details and complete case tables.
- main.tex, supplement.tex, references.bib: editable anonymous CVPR source.
- figures/: vector scientific figures; no image-generation evidence.
- evidence/: exact result-source hashes, figure selection, numbers and audit.

Scope
This paper consolidates the established position-driven mechanics work on
SoftVTBench and real PhysTwin captures. It excludes the recent video-query model.
SoftVT uses the analytic mechanics/slip-release adapter; real PhysTwin uses the
corrected-mask neural and analytic adapters. These are explicitly different
implementations of the same representation principle, not one shared checkpoint.
The strongest archived PhysTwin selector is retained with its historical protocol
in the supplement. No result is reassigned to another architecture. The controlled
synthetic mechanics comparison is supporting evidence, not an external benchmark.

Build
From this directory with Tectonic:
  tectonic main.tex
  tectonic supplement.tex
Alternatively use pdflatex/bibtex/pdflatex/pdflatex for main.tex, and pdflatex twice
for supplement.tex. The bundled official CVPR2027 style and bibliography style
are unchanged from the existing manuscript. Supplied PDF figures make the source
package buildable without raw datasets or training.

Regenerate figures in the full local development repository:
  OPENBLAS_NUM_THREADS=1 runs/phystwin-env/bin/python \
    papers/cvpr2027-position-driven-twins/polish_figures.py
The polish script first runs generate_figures.py. Both read saved predictions;
neither trains, simulates or changes benchmark outputs. Raw data are required for
regeneration, but are intentionally excluded from the paper source package.

Audit in the full repository:
  python papers/cvpr2027-position-driven-twins/audit.py
This verifies recorded source hashes, recomputes summary arithmetic, checks every
published per-case row, and checks PDF text/page counts. It is a same-agent
publication audit, not a fresh independent replication of the experiments.
All 14 final pages were rendered and visually inspected for layout and legibility.

Submission status
Prepared for author review, not submitted. No submission ID or author identity
has been invented. The required LLM usage statement is included. Registration,
authorship and final author approval remain outside this writing task.
Official author guidance checked 4 November 2026:
https://cvpr.thecvf.com/Conferences/2027/AuthorGuidelines

Primary evidence
reports/benchmark-improvement-v14/softvt-summary-b.json
reports/real-transfer-v22/scores.json
reports/real-transfer-v22/summary.json
reports/real-transfer-v22/transfer-scores.json
See evidence/sources.json for additional historical and controlled-study records.
Raw datasets, optimizer checkpoints and videos remain local.
