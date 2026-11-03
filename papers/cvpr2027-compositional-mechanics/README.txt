Compositional Neural Mechanics for Physical Prediction Beyond Visual Agreement
Anonymous CVPR 2027 review manuscript, prepared 3 November 2026.

FILES
main.tex / main.pdf: eight pages of main content, followed by the mandatory LLM
usage disclosure and references. Supplement is a separate PDF.
supplement.tex / supplement.pdf: equations, training, inference, detailed results,
reference sensitivity, failures, artifact accounting and media descriptions.
references.bib: 14 primary-source references, including close constitutive priors.
figures/: vector PDF diagrams and measured-result plots, plus PNG previews.
generate_figures.py: reproducible figure builder using existing local v16/v17
artifacts. It does not simulate, train, modify results or synthesize observations.
evidence/figure-sources.json: source hashes and qualitative selection/crop record.
release/latex-source.zip: editable source with already generated PDF figures.
release/supplement-with-videos.zip: anonymous supplement, README and 12 MP4s.

BUILD
From this directory with a standard TeX installation:
  pdflatex main
  bibtex main
  pdflatex main
  pdflatex main
  pdflatex supplement
  pdflatex supplement
Alternatively use Tectonic on main.tex and supplement.tex.
The bundled cvpr.sty and ieeenat_fullname.bst are unchanged from official
CVPR2027-v1, author-kit commit a2369342011145be8c85e5ad21a9b61f9fa53473.
No raw datasets are required to rebuild the PDFs from the supplied figures.
Regenerating figures requires the existing local raw appearance arrays and
published v16/v17 result JSON files; raw datasets and checkpoints remain local.

SCOPE
The document consolidates completed evidence. No new training or simulation was
run for this writing task. Earlier papers and experiment outputs are preserved.
Forward transfer has oracle descriptors; unknown-descriptor refinement is a
separate experiment. Dynamic comparison uses native PhysTwin and shared initial
appearance, with 8/8 versus 6/8 forecasts and conditional three-world means.
The main dynamic reference is not spatially qualified. No claim of calibrated
uncertainty, superior active probing, real-world performance or general SOTA is
made. The title and presentation do not change those scientific limitations.

SUBMISSION STATUS
Prepared for author review, not submitted. The user has not registered a
submission. No ID or author identities were invented. The official 2027 style
used here has no paper-ID field. Authors must approve the final content and
complete registration, author enrollment and conflicts in OpenReview.
Current official guidance: https://cvpr.thecvf.com/Conferences/2027/AuthorGuidelines
Abstract/profile deadline: 10 November 2026; paper deadline: 16 November 2026.
Check the official submission site for the precise deadline timezone and upload
requirements. The main PDF respects the eight-content-page limit; extra content
is confined to the separate supplement. AI assistance is disclosed.
