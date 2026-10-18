# Corrected PAC-NeRF comparison status

The corrected full comparison has not produced results yet. See the repository
README for the active Actions run and `PACNERF_EXACT_PROTOCOL.md` for the frozen
plan. The earlier four-point-quadrature results are superseded.

The current observed-only development pilot uses exact linear integrals, exact
innovation coordinates and a fixed full-RGB Gauss-Newton sampling metric. It
compares neural/constant noise crossed with marginalized/explicit loads.

The full workflow will independently replay its saved states and scores,
retain failed or unfinished cells with clear labels, and commit its compact
report plus this results document. It cannot claim a ranking unless all
predeclared convergence, pair-agreement, replay and numerical gates pass.

Initial geometry and eleven initial appearance views are supplied. The
12-mode elastic graph is not the original PAC-NeRF MPM solver. Two synthetic
elastic objects and descriptive interval coverage cannot establish calibrated
force uncertainty or unknown-geometry monocular reconstruction.
