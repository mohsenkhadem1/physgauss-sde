# Step 3 tuning log (development mode; freeze these before any prospective run)

| setting | value | chosen from / with knowledge of |
|---|---|---|
| discrepancy v1 (`--hyp d0`) | single Kronecker kernel Matern_t x SE_s, hypers from commanded-D0-only fit | pre-specified before results; gate failed |
| discrepancy v2 (`infer`) | v1 kernel, hypers inferred jointly | declared after a reduced-budget gate smoke test failed; full run stopped after 0F0S (evidence +0..+12 only) |
| `freq` (modal-frequency nuisance) | 6 lowest-mode factors +-5% | truth diagnostic (`step3_diag.py`) showed it removes almost nothing; not run |
| discrepancy v3 (`two`) | a^2 I + b^2 SE_s, fit on commanded D0+D1 | 0F0S evidence barely changed; gate run stopped |
| discrepancy v4 (`shared`, used for all reruns) | episode-shared + episode-specific parts, each a^2 I + b^2 SE_s, common Matern_t | **truth diagnostic** (model at true parameters vs truth: at 10 N ~85% of the coarse-model error is shared across episodes); hypers fit on the **commanded D0 + D1** episodes (uses D1's command, also for the hidden runs) |
| predictive discrepancy for the true response | smooth parts only, sd = sqrt(b^2 + d^2) | choice; per-marker parts treated as tracking error |
| force patch shape (`vol`) | uniform body load in a 1 cm layer under a soft square of half-size h in [5, 40] mm | **truth diagnostic** (`step3_forcediag.py`): surface traction under-reads the 1 cm deep box press by 4-10% (h=20) and more for smaller h; the 1 cm depth copies the protocol press class |
| support patches | unchanged from step 2 (Gaussian, rank-3 spring) | - |
| sampler budget | support configs 2048 x 12, others 1024 x 10 | coordinator request; not tuned |
| risk threshold | 0.25 mm absolute, single definition (report + planner) | chosen from step-2 error scales (Fig-1 ours 0.30-0.36 mm, COMSOL 0.08-0.25 mm) |
| low-amplitude probe candidates | bank presses at x = +0.05, +0.06 (truths pre-generated) | +0.05/+0.06 = bank sites nearest the step-2 MAP force window |
| force prior | per-component scales tau_x,y,z (each U[ln .1, ln 10]) | **truth-informed diagnostic**: with one shared tau the posterior tau~0.2 (x,z ~0) shrank Fy by ~0.8 N of a 1.5 N bias (low Bright); first vol queue (shared tau) superseded after 1 case, kept in step3/superseded/ |
