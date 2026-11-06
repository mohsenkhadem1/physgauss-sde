Solver-driven dense Gaussians: controlled RGB-to-mechanics qualification

This extends the dense soft-constrained PIDG study without replacing its public-data results.
Inference learns 11 physical coefficients from a known-volume object's observed RGB prefix.
Dense Gaussian centers, covariances and material-frame appearance follow integrated tetrahedral
mechanics. No temporal motion network is queried for the final forecasts.

Environment: official PIDG environment plus released upstream renderer; configure() in
scripts/pidg_comparison/run.py establishes its CUDA paths. Use PYTHONPATH=. from repo root.
Python used: ~/.local/share/physgauss-external/pidg-official-v01/env/bin/python.
Large local inputs: shared-law.pt in runs/compositional-neural-mechanics-v16/scale-separated-prototype;
appearance PLY in runs/compositional-dynamics-v17/matched-appearance-native-resolution.
These pretrained inputs and raw trajectories are intentionally not published in Git.

Primary sequence (see --help for GPU/output options):
  reference.py -> make_images.py
  proposal_bank.py (learned and analytic) -> refine_physics.py (80 steps each)
  finish_bank.py selects on reserved calibration RGB, predicts all five cases, and runs pair.py
  evaluate.py scores sealed forecasts and creates six comparison videos
  ambiguity.py -> reference.py --cases force_reverse --output .../reference
  evaluate_ambiguity.py selects a hypothesis on a force probe and scores another force
  test_dynamics.py; test_render_gradient.py; build_report.py
The proposal bank and refinement path is the adopted RGB identifier. The archived
continue_campaign.py reconstructs tracks, identifies forces, and refines that initialization;
this diagnostic failed and is preserved, not silently substituted for the successful fit.
fit_rgb.py and evaluate_temporal.py test fixed-appearance temporal architecture adaptations;
they are not the full PIDG momentum/flow recipe. Existing full public PIDG comparisons remain
in scripts/pidg_dense_mechanics and reports/pidg-dense-mechanics-v01.

Primary run: runs/solver-driven-gaussians-v02. Reports: reports/solver-driven-gaussians-v02.
Selected fits and per-case seals are copied to report metadata for review. SHA256 seals precede
held-out scoring. Cases share one geometry/material: no population confidence intervals.
Forecast rendering uses 320x240, fixed appearance and four calibrated cameras; 2/3 are withheld.
Video playback is 5x slow motion (0.008 simulation seconds per 25-fps frame).

Known limitations: supplied initial volume, density, appearance, camera calibration, floor and
control histories; one fitted scale of a fixed learned material-family descriptor; Gaussian
compliant target rather than a unilateral poker; finite-penalty floor/pair contact without
friction or CCD; reused material for second object. 36.79% surface splat centers use affine
extension outside supplied volume. Reference mesh refinement remains material at sub-mm scale.
The two-hypothesis probing example demonstrates a scale observability ambiguity, not calibrated
uncertainty, a learned active policy, or broad public-data intervention superiority.
