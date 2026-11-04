CONDITIONAL NUMERICAL REFERENCE POSTERIOR
Independent audit, 4 November 2026

Question
Can a small reference inverse problem numerically resolve the coupled uncertainty
between support stiffness and support location before we trust learned physical
parameter intervals?

Conditioning and scope
Four NEW synthetic worlds, seed729131. Only support log-stiffness and x-location
are unknown. Material modulus, load magnitude/location, damping, geometry, mass,
initial state, camera, force direction and boundary type are fixed to truth.
The same corrected acquisition model uses exactly one 50 micrometre Gaussian
noise draw per 2D track coordinate. This is a conditional two-dimensional posterior,
not a six-parameter posterior. Its intervals must NOT be compared directly with
unconditional MDN/flow intervals. Four worlds cannot establish calibrated coverage.

Numerical method
The prior is independent uniform in normalized q, corresponding to uniform
log-stiffness over approximately[11.8,14.2] and position over[-.035,.035]metres.
We use float64 mechanics, a full-prior17x17 scan, six separated multistart bounded
MAP searches, then an affine-whitened two-dimensional trapezoidal integral.
The Jacobian determinant and normalized prior are included in the evidence.
Explicit tests require:
- log-evidence changes <=.001;
- posterior-mean changes <=.005 posterior standard deviations;
- 5th/50th/95th-percentile changes <=.02 standard deviations;
- outer integration-edge mass <1e-5.
The model, source hashes, seeds and thresholds were frozen before observations.
This is operational numerical convergence with a mode search, not a mathematical
proof that all possible remote modes have been excluded.

Results and preserved failure
The initial protocol accepted worlds0,1,3 after domain expansion and resolution
refinement. World2 FAILED the quantile-stability threshold. Evidence and edge mass
had stabilized, but expanding the domain on a fixed65x65 grid coarsened the spacing
and moved the marginal quantile estimates. That failure remains in results.json.

A separately frozen supplemental protocol kept spacing constant while expanding
world2's domain: extent4/n129, extent6/n193, extent8/n257. It then compared n129
and n257 on the SAME extent8 domain. All unchanged thresholds passed. The final
resolution quantile change was .00552 posterior standard deviations, with evidence
and mean changes effectively zero. The supplemental result is stored separately
in world2-fixed-spacing.json; no initial failure was overwritten.

Posterior geometry
World0: stiffness/location correlation +.941
World1: stiffness/location correlation +.920
World2: stiffness/location correlation -.980 (supplemental refined diagnostic)
World3: stiffness/location correlation +.787
All accepted conditional 90% marginal intervals contain their corresponding true
parameters. This is four-case sanity evidence only, not a calibration claim.
posterior-geometry.png shows all four densities with the truth markers.

What this changes in our decisions
1. Strong coupling exists even in this two-parameter conditional problem. Treating
   parameter errors or uncertainty dimensions independently is inadequate.
2. A naive grid can appear evidence-stable while marginal quantiles are not stable.
   Resolution and tail tests must be separated instead of relying only on mass.
3. These results do not explain or validate the unconditional MDN support-location
   coverage. A matched-conditioning learned/reference comparison remains future
   work; the present result is deliberately narrower.
4. Forward-model adequacy still comes first for real data. The reported sloth
   out-of-bank residual7.04mm and context norm33.8 vs training max9.88 require
   checking rigid/deformation separation, initial pre-load, contact/fixture
   assumptions, timing and correspondence geometry before larger posterior models.
   Covariance-aware density modeling cannot repair a missing physical mechanism.

Artifacts and execution
- results.json: initial protocol, including its failed world2.
- world2-fixed-spacing.json: separately frozen supplemental numerical result.
- summary.json: combined provenance-aware summary; never rewrites initial status.
- posterior-geometry.png: all four conditional densities.
- tail_refinement.py and render_reference.py: supplemental/replot scripts.
- scripts/gaussian_boundary_flow_v02/reference_posterior.py: primary runner.
- tests/test_gaussian_boundary_reference_v02.py: 4 tests passed, including known
  correlated-Gaussian normalization, covariance/quantiles and convergence gates.
- runs/gaussian-boundary-flow-v02-reference: frozen plans, observed data and grids.

Only GPU0 was used. Numerical runs totaled about299 seconds, well below20 minutes.
Reproduction uses the repository virtual environment; run the main module with
python -m scripts.gaussian_boundary_flow_v02.reference_posterior. For supplemental
scripts in this report directory, set PYTHONPATH=. from the repository root.
