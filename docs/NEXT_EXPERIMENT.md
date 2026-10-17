# From the controlled benchmark to partial-view reconstruction

This is the development sequence after the version-0.3 nonlinear sampler investigation. Version 0.4 implements an initial nonlinear-curvature proposal refit and a restricted structured external-force prior; see `FORCE_POSTERIOR.md` for actual capabilities. Broader spatial force inference, viscosity, uncertain geometry/camera and real video below remain proposed. The target remains one moving camera with missing frames and actual irregular timestamps.

Version 0.6 adds density-correct second-order dynamics and continuous depth integration; see [ACCURATE_PIPELINE.md](ACCURATE_PIPELINE.md). Gaussian force elimination, matched solvers and a deterministic independent-reference rerun are implemented. Nonlinear posterior validation, repeated-realization calibration and stochastic/mesh refinement still precede broader model claims.

## First resolve nonlinear exploration

A Metropolis kernel with the correct invariant distribution can still mix slowly. Refit the proposal reference around a nonlinear stationary point for each scale, with line-searched Gauss-Newton steps. Treat those fitted Gaussians as arbitrary proposals: evaluate their normalized log densities and the complete nonlinear target in the acceptance ratio. Do not reuse the current `L_nonlinear/L_linear` shortcut after changing a reference without re-deriving its normalization. Freeze all fitted proposal parameters before retained sampling, or use a formally valid adaptation scheme.

Compare the current common reference, component-specific nonlinear references, and convergence-checked fixed-scale controls. Monitor physically relevant predictions as well as latent coordinates. Retain strict scale diagnostics and inspect occupation/switching: a constant scale trace is ambiguous. Use independent chains and more than one initialization region. Validate a smaller physical problem against dense quadrature or an independent sampler before enlarging the model. Predictions from flagged chains remain diagnostic outputs.

## Separate missing forces from process noise

Introduce a low-dimensional deterministic external-force model before giving the SDE more flexibility. A first controlled option is a smooth temporal basis multiplying physically specified nodal force patterns:

```
f_external(t) = sum_j B_j(t) * b_j
b ~ N(0, K_force)
```

Keep its units, supported nodes, and external momentum exchange explicit. Compare four matched hypotheses: deterministic mechanics; mechanics plus structured force; mechanics plus stochastic impulses; both. Infer force coefficients jointly with stiffness, damping, initial state and stochastic increments using observations alone. Fix model priors before evaluating targets. Reuse the hidden-pulse experiment, then vary pulse shapes and directions on new realizations. A positive SDE scale is an effective discrepancy parameter until these alternatives are distinguished.

## Make dissipation a material property

The existing mass-proportional drag is external dissipation. Add objective internal viscosity, for example using spatial strain rate `D = sym(dot(F) F^-1)` and viscous Kirchhoff stress proportional to positive shear/bulk viscosity coefficients. Derive nodal viscous forces consistently from the selected stress and volume convention. Require nonnegative continuum dissipation, zero internal net force/torque, frame objectivity, and time-step convergence. Check the discrete energy balance including gravity, support work, viscosity and stochastic work.

Do not identify physical process noise with thermal fluctuations unless its covariance is derived consistently with dissipation and temperature. The current neural noise is a calibrated synthetic forcing prior. Contact, friction, near-incompressibility and plasticity require their own constitutive/solver changes; a larger photometric loss weight cannot enforce them. Near-incompressibility also needs a discretization that avoids volumetric locking.

## Introduce camera uncertainty with a fixed gauge

Start from camera estimates and their uncertainty, preferably using static background information. Add small pose perturbations in the six-dimensional Lie algebra of SE(3), with a temporal prior. Infer them jointly with deformation rather than interpreting camera error as object motion. Fix the first pose and retain an independent metric-scale anchor; otherwise camera/object/scale ambiguities remain. Keep intrinsics fixed in the first camera experiment, then test uncertain calibration separately.

Use controlled camera perturbations with known truth. Report pose rotation/translation error in the declared gauge, trajectory error, held-out view error and interval calibration. Include a static-object test to check that pose error is not absorbed into material deformation, and a fixed-camera test to check the reverse ambiguity.

## Then infer partially observed rest geometry

Use a low-dimensional rest-mesh deformation prior or a geometry ensemble, initially with fixed topology. Recompute positive rest volumes, reference gradients, masses and Gaussian attachment locations for every geometry hypothesis. Maintain a consistent density/metric-scale assumption; geometry, density and stiffness cannot all be identified freely from sparse images. Reject invalid meshes explicitly. Distinct unseen geometries should remain distinct hypotheses when observations cannot resolve them.

Evaluate visible and unobserved regions separately using held-out geometry and views. Report geometric error, coverage with interval width, proper scores where the target representation permits them, and sensitivity to the geometry prior. A generative completion is a prior proposal, not an additional observed frame. Independent Gaussian jitter around one completed surface does not represent uncertainty over unseen topology.

## Real-video input contract

The future loader should retain the original RGB frames, actual timestamps, masks if available, camera intrinsics, pose estimates/priors and split assignments. Missing frames must remain absent from the observation likelihood. Keep an immutable input manifest and hashes. Start with a short controlled capture with scale, static background, known supports and approximately stable lighting; add unknown lighting/appearance before interpreting photometric residuals as physical forcing.

Train/fitting access and evaluation access must remain separate. Declare temporal gaps and novel-camera tests before inference. Report observation count and camera coverage alongside scores. Expand beyond the current single geometry and three realization seeds before making a calibration or state-of-the-art claim.
