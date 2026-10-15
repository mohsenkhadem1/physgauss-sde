# Milestone 2: infer when process noise is needed

## Problem addressed

The first ablation used a fixed, simulator-calibrated noise field. It helped on stochastic truth but produced unnecessarily wide intervals on deterministic truth. Increasing diffusion is not a substitute for posterior uncertainty over an initially unknown scene. We now give the process-noise scale its own posterior, including an explicit atom at zero.

## Declared hierarchy

The physical state and noise network are unchanged. A nonnegative scalar alpha multiplies the correlated Brownian forcing. In the default study, the scale grid is `[0, 0.125, 0.25, 0.5, 1, 2, 4]`.

The prior assigns probability 0.5 to alpha=0. The remaining mass is a discrete approximation to a unit half-normal: bins are separated by geometric grid midpoints, and the last bin includes the entire tail. This is a declared discrete prior. It is not a proof that the continuous alpha integral is accurately approximated. The output also reports sensitivity to predeclared zero-noise masses 0.25 and 0.75; these alternatives are never selected using held-out performance.

## Common local likelihood

First fit a deterministic physical reference to observed images. Around its standardized scene coordinates z_ref and zero stochastic increments, construct ONE image Jacobian, whitened by the known image-noise standard deviation:

```
whitened_observed - whitened_f_ref
    ~= J_scene (z_scene - z_ref) + alpha * J_noise w + epsilon

d = whitened_observed - whitened_f_ref + J_scene z_ref
A_alpha = [J_scene, alpha * J_noise]
d ~= A_alpha eta + epsilon

eta ~ N(0, I)
epsilon ~ N(0, I)
```

All three controls share this linearization, priors, material model, observation noise, and observed images. The controls are alpha=0, alpha=1, and posterior averaging over the scale grid. They deliberately use a different approximation from milestone 1's separately fitted nonlinear posteriors, so the old two-case scores are not directly comparable with this study.

## Integrated evidence and conditional posterior

For each alpha:

```
H_alpha = I + A_alpha.T A_alpha
m_alpha = solve(H_alpha, A_alpha.T d)
eta | d, alpha ~ N(m_alpha, inverse(H_alpha))

negative_log_evidence = 0.5 * (
    norm(d - A_alpha m_alpha)^2 + norm(m_alpha)^2 + logdet(H_alpha)
)

p(alpha | d) proportional to p(alpha) * exp(-negative_log_evidence)
```

Only an additive constant common to every scale is omitted. These equations integrate both standardized scene parameters and Brownian increments for the linearized model. The log determinant accounts for latent prior volume; optimizing alpha and latent increments jointly without this integration would be a different method.

Computations use latent-sized Cholesky factorizations instead of a dense image covariance. Unit tests compare the evidence, means, and covariances against dense observation-space calculations. At zero scale, unused noise coordinates integrate out exactly. If noise directions are unobserved, scale probabilities correctly remain at the prior.

## Physical posterior predictions

Draw a scale from its posterior, then draw the complete correlated scene/noise vector from that component. Run the original nonlinear physical simulator and renderer for every draw. We keep joint correlations, including correlations between material parameters, initial state, and forcing. Gaussian shape covariance is still distinct from posterior uncertainty. Invalid physical samples are not silently filtered.

This construction is exact only for the local Gaussian observation model. It is approximate for nonlinear elasticity, time integration, and alpha-composited rendering. We explicitly measure disagreement between exact nonlinear renders and the local approximation in units of known image-noise sigma. Large disagreement undermines the evidence approximation and should motivate iterative smoothing or stronger inference.

## Observation-only inference and evaluation

The scale inference function receives the model, observed images, observation noise, and fitted reference. It receives no withheld image or ground-truth motion target. A test checks that the unmodelled force flag changes truth while leaving inference-model predictions unchanged.

The study uses seeds 3101, 3102, and 3103, separate from diffusion calibration (918) and the original exploratory realization (1729). Each condition has two schedules, each containing six input images over the same 21 timestamps:

- Short gaps: `[0, 4, 8, 12, 16, 20]`, maximum observation spacing 0.2 s.
- Long gap: `[0, 3, 6, 14, 17, 20]`, maximum spacing 0.4 s.

The schedules cover different motion phases, so this is an observation-design comparison, not a universal causal estimate of gap length. Primary tables score ALL withheld timestamps; a largest-gap diagnostic is also retained per case.

Conditions are deterministic truth, stochastic truth, and deterministic truth with a brief external acceleration pulse absent from inference. The last condition probes model mismatch. Its nonzero effective noise must not be interpreted as evidence that the underlying world was stochastic. Process-presence Brier scores are recorded only for the correctly specified deterministic/stochastic conditions.

## Metrics and replication

- Position RMSE, componentwise CRPS, vector energy score, image PSNR, and novel-view PSNR.
- Coverage at 50%, 80%, 90%, and 95%, read with interval width.
- AUSE and Spearman error-uncertainty correlation, which assess ranking rather than calibration.
- Scale probabilities, scale-prior sensitivity, physical diagnostics, and local-likelihood approximation error.

There are three independent noise realizations per condition/schedule. Frames, pixels, coordinates, posterior draws, and repeated schedules are not independent experimental replicates. Error bars show standard deviation across seeds, not confidence intervals. One known geometry does not establish cross-scene calibration.

## Reproduce and resume

```sh
physgauss check
physgauss study --quick --output runs/scale-smoke
physgauss study --plan configs/scale_study.json --output runs/scale-study
physgauss study --plan configs/scale_study.json --output runs/scale-study --resume
```

The study writes a manifest, per-case records and posteriors, a progress record, aggregate CSV/JSON, scientific plots, and a self-contained HTML report. Exact source and plan matching are required for resume. Source changes during a study stop the run, preventing mixed-provenance results. Per-case records are written atomically after their posterior exists.
