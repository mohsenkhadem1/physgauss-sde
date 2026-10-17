# External-force distributions from sparse RGB observations

Version 0.4 adds a conditional posterior over smooth external nodal loads. It compares force-only, fixed-amplitude internal SDE-only, and combined models. This is a restricted force-field inverse problem on the existing known geometry/camera, not arbitrary unseen force-location or contact inference.

## Force prior and units

Let `X_i` and `m_i` be reference nodal coordinates and lumped masses. A Gaussian spatial profile is centered on the free-node centroid, has length scale 0.6 m, and is zero on pinned nodes. Its mass-weighted nodal weights are normalized to sum to one. For total world-frame load `q(t)`:

```
w_i = free_i * m_i * exp(-||X_i-center||^2/(2*length^2)) / normalization
F_external_i(t) = w_i * q(t)
sum_i F_external_i(t) = q(t)
```

`q` and nodal forces are in N. The solver receives acceleration `F_external_i/m_i`; force and acceleration are not interchanged. This is a material-attached load profile whose vector components are expressed in the world frame. External net force/torque need not vanish. The internal central SDE impulses retain their existing force/torque balance, before support constraints act.

On the present cube, the four free corners are equally distant from the profile center. Their radial factors are equal, so the normalized load is mass-proportional and gives uniform external acceleration on free nodes. Spatial length therefore does not change this particular load pattern. This milestone tests force-history uncertainty, not spatial load localization.

Five equally spaced temporal inducing points and three directions give 15 independent standard-normal coordinates. For a unit-variance Matérn-3/2 kernel with length scale 0.15 times the recording duration, `K_ZZ=L L^T` (jitter 1e-8), define:

```
Phi(t) = k(t,Z) L^-T
q(t) = 0.05 N * Phi(t) A,       vec(A) ~ N(0,I_15)
```

The covariance is `0.05^2 * Phi(t) Phi(s)^T` independently for each force direction. This is a finite inducing projection, not the full Matérn GP: the conditional residual covariance is excluded. Its prior variance is approximately the declared variance at inducing points and can be smaller between them. Queries outside the recording interval are rejected rather than presenting a vanishing projected variance as confident extrapolation. No temporal length scale, spatial location, kernel hyperparameter or diffusion-network parameter is learned from the video in this milestone.

## Observation-only inverse problem

Coordinates contain eight standardized material/initial-state parameters, optional force coefficients, then optional Brownian increments. All have declared independent standard-normal priors. Density, rest geometry, supports, appearance, camera poses and observation noise remain known. The posterior is proportional to the full observed-image likelihood times these priors. The likelihood uses only the available RGB frames; missing frames and force ground truth are never supplied to fitting.

Each model gets its own nonlinear MAP fit and Gauss-Newton Gaussian proposal, including the existing damped/line-searched refinements. This is proposal preconditioning, not a posterior approximation assumed exact. An optional nonlinear-curvature branch estimates the Hessian by central differences (step 1e-3) of automatic gradients and can attempt two additional Newton polishing steps. Positive curvature eigenvalues are floored at 0.2; nonpositive curvature causes a Gauss-Newton fallback. A pilot encountered large negative finite-difference eigenvalues near renderer depth-order changes and poor proposal behaviour. The main study therefore fixes `exact_hessian=false` before its run. The branch, floors, stationarity diagnostics and any rejected curvature are recorded. Neither reference is assumed to describe the final posterior exactly.

For any fitted normalized reference `q_ref=N(m,H^-1)`, use the exact correction:

```
r(z) = log L_nonlinear(z) - ||z||^2/2
       + ||L_H^T (z-m)||^2/2 - logdet(L_H)
```

Standard Gaussian constants cancel between the target prior and reference. The existing pCNL/Metropolis sampler uses this ratio with full nonlinear gradients. Unlike the historical linearized-evidence sampler, this formula remains valid after nonlinear proposal refitting. Proposal adaptation ends after warmup. The force study uses 75% within-model pCNL and 25% independent reference refresh, without discrete scale-transport moves. Model comparisons are conditional fits; there are no inferred model probabilities or Bayes factors.

This protocol differs from the historical common-linearization study. Do not compare its old numbers directly with new conditional-model results. All force-study methods use the same images and proposal-building procedure. The SDE amplitude is fixed at the calibrated unit scale; a force/noise scale hierarchy is not implemented here.

## Posterior fields and impulses

The saved fields include posterior total-force mean, componentwise 5-95% intervals and 3x3 marginal covariance at each solver time, plus external impulse mean/covariance. Raw run files retain paired external nodal-force samples, stochastic impulses, finite-window combined forces, and the full cross-time covariance of the total external load. Geometry and spatial profile are fixed, so the external nodal field inherits the same coherent temporal coefficients rather than independent jitter.

Internal white-noise forcing has no ordinary pointwise force value. Each solver-step impulse is logged using the exact pre-step state and Brownian increment. Over each frame interval, the reported combined nodal average is:

```
average_force = (sum_steps F_external*dt + sum_steps stochastic_impulse) / interval_duration
```

This quantity is in N and depends on the stated averaging window. External impulses are in Ns. Gravity, elastic forces and drag are part of the mechanics, not silently added to the reported unknown external load. The SDE-only model's external load is a conditional point mass at zero because that component is excluded; it is not evidence that the external load is absent.

Every latent coordinate and observed log likelihood is checked with rank/folded Rhat, bulk/tail ESS and frozen-chain checks. External-impulse functionals have separate diagnostics using every retained state. A rendering subset controls memory and does not create independent experiments. Failed diagnostics leave force intervals and scores provisional. Even a converged posterior remains conditional on the material, geometry, camera, load-profile and bandwidth assumptions.

## Controlled protocol and artifacts

`configs/force_study.json` declares four conditions (no load, deterministic external pulse, internal stochastic forcing, both), three models, and seed 4101, separate from earlier seed sets and calibration seed 918. Each model uses four chains, 500 warmup transitions and 1,000 retained transitions per chain. One realization per condition is a numerical investigation, not a coverage-calibration study. The spatial force pattern is restricted by design; the pulse waveform is not fitted as a known pulse template. An evaluation-only least-squares force-basis oracle quantifies waveform representation error and never chooses the prior.

After inference, a separate logger reconstructs true substep external loads and internal impulses and verifies equality with the original benchmark trajectory. Evaluations include force/nodal-force RMSE, CRPS, coverage and interval width; external impulse scores; finite-window combined forcing; withheld trajectory/rendering scores; and mechanical diagnostics. All observed RGB tensors, network weights, source hashes, numerical environment, fitted references and chains are retained in raw runs. Resume is allowed only at completed-model boundaries with an identical plan, source and numerical environment.

```sh
physgauss check
physgauss forces --quick --output runs/force-smoke
physgauss forces --plan configs/force_study.json --output runs/force-study
physgauss forces --plan configs/force_study.json --output runs/force-study --resume
```

The quick command is an integration check with a smaller scene and short chains; it cannot establish convergence. Open `report.html` for posterior load bands and diagnostics. Per-model `fields.json` provides compact distributions with explicit units; `force_samples.npz` contains joint sample/covariance arrays, and `posterior.pt` holds observations, reference and full chains. Large raw files remain ignored by Git. Compact report/evidence snapshots are versioned separately after completed experiments.

## Remaining identifiability limits

Material uncertainty and force history can explain the same observed motion. Absolute N units depend on the known mass/density and metric scale. The current spatial pattern cannot locate an unknown contact or infer an arbitrary three-dimensional force density. Different unseen geometries, camera errors, lighting changes or an incorrect material law can be absorbed by an apparently plausible load. These alternatives require new model components, experimental constraints and independent validation; broader posterior bands alone cannot identify the cause.

References: [Latent Force Models](https://proceedings.mlr.press/v5/alvarez09a.html); [GP joint input-state estimation](https://arxiv.org/abs/1904.00093); [pCN](https://arxiv.org/abs/1202.0709). The nonlinear rendering, force basis and proposal protocol above are this repository's controlled implementation, not a reproduction of those papers.
