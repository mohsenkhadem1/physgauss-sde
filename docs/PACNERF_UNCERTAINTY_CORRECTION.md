# PAC-NeRF uncertainty audit and hierarchical correction

This is an exploratory correction developed after inspecting poor uncertainty
coverage on the two deterministic PAC-NeRF elastic objects. It is not an
independent validation dataset or a demonstrated calibration improvement until
the new posterior comparison completes its checks.

## What the baseline audit establishes

The exact-integral study produced nine converged cells and seven time-limited
prefixes. All seven were explicitly resumed on standard GitHub CPU workers;
[their run IDs and original source](../configs/pacnerf_resumptions_round1.json)
are versioned. They retain worker `219665b`, the fitted reference, retained
prefix, adapted kernel and RNG. They are never resumed under the new model.

[The evaluation-only audit](../examples/pacnerf-overconfidence-audit/audit.json)
independently checks true and projected-true positions, axis-specific bias,
interval width and image residual correlation. Torus gaps have 97–158 mm
coordinate RMSE versus 31–46 mm mean 90% interval width. Projecting the truth
into the twelve-mode space leaves coverage roughly 12–15%: the failure is not
solely the irreducible coordinate residual outside that space. Omitted modes
can still indirectly bias the image fit. Baseline adjacent-pixel residual
correlations are approximately 0.56–0.68 horizontally and 0.56–0.62 vertically.
The previous likelihood assumes independent pixels with fixed sigma 0.08.

These are descriptive signs of model mismatch, not a causal decomposition.
Raw-chain quantiles and exact density replays already passed: intervals were
not accidentally computed as Monte Carlo standard errors. Chain convergence
does not establish correct physics, global mode exploration or calibration.

[A direct-render diagnostic](../examples/pacnerf-overconfidence-audit/renderer-bias.json)
was run after freezing the first candidate. With fixed initial appearance,
the old image likelihood prefers the biased estimate over true particle
centers for Bird four-frame and Torus six-frame, but not Torus four-frame.
It prefers the estimate over the projected truth in all three checks. This
distinguishes representation and rendering effects; neither can be dismissed.
At a fixed hyperprior center, the combined correlated/error-scale likelihood
reduces some preference gaps but does not remove them. This is not posterior
or calibration evidence and did not change the candidate's physical priors.

## New physical target

The supplied initial geometry, frozen appearance, twelve mechanical modes,
camera schedules, exact linear propagators and transferred internal-noise
network remain the same. This model adds explicit *external model discrepancy*;
it does not label arbitrary full-rank noise as internally conserved forces.
Three independent standard-normal hyperparameters h_d, h_g, h_r are sampled.

For each frame interval, the existing transition has residual r_t and covariance
Q_t. Define Q_unit as the exact covariance integral of unit modal acceleration
diffusion. The new covariance is

    Q'_t = Q_t + d_0² exp(2 h_d) Q_unit.

Let r_phi² be the mean, over the 128 Gaussian centers and three coordinates,
of the squared modal projection row norm. The dimensional prior center is
d_0 = g sqrt(dt) / r_phi, with g=9.8 m/s² and dt=1/24 s. Its purpose is to allow
frame-scale unknown accelerations rather than transfer an arbitrary fixed
small cube-noise amplitude. The log-amplitude has standard deviation one;
this is a broad modeling choice, not an empirically calibrated PAC prior.

The original 24 load coefficients are retained. Four normalized temporal
Gaussian RBFs, width 0.14 s, in each of twelve modal accelerations add 48
smooth-discrepancy coefficients. Their acceleration prior center is
a_0 = (g/3)/r_phi, multiplied by exp(h_g). Thus the total 72 whitened load
coefficients a have N(0,I) prior and the mapping is

    G'_t = [G_original,t, exp(h_g) G_smooth,t].

The factor 1/3 is a declared exploratory prior choice based on gravity, not a
fit to withheld positions. No particular physical force in Newtons is claimed.
These finite-basis smooth forces and fast diffusion are distinct sources of
error. Their amplitudes can be weakly identifiable from very sparse images;
the posterior retains that uncertainty instead of plugging in an estimate.

Joint inference samples a explicitly. Marginal inference integrates exactly:

    r | path, h ~ N(0, blockdiag(Q'_t) + G' G'^T).

The force arms therefore have the same integrated physical path and scale
posterior. Any disagreement beyond Monte Carlo error is an inference warning,
not a reconstruction advantage of marginalization. The analytic conditional
force mixture uses E[Cov(a | path,h)] + Cov(E[a | path,h]). A mixture over
unknown amplitudes is generally non-Gaussian.

## Correlated rendering discrepancy

RGB residual covariance is sigma_r² (K_y ⊗ K_x ⊗ I_RGB), where K_x and K_y
are unit-diagonal AR(1) spatial correlations. Their coefficients are estimated
once from the eleven supplied frame-zero appearance residuals, clamped to
[0,0.95]. They do not use later or withheld image residuals. Their sampling
uncertainty and the appearance-fitting uncertainty remain excluded.

    sigma_r = 0.08 sqrt(1 + exp(h_r)).

This retains the original 0.08 observation component and adds uncertain
renderer discrepancy. It is a conservative modeling choice, not a guarantee
of coverage. The Gaussian likelihood includes n log(sigma_r), the spatial
covariance determinant and the quadratic whitened residual. Omitting the
learned-scale determinant would give an incorrect posterior.

## Inference, CPU plan and source separation

The current execution order is **one full Bird case first**, using
`pacnerf-bird-check.yml`. The strict final requirements are Rhat<1.01 and
bulk/tail ESS>=400 in every monitored quantity, with at least2000 retained
draws per chain. Independent raw-state replay compares against the converged
original Bird case, including 50/80/90/95% intervals and CRPS. Before seeing
that corrected posterior, the exploratory development readiness guardrails
are declared: 90% intervals cover at least80% of withheld coordinates, and
CRPS does not increase versus baseline. This combines coverage and a proper
score to discourage merely widening intervals. It is a development-scene
criterion, not an independent calibration test. The larger workflow does not
automatically start from the single-case workflow; inspect the report first.

The first candidate, Actions **39143185132** at **a575e18**, used prior-innovation
coordinates and a fixed Gauss-Newton reference. Its eight observed-only
100-burnin/100-draw pilots passed mathematical density checks but mixed poorly:
worst Rhat 3.38–3.85, minimum bulk ESS about 4.3. Its feasibility gate failed
to require useful mixing and allowed full jobs to start. That candidate was
explicitly stopped; all seven baseline continuations were untouched. The
[failed-pilot evidence](../examples/pacnerf-discrepancy-inference-audit/stopped-pilot-proof.json)
is retained. It is not reconstruction or calibrated uncertainty evidence.

The replacement retains that candidate's physical model and priors and changes
the posterior coordinates. At an observed-only reference, linearize contact
and RGB residuals and freeze the diffusion geometry. At each current material,
noise and load parameter vector psi, analytically condition this Gaussian
*guide* state path on the image surrogate. The marginalized arm eliminates
force coefficients; the joint guide retains and samples them explicitly.

For standardized physical states u, let the guide mean be m(psi) and its
precision P(psi)=L(psi)L(psi)^T. Define

    u = m(psi) + L(psi)^(-T) e,
    log|J| = -sum_i log L_ii(psi),
    U_transport(psi,e) = U_full_physical(psi,u) - log|J|.

Full nonlinear contact, causal neural diffusion and image density are evaluated;
the Gaussian surrogate is never substituted as the target. The parameter-
dependent Jacobian retains the state-volume effect missed by a fixed
profile-mode reference. Two observed-only guide linearizations and fixed
proposal metrics are built before sampling. Tests independently check the
affine determinant, roundtrip map, parameter gradients and agreement of
joint/marginal Gaussian-guide moments. No future particles or withheld images
enter either guide construction or inference.

Blocked [elliptical slice sampling](https://arxiv.org/abs/1001.0175) uses the full
target/reference ratio: 24-coordinate state blocks and five scalar reference
parameter blocks. These are reference-coordinate blocks, not independent
physical variables. Four chains start with 0.1 reference jitter. No step-size
adaptation occurs. A bracket cap raises an error instead of silently accepting
a failed update. Exact caches reuse guides at identical parameter values and
image residuals at identical observed poses; gradient calculations bypass them.
The no-gradient renderer uses separable full-grid Gaussian kernels, with
tail/log-series error below 1e-21 RGB apart from ordinary float64 rounding.
Dense-reference value checks cover that rounding. No pixels are subsampled
and volumetric depth compositing is unchanged.

The workflow runs numerical tests and independent repeated-data interval
validation, then one hard-case observed-only pilot before the other seven
pilot arms. Pilots discard 500 burnin and retain 500 draws, extending to 1000
if needed. Every monitored quantity requires Rhat<1.1, bulk/tail ESS>=100,
no frozen/invalid retained chains, stationary references and target/Jacobian
replay. Failure blocks the full comparison. Pilot states are kept in 30-day
artifacts for diagnosis. They do not score future geometry or establish
calibration. Each pilot has a 180-minute CPU cap. The sixteen
full cells start afresh with 500 burnin and retained budgets 500,1000,2000,4000,
8000; no final cell is eligible before 2000 draws. All physical and native
coordinates, scale hyperparameters, log density, covariance normalizers and conditional load means
require rank/folded Rhat<1.01 and bulk/tail ESS≥400 with no frozen chain.

Each full attempt has 330 compute minutes on ubuntu-latest. One automatic,
bounded recovery continues unfinished cases with the same target, states and
RNG for up to another 330 minutes; it does not add burnin or extend the 8000
draw protocol. Completed summaries, including failed diagnostics, are not
silently rerun. Standard public-repository CPU runners are used; no paid GPU
or external server is provisioned.

The workflow independently replays raw targets, density identities, NumPy
means/quantiles, scalar convergence diagnostics, full matrix covariance/load
integrals and original data hashes before committing a new immutable compact
snapshot. Approximate global Bonferroni/batch-MCSE checks compare marginalized
and joint path, material, hyperparameter and conditional force means. Missing/failed cells remain visible.
The resumed baseline report is used if it has been published by that point;
otherwise the original nine completed baseline cells remain available and
an overall sixteen-cell old/new ranking stays disabled.

## What uncertainty validation can establish

[The independent linear test](../examples/hierarchical-interval-validation/validation.json)
generates 2000 independent integrated-acceleration datasets with a Gaussian
global force and a genuinely random, unknown log diffusion scale. Independent
121-point quadrature includes the observation determinant and computes mixture
intervals by CDF inversion. Nominal 90% intervals cover 89.6–90.55% at four gap
times, within Monte Carlo variation. Hyperparameter posterior-CDF ranks pass
the declared 99% DKW bound; the total-variance identity agrees to 8.1e-15.
This checks hierarchical interval arithmetic, not nonlinear PAC sampling,
neural diffusion transfer or calibration under model mismatch.

For PAC, report 50/80/90/95% component coverage, widths, interval scores and CRPS
alongside reconstruction RMSE and held-out image error. Wider intervals alone
are not sufficient improvement. The independent unit tests cover Gaussian
slice moments and bit-identical checkpoint/RNG continuation. None of this
removes the need for new objects, repeated synthetic draws, unknown-geometry
tests, richer mechanics/contact and independent calibration data.

Four/six-frame schedules change camera choices and gap ranges, so their
difference does not isolate observation count. Known camera calibration,
empty background plates, initial geometry and multiview initial appearance
are supplied.

Smooth-force moments retain correlation between learned amplitudes and load
coefficients. The reported field is a non-Gaussian hierarchical mixture of
model-dependent accelerations; it excludes white-noise forcing, gravity and
known contact. There is no force ground truth or Newton-unit recovery claim.
Posterior handle penetration below the soft floor is reported so wider
intervals are not automatically mistaken for improved physical realism.

The scripts are in `scripts/pacnerf_discrepancy.py`,
`scripts/compare_pacnerf_discrepancy.py` and `scripts/pacnerf_elliptical.py`;
`.github/workflows/pacnerf-discrepancy.yml` runs the plan. New code does not alter
the four frozen old inference modules or the numerical src fingerprint.
Code, compact figures and numerical manifests enter Git. Full states are
recoverable Actions artifacts with 30-day retention; the original experiment's
separately approved permanent public archive remains unchanged.
