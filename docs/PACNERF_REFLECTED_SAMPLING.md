# Sampling the existing soft floor with temporary walls

This changes inference, not the physical contact law or posterior. A reflection
acts on artificial MCMC momentum. It is not a simulated impact of the object.

For fixed material/noise parameters, the parameter-conditioned Gaussian guide
maps state coordinates linearly into physical heights: h = c + A w.
The conditional state density has the form

    p(w) ∝ Normal(w;0,I) exp(R(w)) ∏ sigmoid(h_i/s),  s=0.002m.

R includes the complete dynamics density, nonlinear contact, state-dependent
neural covariance, observed RGB likelihood and guide Jacobian, with the sigmoid
floor factor removed. Introduce

    u_i | w ~ Uniform(0, sigmoid(h_i/s)),  b_i=s logit(u_i).
    p(w | u) ∝ Normal(w;0,I) exp(R(w)) 1{c+A w ≥ b}.

Integrating the auxiliaries restores the original sigmoid product. There is no
replacement by a hard physical floor and no missing parameter normalization.

Residual-gradient kicks alternate with harmonic Gaussian flow reflected at
the temporary linear sampling walls. Intersections are solved analytically;
momentum reflects in the wall normal. This Gaussian-flow component follows
[Pakman and Paninski](https://arxiv.org/abs/1208.4118). A Metropolis correction
uses the augmented Hamiltonian, including numerical energy error. Numerical
invalidity or a reflection cap rejects the proposal. The full non-Gaussian
posterior is not replaced by a quadratic approximation.

The auxiliary walls are discarded before a joint five-parameter elliptical
slice update using the original marginal soft-floor target. Saved log densities
also use that marginal target. Physical contact, force conjugacy and model
priors stay unchanged. The Gaussian reference is learned from discarded
observed-data states; its centre is not claimed to be a stationary mode.

## Interweaving physical and innovation coordinates

The new optional `--interweave` update alternates two exact parameter
conditionals. The ordinary update holds bridge innovations fixed, allowing the
physical path to change with material/noise parameters. The added update holds
the physical path fixed and samples the five parameters directly. A change of
coordinates then maps the unchanged path back into the bridge. This addresses
the scale/path coupling without changing the model or forcing the chains to
agree through a shared random state.

For physical parameters psi = mu + A v and fixed physical path x, its
elliptical-slice residual is

    R_centered(v) = log p(x, mu+A v, observed RGB, F=1) + ||v||²/2.

The bridge Jacobian is absent from this physical-coordinate conditional.
The five-parameter affine Jacobian is constant. The subsequent bridge mapping
restores the full saved innovation-coordinate density, including its Jacobian.
An independent nonlinear-funnel quadrature test detects incorrectly adding
the bridge Jacobian. Actual neural joint/marginal tests check path preservation;
a single-chain test checks bit-identical warmup and retained continuation.

### Invariant measure of the composed sweep

Write the unchanged physical posterior as pi(psi,x), with a linear Gaussian
bridge x = mu(psi) + L(psi)^(-T) xi. The full change of variables is triangular:
the material/noise parameters are unchanged by the bridge. Thus its determinant
is det L(psi)^(-T), even though the path also depends on psi. The outer fixed
affine reference contributes only a constant determinant.

The innovation-coordinate target is therefore proportional to
pi(psi,x(psi,xi)) times |det L(psi)^(-T)|. The reflected state update and the
ordinary parameter slice preserve conditionals of this density. Mapping to
physical coordinates, sampling psi conditional on fixed x using pi itself,
and mapping back also preserves that same innovation-coordinate measure.
Composing these invariant kernels needs no additional acceptance step. The
composition need not itself be reversible. All adaptation is discarded before
retained sampling; the fixed reference is shared, but seeds and warmup are
independent per chain.

The Gaussian-force integral is unaffected: at a fixed path, the transition
residual is affine in force coefficients, its covariance is independent of
those coefficients, and images/floor feasibility depend on the path. Both
transition and force-integral log determinants remain in every physical
density evaluation. Marginalization reduces the sampling dimension by72 and
allows conditional-moment force estimates; it does not repair a misspecified
mechanical model or establish uncertainty calibration.

## Four-worker Bird protocol

`pacnerf-bird-reflected.yml` prepares one fixed analytic state guide using the
discarded3350-draw observed-data checkpoint. It relinearizes at the physical
trajectory mean of2048 evenly selected parent states, removes floor curvature
from the guide only, and learns only five-parameter moments. The state Gaussian
is the analytic bridge rather than an empirical full covariance. Preparation
checks the old cached density and the preserved physical starting trajectories.

Four independent standard GitHub CPU workers use distinct seeds970009–970012,
500 discarded warmup transitions and planned2000 retained draws each. Each
retains a different parent endpoint; parent samples never enter the new
posterior. The first256 retained draws are archived for early mixing inspection;
each worker then continues its exact checkpoint to2000. No paid service is used.
Learning time is research development cost, not a fair runtime comparison.

The aggregator rejects different sources, references, inputs, seeds/indices or
incomplete budgets, replays the density and monitors physical states, native
states, log density,72 conditional force means and both covariance normalizers.
Only four-chain diagnostics count. A passing convergence gate still requires
independent uncertainty replay; it does not start the wider comparison.

The active run is [39399618548](https://github.com/mohsenkhadem1/physgauss-sde/actions/runs/39399618548),
immutable worker`ad3ec582aaa5e5ff1f4fdddf76318bd9d0bde031`. Preparation passed
the full188-test suite on GitHub and the four chains started. The local full
regression ran187 tests in282.911s; the additional interwoven recovery test
passed separately, bringing distinct tested checks to188.

`pacnerf-bird-reflected-continue.yml` can resume all four original shards or
extend to4000/8000, using that exact worker commit and the same reference blob,
initial states, warmup and saved RNG. It rejects source/reference/kernel drift
and proves the old retained prefix unchanged. It has not been dispatched.
The bounded follow-up workflow listens for completion of the initial run;
it extends a failed2000-draw convergence gate to4000, then8000 if needed.
Continuation workflows route their own next step with `workflow_dispatch`.
This avoids recursively chaining `workflow_run` jobs. Dispatching with the
repository token is supported by [GitHub's documented exception](https://docs.github.com/en/enterprise-cloud@latest/actions/how-tos/write-workflows/choose-when-workflows-run/trigger-a-workflow).
It checks the exact shared reference SHA and numerical-source fingerprint,
accepts only the one Bird case, stops at8000, and never starts the full matrix.
Failed/partial workers preserve checkpoints but do not trigger an unbounded retry.
If a dispatch returns an unknown outcome, inspect the remote runs before
rerunning the coordinator; a route record is written before the API call.

`pacnerf-bird-reflected-validate.yml` reruns the complete raw convergence gate
before opening evaluation particles. It compares the full-image shader against
the dense shader, independently checks the transition integrals and floor
likelihood, decodes states in NumPy and checks all interval/proper scores against
the Torch path. It independently replays the original baseline and applies the
existing80%-coverage/CRPS/floor development guardrails. A unit test verifies that
a failed fresh convergence gate cannot open any tensor labels. This workflow
has not been dispatched; no uncertainty result is claimed. The bounded
follow-up will dispatch it when the observed-only convergence gate passes.

A separate real-data pipeline integration check ran four16-draw, zero-warmup
shards with seeds990009–990012. Raw target replay passed, all force/state
diagnostics were calculated, and production eligibility was correctly false.
These samples are not mixed with the main run or used as convergence evidence.

The actual new Bird checkpoint also passes recovery: from the GitHub chain0
256-draw preview after500 warmup, two local runs added32 versus16+16 draws.
All samples, saved log ratios, energy errors, final position/density, RNG,
acceptance/reflection counters and parameter-slice counters matched bit-for-bit.
The original256-draw prefix is exact. See
`examples/pacnerf-bird-mixing-audit/reflected-real-checkpoint-recovery-proof.json`.
These local diagnostic continuations are not pooled with the four GitHub
chains. This checks continuation on the same local environment; it does not
assert bitwise equality of future trajectories across different CPU platforms.

## Validation and current status

### First four-chain preview (256 retained after500 warmup)

All four main workers reached the preview and continued their exact checkpoints
to2000. The full gate still fails: worstRhat1.87275, minimum bulkESS5.688,
minimum tailESS11.186. There are no invalid/divergent retained state steps.
518 of733 monitored quantities fail. Stiffness/dampingRhat are1.063/1.062;
the worst coordinate is the frame10 displacement of elastic mode6, not a
material parameter. This frame has one camera observation; its full3D modal
state is not directly measured. The empirical native-state covariance has
eigenvalues0.0614–74.41, showing substantial remaining geometric distortion.
These unconverged moments are diagnostic, not estimates of the final posterior.

Six41-point straight physical interpolation paths between actual representative
samples show no uphill excess above the endpoint chord. This does not establish
unimodality or exclude narrow-volume/entropic barriers. The new chains cannot
be called converged merely because material parameters overlap or steps accept.
No withheld images or particle scores were evaluated. The predeclared full
budgets and thresholds are unchanged.

Evidence: `examples/pacnerf-bird-reflected-preview`. Reproduce the full gate with
`aggregate_pacnerf_reflected.py --draws 256`, then run
`experiments/audit_reflected_preview.py` with the same inputs, shards and saved
convergence path. An optional`--worker` selects a matching frozen checkout;
source and input hashes are checked. Parameter and slow-state trace figures
are included. The256-draw gate cannot trigger production eligibility.

The event-driven flow passes energy conservation, reversal through multiple
walls and an independent orthant solution. Logistic augmentation plus residual
Metropolis correction reproduces independent posterior quadrature. The full
alternating state/parameter sampler reproduces correlated joint posterior
moments computed by a separate Gaussian-conditioning integral. Warmup and
retained-prefix resumption are bit-identical. Neural joint/marginal fixture
tests verify the physical height map and residual gradient.

The original182-test regression suite passed before the guide/interweaving
extensions. The new mean-anchor Bird preflight height error is3.33e-16m and
residual-density identity error9.74e-13. A16-draw one-chain integration check
replays the full target to3.64e-12. This is not a convergence run.
The preceding guide trial was stopped during warmup to free local CPU for the
validated interwoven variant. Its checkpoint is preserved and no samples from
it are pooled. No convergence is claimed. Production requires at least2000
retained draws/chain, Rhat<1.01,
bulk/tailESS>=400 for every monitored state, parameter, conditional force and
normalizer, followed by independent uncertainty evaluation. The larger
comparison stays gated.
