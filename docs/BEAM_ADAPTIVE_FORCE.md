# Adaptive temporal force priors and observation-conditioned actions

This extends [active beam design](BEAM_ACTIVE_DESIGN.md). A **policy** is the
rule choosing the next measurement. In this experiment it is a numerical,
one-step Bayesian decision rule, not a trained network or reinforcement
learning controller. Observations determine the current posterior. Expected
reduction in hidden-deformation prediction uncertainty determines the action.

## Why faster forces broke uncertainty

The old model fixed the squared-exponential force correlation length at 2s,
with force standard deviation 0.06N and the static condition F(0)=0. The stress
generator used 0.8s. The smoother model assigns very little variance to rapid
changes, especially in differences between nearby times. It can therefore
prefer a wrong smooth interpolation even after observing more data.

For the actual eight-time covariance matrices, the condition numbers are
151,030 (2s) and 9.49 (0.8s). The expected squared standardized energy of a
fast-force draw under the smooth prior is 1,140.8 per degree of freedom; it
would be 1 under the assumed model. This is evidence of prior mismatch, not
proof of numerical failure. Independent dense Gaussian conditioning and
replay tests verify that inference computes its stated model correctly.

Even when forces at t=1,4,8s are observed exactly, the conditional force SD at
t=5s is 0.0221N under the smooth prior and 0.0533N under the faster prior.
At a fixed boundary and timescale, Gaussian conditional covariance depends
on the measurement design/noise, not on the observed values. Large residuals
can change the estimated mean without automatically increasing uncertainty.
More observations can therefore narrow a misspecified posterior around a
biased prediction. Boundary ambiguity is an additional source of uncertainty,
but cannot substitute for missing fast temporal modes.

The new comparison reports prequential observation PIT and negative log
predictive density: every selected measurement is scored BEFORE its update.
It also includes an oracle-timescale model given the true correlation length,
while still inferring supports and forces. That privileged diagnostic helps
separate temporal misspecification from mechanics and boundary ambiguity.

## Adapt the timescale while retaining uncertainty

Let beta index the 24 boundary configurations and ell the force timescale.
The new prior uses ell in {0.35,0.6,0.8,1.2,2,3}s with equal prior mass. It is
independent of the initial boundary prior. The kernel remains

\[
K_\ell(t,t')=0.06^2\left[
e^{-(t-t')^2/(2\ell^2)}-
e^{-t^2/(2\ell^2)}e^{-t'^2/(2\ell^2)}\right].
\]

The second term conditions on F(0)=0. Force amplitude, Gaussian kernel family,
camera pose, measurement noise, material and reference geometry remain given.
The timescale is constant within a single eight-time episode; this does not
yet model within-episode changes of smoothness or discontinuous contact.

The full posterior is

\[
p(\beta,\ell,F\mid D)=w_{\beta\ell}
\mathcal N(F;m_{\beta\ell},C_{\beta\ell}).
\]

There are 144 components. Each action has the existing linear observation
model y=h_beta F+d_beta+epsilon. Its force-integrated predictive likelihood is

\[
p(y\mid a,\beta,\ell,D)=\mathcal N(y;h_\beta m_{\beta\ell}+d_\beta,
h_\beta C_{\beta\ell}h_\beta^T+\sigma^2).
\]

This likelihood updates the JOINT boundary/timescale weight. Conditional
Gaussian force means and covariances update by the same exact formulas as
before. The timescale posterior is the sum over boundary weights, and the
boundary posterior is the sum over timescale weights. No true timescale,
unselected measurement or hidden deformation is passed to either estimator.

Predictions keep both within-component and between-component variance:

\[
\operatorname{Var}(u\mid D)=\sum_{\beta,\ell}w_{\beta\ell}
\left[V_{\beta\ell}+(\mu_{\beta\ell}-\bar\mu)^2\right].
\]

This is Bayesian adaptation of a force-prior hyperparameter, not a plug-in
maximum-likelihood timescale or ad-hoc widening of output intervals. Forces
are still marginalized analytically. It does not train a neural SDE or a
flow-matching network.

## What guides the active probing policy?

For all candidate actions, Q is the SAME original hidden deformation at all
eight times and four hidden reference coordinates. Define prediction risk as

\[
R(D)=|Q|^{-1}\sum_{q\in Q}\operatorname{Var}(u(q)\mid D).
\]

The policy chooses

\[
a^*=\arg\min_{a\in A_{\rm remaining}}
\mathbb E_{y\sim p(y\mid a,D)}R(D\cup\{a,y\}).
\]

Thus **observations condition the model; expected uncertainty reduction guides
the choice**. A point being uncertain is not sufficient: the proposed
measurement must be expected to resolve uncertainty that matters for hidden
motion. A probe that tells us about an irrelevant support detail may score
poorly. New views distinguish temporal force hypotheses directly. A known
probe tests support response; it can also alter timescale probabilities through
their posterior dependence on support and original force ambiguity.

Candidate predictions are hypothetical and used only for ranking. After
selection, the system receives the selected observation and performs the
actual update. It never trains against hidden evaluation truth or scores
actions by their realized future error. Costs remain equal per scalar
measurement; robot execution costs and safety constraints are outside scope.

## Numerical integration

The wider timescale mixture can contain narrow and broad predictive peaks
together. The previous low-order component Gauss-Hermite rule was inadequate
for some such scores. The new scorer uses composite Gauss-Legendre integration
over every component's +/-9SD envelope, with partitions near component means
and scales. It refines until probability mass, the second-moment tower identity
and risk-score change satisfy 2e-5 tolerance. It raises an error on nonconvergence.
The first four cases of each population, at every active step and every
available action, are audited at 1e-7 tolerance. Unit tests also compare to
independent dense trapezoidal integration. No samples or MCMC are needed.

## Comparison and validation

The [configuration](../configs/beam_adaptive.json) fixes six actions and the
same 13-action pool as the previous experiment. Fixed observes all five
initially missing times chronologically at x=0.7m, then uses the distal known
probe. Random uses a separate seeded permutation. Active plans using its own
current posterior. The two inference models (old fixed-2s and adaptive) are
crossed with all three action policies. **Fixed prior** and **fixed policy**
are different concepts.

The comparisons include:

- 400 slow-force cases replayed from the prior study, with identical data.
- 200 fast-force cases replayed from the failing prior study.
- 400 fresh cases with timescale drawn from the new hyperprior: a test of
  marginal calibration under the complete hierarchical model.
- 200 fresh cases with true timescale 0.5s, absent from the model's grid:
  a limited off-grid robustness test.

Replays diagnose the changed model; they are not new independent validation.
The timescale grid includes the known historical test values. The fresh
hierarchical and off-grid populations are declared before the full run and
are not used to tune the prior. A small implementation pilot checks numerical
correctness and runtime, not statistical performance.

Report every stage's hidden RMSE, predictive SD, 50/80/90/95% scalar coverage,
Wilson intervals, PIT and proper interval score. Paired bootstrap differences
resample entire cases. Conditional fixed-timescale coverage is not equivalent
to marginal hierarchical calibration. Gates use the previous three-binomial-SE
and PIT convention; multiple-stage checks are descriptive, not a familywise
test or simultaneous trajectory confidence statement.

Known material, replayable quasistatic loads, calibrated probes and tracked
centres remain assumptions. Action adaptivity is compared with chronological
fixed ordering, not an optimized nonadaptive schedule. Calibrated uncertainty
under this model does not imply robustness to jumps, unknown force amplitude,
wrong support families, contact, camera error or nonlinear mechanics.

## Reproduction and archive

```sh
OPENBLAS_NUM_THREADS=1 python -m unittest discover -s tests -p 'test_beam*.py' -v
OPENBLAS_NUM_THREADS=1 python experiments/run_beam_adaptive.py --config configs/beam_adaptive.json --output examples/beam-adaptive --workers 4
OPENBLAS_NUM_THREADS=1 python experiments/verify_beam_adaptive.py --output examples/beam-adaptive
```

On PowerShell set `$env:OPENBLAS_NUM_THREADS='1'` first. Workers run independent
populations on CPU, never GPU. The read-only
[GitHub workflow](../.github/workflows/beam-adaptive.yml) uses two standard CPU
workers and uploads the reproduction as an artifact.

Compressed JSON archives include input truth/observations, all potential
action outcomes, selected measurements, candidate scores and per-case metrics.
Every posterior is reconstructible from the inputs and action history; the
verifier replays them, checks mixture interval CDFs, covariance PSD, length
probabilities, action budgets and identical-final-data invariance. Source
hashes normalize LF/CRLF; compressed data hashes are byte-exact.

See [results](../examples/beam-adaptive/results.json),
[report](../examples/beam-adaptive/report.html), and
[verification](../examples/beam-adaptive/verification.json).
