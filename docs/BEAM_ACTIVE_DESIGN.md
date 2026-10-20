# Active measurements for hidden beam deformation

The scientific objective is to choose the next observation or controlled load
that improves prediction of the hidden deformation. Exact recovery of support
labels is not the objective. This experiment extends the
[two-Gaussian beam benchmark](GAUSSIAN_BOUNDARY_FORMULATION.md), retaining known
material, reference geometry, camera poses and correspondences.

## Posterior and actions

Let beta denote a support/load-kernel hypothesis and F the eight original force
amplitudes. The state is an exact finite Gaussian mixture:

\[
p(\beta,F\mid D)=w_\beta\mathcal N(F;m_\beta,C_\beta).
\]

An action a obtains one scalar vertical-track observation:

\[
y_a=h_{a,\beta}F+d_{a,\beta}+\epsilon,
\qquad \epsilon\sim\mathcal N(0,\sigma^2),\quad\sigma=0.25\text{ pixels}.
\]

For another view, h contains the known camera projection and the beam's
unit-load response, with d=0. A calibrated probe uses the unloaded/reset beam,
the same support, a known 0.04 N load and a known reference position. Then h=0
and d is its projected deterministic displacement. Probe noise is the total
track noise, not a difference of two noisy measurements. Load calibration,
repeatability and reset are idealized assumptions.

Each action's predictive likelihood is available after integrating force:

\[
\mu_{a,\beta}=h_{a,\beta}m_\beta+d_{a,\beta},\qquad
s_{a,\beta}=h_{a,\beta}C_\beta h_{a,\beta}^{T}+\sigma^2.
\]

After observing y, update the mixture weights and conditional force state:

\[
w'_\beta\propto w_\beta\mathcal N(y;\mu_{a,\beta},s_{a,\beta}),\quad
g_\beta=C_\beta h_{a,\beta}^T/s_{a,\beta},
\]
\[
m'_\beta=m_\beta+g_\beta(y-\mu_{a,\beta}),\qquad
C'_\beta=C_\beta-g_\beta s_{a,\beta}g_\beta^T.
\]

A probe changes support probabilities while leaving original force means and
covariances conditional on a support unchanged. Reweighting those components
can still change the marginal original force and deformation predictions.
This is the force-marginalization approach: no force MAP substitution or force
sampling is required, including when evaluating candidate actions.

## Prediction-driven choice

For every action the target is the SAME original load sequence at all eight
times and hidden positions x={0.05,0.20,0.85,1.0} m. The current squared-error
Bayes risk is the average predictive variance:

\[
R(D)=\frac1{|Q|}\sum_{q\in Q}\operatorname{Var}[u(q)\mid D].
\]

Choose

\[
a^*=\arg\min_{a\in A_{\rm remaining}}
\int R(D\cup\{(a,y)\})p(y\mid a,D)\,dy.
\]

Fifteen-point Gauss-Hermite quadrature under EACH mixture component integrates
the possible observations. The first eight cases of each population, at every
active step, are checked against order 61. Tests additionally compare against
independent dense numerical integration and the analytic known-boundary case.
Expected gains are relative to the current risk, not to a different target
chosen by the action. Planning never reads true boundaries, original loads or
unselected action measurements. A hypothetical observation only scores an
action; inference updates only with the actually acquired synthetic measurement.

The policy is one-step greedy. It is not a globally optimal six-action plan.
The risk is justified under the assumed model; posterior variance can be
misleading under model mismatch. All actions cost one scalar measurement.
Physical execution time and robot safety/control costs are not modeled.

## Paired comparison

The [frozen configuration](../configs/beam_active.json) specifies 400 matched
cases, 200 shorter-force-prior stress cases and six actions per policy. All
policies start with the same observations at times 1,4,8 seconds and visible
positions 0.4,0.55,0.7 m. The 13 candidate actions are:

- A new vertical track at times 2,3,5,6,7 seconds and x=0.4 or 0.7 m (10 actions).
- Apply a known load centered at x=0.3,0.55,0.8 m and observe x=0.7 m (3 actions).

Fixed observes the five missing times in chronological order at x=0.7 m, then
applies the distal probe. Random samples uniformly without replacement. Active
minimizes the expected risk above. Action order matters for interim predictions;
with the same final measurement set, Bayesian posteriors must agree. This is
explicitly checked, so an ordering benefit is not confused with a better final
posterior from identical data.

An eight-case implementation pilot originally gave fixed the middle probe.
After active consistently selected the distal probe, fixed was strengthened to
use that probe before the full run. The final configuration is used unchanged
in both populations. We do not present the pilot as independent validation.

These are repeated/replayable quasistatic loading experiments. Querying a
missing earlier time requires replay or an existing acquisition pool; this is
not a causal online-camera policy that travels back in time. Loads in the
original sequence are unknown but assumed repeatable across acquisitions.

Ground truth uses 80 finite elements; inference uses 40. All per-case truths,
initial observations, potential action observations, selected actions, action
scores and tip posterior components are archived. Common action noise is shared
across policies, and only selected measurements enter inference. Random-policy
RNG is separate from the data generator. Case zero is the predeclared example.

## Metrics and uncertainty

After EVERY action, evaluate hidden original motion over all eight times and
dense x<0.3 or x>0.8. Report mean per-case RMSE, the initially missing-time RMSE,
integrated posterior standard deviation, and paired case-bootstrap differences
(2,000 resamples). Pointwise 50/80/90/95% intervals for the never-observed tip
at time 5s use the full mixture CDF, with Wilson coverage intervals and PIT KS.
The 90% interval score penalizes both excessive width and uncovered truth.

The matched scalar gate uses the previous beam convention: 90% coverage within
three binomial standard errors and PIT KS <=1.63/sqrt(N), at each stage/policy.
It is a set of descriptive marginal checks, not a familywise test, simultaneous
path coverage, or a guarantee conditional on every support. Benefits are not
enforced as software assertions. The stress arm changes true force correlation
length from 2s to 0.8s but leaves the estimator unchanged; failed calibration
must remain visible rather than being tuned away.

## Reproduce

```sh
OPENBLAS_NUM_THREADS=1 python -m unittest discover -s tests -p 'test_beam*.py' -v
OPENBLAS_NUM_THREADS=1 python experiments/run_beam_active.py --config configs/beam_active.json --output examples/beam-active
OPENBLAS_NUM_THREADS=1 python experiments/verify_beam_active.py --output examples/beam-active
```

On PowerShell set `$env:OPENBLAS_NUM_THREADS='1'` first. Python 3.12, NumPy 2.0
and Matplotlib 3.9 reproduce the study on CPU; no SciPy or GPU is needed.
The [free CPU workflow](../.github/workflows/beam-active.yml) uses read-only
repository access and uploads its full reproduction as an Actions artifact.
It never pushes or overwrites repository results automatically.

See the [report](../examples/beam-active/report.html),
[aggregate results](../examples/beam-active/results.json), and
[independent archive replay](../examples/beam-active/verification.json).
This is a small controlled reference, not a trained encoder, flow-matching
model, nonlinear mechanics solver or real-image robotics demonstration.
