# Posterior inference after noise calibration

The physical and image models are unchanged from v0.8.1. The new inference
integrates over trajectories and reports posterior averages, instead of treating
the maximum-density trajectory and one local Gaussian as the answer.

## Why better diffusion calibration can worsen a best-fit reconstruction

For normalized scene/path coordinates z, write the dynamics residual as

\[
r(z)=G a+\eta,\quad a\sim N(0,I),\quad \eta\mid z\sim N(0,Q(z)).
\]

The force coefficients have a Gaussian conditional because Q depends on the
observed candidate trajectory, not on a when that trajectory is held fixed.
Define C=Q+GGᵀ. The negative log posterior, up to fixed constants, is

\[
U(z)=\ell_{RGB}(z)+\frac12\|\theta\|^2+
\frac12r^TC(z)^{-1}r+\frac12\log|C(z)|-\log J_{dyn}(z).
\]

The full implementation also retains velocity-coordinate scale constants. The
centered drag Jacobian is Jdyn. Its omission, or removal of logdet C, would
change the density. These terms remain in every final acceptance calculation.

Two effects matter:

1. The old operating noise amplitude was too large. Correcting it increases
   the penalty on departures from the specified mechanics. That can expose
   remaining discrepancies in a restricted force basis, a coarse stochastic
   integrator, priors, or a separately assumed independent diffusion floor.
2. A state-dependent covariance changes both the residual penalty and the
   local density volume. In a path with 1,440 velocity coordinates, the highest
   density path need not represent the region containing most posterior mass.
   A maximum-density point also depends on the choice of coordinates.

Neither effect means that a wrong noise model is scientifically preferable.
Neither implies that correct calibration must improve every individual video.
The old/new source history alone did not isolate calibration; the new controlled
MAP ablation uses the same final renderer, kernel and mechanics for every variant.
Even an oracle analytic amplitude closely reproduces the calibrated network's
MAP errors, so its remaining amplitude approximation error is not the main cause.

For coordinate squared error, the conditional Bayes estimator is the posterior
mean. The identity

\[
E[\|X-b\|^2\mid I]=\operatorname{tr}\operatorname{Cov}(X\mid I)
+\|b-E[X\mid I]\|^2
\]

explains why a posterior mean is the appropriate comparison to a best-fit path
when scoring coordinate RMSE. This is an expectation under the assumed model,
not a guarantee on a particular synthetic realization or under model mismatch.

## What was wrong with the previous inference evidence

The optimizer passed its stationarity test. That established a local stationary
point, not posterior convergence. The Fisher Gaussian was also only a proposal:
its importance weights collapsed. Replacing it with the exact observed Hessian
did not repair the nonlinear posterior approximation. The previous exact-target
pCN sampler accepted only 2–11 independent refreshes per chain out of 180–226
attempts; one chain stayed in a different damping region. All 1,548 monitors
failed the rank/folded Rhat and ESS screen. Those samples were correctly labelled
unvalidated and are retained in the historical report.

The causal density, force elimination, derivatives and observed curvature had
already passed independent numerical checks. There was no justification for
dropping density normalizers or freezing Q to make a reconstruction look better.

## Fixed-reference split Hamiltonian proposals

Let P=LLᵀ be the positive Fisher precision at the saved original MAP m. Use

\[
z=m+L^{-T}w,\quad R(w)=-U(z)+U(m)+\tfrac12\|w\|^2.
\]

Then the target in w is proportional to exp(R(w)) times a standard Gaussian.
Introduce algorithmic momentum p~N(0,I), unrelated to physical nodal velocity.
For one proposal substep of size epsilon:

\[
p\leftarrow p+\tfrac\epsilon2\nabla\widetilde R(w),\quad
\begin{pmatrix}w\\p\end{pmatrix}\leftarrow
\begin{pmatrix}\cos\epsilon&\sin\epsilon\\-\sin\epsilon&\cos\epsilon\end{pmatrix}
\begin{pmatrix}w\\p\end{pmatrix},\quad
p\leftarrow p+\tfrac\epsilon2\nabla\widetilde R(w).
\]

The rotation solves the Gaussian part exactly. This follows the Gaussian-reference
splitting idea in [Beskos et al., *Hybrid Monte Carlo on Hilbert spaces* (2011)](https://discovery.ucl.ac.uk/id/eprint/10052120/).
It avoids introducing numerical integration error into thousands of Gaussian
directions simply because the path representation is high dimensional.

The proposal correction Rtilde keeps exact mechanics, Q(z), force marginalization
and all their gradients. Only its image NLL uses an observed second-order Taylor
polynomial in each image's 12 free-node pose coordinates. These coefficients are
computed once at the reference. No withheld image or truth state enters them.
The proposal gradient is therefore faster, while the final acceptance probability
uses the original nonlinear renderer:

\[
\alpha=\min\{1,\exp[H(w,p)-H(w',p')]\},\quad
H=U(m+L^{-T}w)-U(m)+\tfrac12\|p\|^2.
\]

Momentum reversal makes the deterministic proposal an involution. Symmetric
kicks and rotations preserve volume. Therefore this Metropolis correction
targets the full density even when the image approximation is imperfect. Fast
approximate Hamiltonian trajectories with full-target correction are discussed
in [Neal, *MCMC Using Hamiltonian Dynamics*](https://arxiv.org/abs/1206.1901).
Invalid intermediate paths are rejected, not silently repaired or resampled.
Finite mixing and numerical sensitivities still require empirical checks.

The final protocol randomizes trajectory duration, adapts step size during
discarded warmup, freezes it, and extends sampling only if convergence monitors
fail. See `POSTERIOR_CONVERGENCE_PROTOCOL.md` for the exact fixed budgets and
all-coordinate diagnostic thresholds. Raw retained proposals, energy errors,
invalid counts and complete latent chains are run artifacts. Some deliberately
inaccurate proposals can be rejected with large energy errors; smaller-step
independent replicas test numerical sensitivity rather than concealing these.

## Force uncertainty after sampling

At each retained state, compute

\[
A(z)=I+G^TQ(z)^{-1}G,\quad S_a(z)=A(z)^{-1},\quad
m_a(z)=S_a(z)G^TQ(z)^{-1}r(z).
\]

Average over sampled trajectories:

\[
E[a\mid I]\approx\overline{m_a},\qquad
\operatorname{Cov}(a\mid I)\approx\overline{S_a}
+\operatorname{Cov}_{samples}(m_a).
\]

Both covariance components are saved. Conditional means and variances, rather
than added independent Gaussian force draws, enter convergence diagnostics.
Portable force draws retain their paired trajectory and physical parameters.
The force family is still a fixed spatial profile with 15 temporal coefficients;
this does not infer arbitrary force locations or contact topology.

## What convergence and uncertainty claims mean

Passing the finite Rhat/ESS screen means that the independent chains agree at
the monitored resolution and provide a specified effective sample count.
It cannot prove that every mode was visited or that the physical model is true.
Central posterior component intervals, simultaneous conformal whole-gap sets,
and calibration of scalar diffusion innovations answer different questions.
The new benchmark comparison applies no conformal widening and uses already
inspected synthetic cases. Repeated-realization posterior calibration, model
discrepancy sensitivity and real-video validation remain separate experiments.
