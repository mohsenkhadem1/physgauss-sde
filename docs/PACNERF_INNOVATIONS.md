# Causal innovations: algebra audit, not a new benchmark result

The production comparison still samples standardized positions and velocities
with the original frozen worker. `scripts/pacnerf_innovations.py` implements an
equivalent coordinate system for investigating difficult inference. It has
**not** replaced any production chain, demonstrated faster convergence, or
improved the reconstruction results.

Write the existing discrete transition as

$$
u_{i+1}=F_\theta u_i+b_\theta(u_i)+G_{\theta,i}a+\eta_i,
\qquad a\sim N(0,I),\quad\eta_i\mid u_i\sim N(0,Q_\theta(u_i)).
$$

Here $u_i$ contains modal positions and velocities, and $a$ is the same global
24-coordinate Gaussian load used by the joint and marginalized production
models. Diffusion depends on the preceding state, not the unknown coefficients.
Conditioned on an entire preceding path, the load posterior remains Gaussian:
$a\mid u_{0:i}\sim N(m_i,S_i)$. Initialize $P_0=S_0^{-1}=I$ and $h_0=0$.
The marginalized predictive transition is

$$
\begin{aligned}
\mu_i &= F_\theta u_i+b_\theta(u_i)+G_i m_i,\\
C_i &= Q_i+G_i S_iG_i^\top,\\
u_{i+1} &= \mu_i+L_i\xi_i,
\qquad L_iL_i^\top=C_i,\quad\xi_i\sim N(0,I).
\end{aligned}
$$

After producing or observing that next state, update the load information:

$$
\begin{aligned}
r_i&=u_{i+1}-F_\theta u_i-b_\theta(u_i),\\
P_{i+1}&=P_i+G_i^\top Q_i^{-1}G_i,\\
h_{i+1}&=h_i+G_i^\top Q_i^{-1}r_i,\\
S_{i+1}&=P_{i+1}^{-1},\qquad m_{i+1}=S_{i+1}h_{i+1}.
\end{aligned}
$$

The implementation uses Cholesky solves rather than explicit matrix inverses.
For the joint arm, retain $a$ explicitly and use $C_i=Q_i$ and mean
$F_\theta u_i+b_\theta(u_i)+G_i a$. Both constructions preserve their
respective original probability densities, including the identical Gaussian
load prior. They also preserve the equality of the integrated motion posteriors.

Because each next state depends only on previous states and its new innovation,
the trajectory map is triangular. If $s$ denotes the original physical-state
coordinate scales, its log Jacobian is

$$
\log\left|\frac{\partial z_{\rm path}}{\partial\xi}\right|
=\sum_i\log|L_i|-\sum_j\log s_j.
$$

The covariance normalizers cancel this Jacobian. In innovation coordinates the
negative log posterior therefore becomes standard Gaussian penalties for the
mechanical parameters, initial velocity, innovations and (for the joint arm)
explicit load, plus the **same nonlinear image likelihood**. The transformed
MAP can differ because density modes depend on coordinates; the probability
distribution and physical posterior expectations must agree after convergence.
One must include the Jacobian when comparing old and new densities. Dropping
normalizers in the original position/velocity density would be incorrect.

Three unit tests cover density and inverse identities for all four arms,
autodifferentiated density gradients including the Jacobian, and strict causal
dependence. Sixteen saved production states per completed case additionally
pass the round-trip and density identities: across cases 0, 2, 4 and 6,
maximum standardized path error is 1.45e-12 and maximum negative-log-density
discrepancy is 1.23e-11. Joint cases 5 and 7 also pass the same sixteen-state
production audit (maximum density discrepancy 5.01e-12). The audit reads no future geometry. Its record is
`examples/pacnerf-sparse/innovation-coordinate-diagnostic.json`.

These checks establish algebraic equivalence at the tested points, **not**
adequate sampling or a superior sampler. Sequential transformation still needs
a forward pass through time; it removes a nested force-fitting loop rather
than eliminating temporal computation. Any sampling study using these
coordinates needs separately recorded inference settings, physical-state and
force diagnostics, cached-target replay, and comparisons against the unchanged
production attempts. No production data, weights, thresholds or physics have
been retuned for this audit.

## Bounded observed-data-only feasibility pilot

`pacnerf-innovation-pilot.yml` separately tests all four Torus four-view arms
using the same frozen prepared inputs and physical source. It uses a standard
Gaussian prior reference, prior-dispersed initial chains, 100 warmup transitions
and 100 retained draws per chain, with a 25-minute compute limit per arm. The
original split-HMC kernel, target acceptance .9 and randomized integration
times from 1 to 4 are retained. The seed is 250001 plus case index. Warmup and
retained states are checkpointed by the existing tested wrapper.

No future geometry is read and no reconstruction or coverage score is computed.
This deliberately short pilot tests whether the coordinate change with a plain
prior metric is feasible; it cannot establish convergence or model superiority.
It also monitors original physical coordinates, force conditional means and
normalizers, and checks the original/Jacobian-adjusted density and cached target.
Poor results would concern this particular coordinate/metric/initialization
combination, not rule out other valid uses of innovations. Pilot chains must
never be substituted into the original production comparison.
