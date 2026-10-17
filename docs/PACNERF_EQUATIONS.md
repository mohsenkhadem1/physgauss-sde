# Equations for the frozen PAC-NeRF transfer model

This describes `src/physgauss/pacnerf_model.py` at production source
`b2d081a7a55ecf1027921f5d841c043e23713a9d`. It is a reduced graph model,
not the authors' MPM or the earlier cube's tetrahedral mechanics.
The [protocol](PACNERF_PROTOCOL.md) defines observations and diagnostics.

## Motion and rendering

There are 32 mechanical handles and 12 orthonormal displacement modes, the
lowest modes of the initial central-spring stiffness matrix. For handle n,

\[
x_n(t)=X_n+\Phi_n q(t),\qquad \dot x_n(t)=\Phi_n v(t),\quad q,v\in\mathbb R^{12}.
\]

Six modes describe infinitesimal rigid motions; six describe deformation.
This linear subspace does not implement finite-rotation-invariant elasticity.
The 128 rendering centers use fixed RBF interpolation weights W:
\(\mu_g(t)=\mu_g(0)+(W\Phi)_gq(t)\). Their isotropic radius, opacity and
initially fitted colors stay fixed. Continuous depth extinction/emission
composites them over supplied background images.

Let \(u=(q,v)\) and standardized material parameters be
\(\theta=(\theta_k,\theta_c)\sim N(0,I_2)\). The linear generator is

\[
A_\theta=\begin{bmatrix}0&I\\-e^{0.7\theta_k}\Lambda&-0.8e^{0.7\theta_c}I\end{bmatrix},
\qquad F_\theta(h)=\exp(hA_\theta).
\]

Lambda is the fixed graph spectrum. Its scale is not a continuum Young's
modulus estimate. Initial q is zero and initial modal velocity has independent
N(0,0.5²) components.

On each h=1/24 s interval, the known modal acceleration b(q_i) includes gravity
and the preceding-pose ground reaction. At each handle this is
\([0,-9.8+6\,\operatorname{softplus}(-x_{n,y}/0.01),0]\), projected by Phi.
Holding contact over a whole interval is a material integration approximation,
as the separate post-hoc contact audit shows.

## Structured diffusion and correlated unknown loads

The modal velocity-diffusion covariance at the preceding state is

\[
C_i=H_iH_i^T+0.15^2I.
\]

Each column of H is the modal projection of an equal/opposite central edge
impulse, multiplied by \(0.20/\sqrt{E/32}\) and an amplitude ratio. E is the
edge count. The constant control uses ratio 1. The neural arm uses
\(h_\psi(\epsilon_e,s_e)/\sigma_\text{old}\), where strain and radial edge
speed feed the frozen 33-parameter network:

\[
h_\psi(f)=10^{-3}\operatorname{softplus}
 \left(W_2\tanh\left(W_1\frac{\sqrt{f^2+10^{-6}}}{[0.10,0.50]}+b_1\right)+b_2\right)+10^{-4}.
\]

The old constant MLE is about 0.00180739. Both arms retain geometry-dependent
edge directions and modal projection; "constant" means constant **edge
amplitude**, not a constant isotropic state covariance. The 0.15 modal floor is
additional full-rank discrepancy and is not itself a balanced internal edge
force. All these amplitudes are transfer assumptions, not PAC-calibrated truth.

Unknown applied acceleration at handle n is

\[
f_n(t)=2\sum_{k=1}^4\sum_{p=1}^2\psi_k(t)w_p(n)a_{k,:,p},
\qquad a\sim N(0,I_{24}).
\]

The four temporal Gaussian RBFs have length 0.14 s and are normalized so
\(\sum_k\psi_k(t)^2=1\). The two fixed spatial profiles are uniform and
lower-rest-region, each with handle mean one. This finite-rank prior induces
correlations across space and time; it cannot describe arbitrary localized
contacts. Its output has acceleration units, not calibrated Newton forces.

With \(J=[0;I]\), define

\[
D_\theta=\int_0^h F_\theta(s)J\,ds,\quad
Q_i=\int_0^h F_\theta(s)JC_iJ^TF_\theta(s)^Tds.
\]

The implementation uses exact analytic F and four-point quadrature for these
integrals. Let P_i map a to modal applied acceleration at time i. Then

\[
r_i=u_{i+1}-F_\theta(h)u_i-D_\theta b(q_i),\qquad
r_i\mid a\sim N(G_i a,Q_i),\quad G_i=D_\theta P_i.
\]

Diffusion depends on the **previous** state, never the next innovation or a.
The state-to-residual Jacobian is triangular with unit diagonal in physical
coordinates. Fixed latent-coordinate scales contribute only a constant.

## Exact elimination of the load coefficients

Stack all 13 residual blocks into r, use block-diagonal Q, and stack G. Set

\[
L=I+G^TQ^{-1}G,\qquad m=L^{-1}G^TQ^{-1}r.
\]

The joint dynamics density keeps a and has negative log density

\[
\tfrac12\left[(r-Ga)^TQ^{-1}(r-Ga)+a^Ta+\log|Q|+(n+24)\log(2\pi)\right].
\]

Integrating a gives

\[
\tfrac12\left[(r-Gm)^TQ^{-1}(r-Gm)+m^Tm+\log|Q|+\log|L|+n\log(2\pi)\right].
\]

Here n=13×24. Every normalizer is retained. Joint and marginalized arms have
the same integrated motion posterior; they differ in sampled dimension
(350 versus 326), parameterization and computational cost. Their path MAPs
can differ because maximizing and integrating a density are different operations.

Given a motion sample, \(a\mid u,\theta,\mathrm{RGB}\sim N(m,L^{-1})\).
Reported coefficient uncertainty uses total covariance
\(E[L^{-1}]+\operatorname{Cov}(m)\) in both arms; joint raw sample moments
are saved separately. These are model-conditioned distributions, not validated
physical force uncertainty.

## Image posterior and sampling

Independent RGB residuals have declared sigma 0.08. The target combines their
squared error with the dynamics density and the same parameter/initial-velocity
priors. All eleven initial images fit appearance; sparse later images fit motion.
Future geometry enters only scoring after sampling.

Despite Gaussian load elimination, the motion posterior is **not Gaussian**:
rendering, contact, and state-dependent diffusion are nonlinear. Four chains
therefore use Gaussian-reference split HMC with full nonlinear image/physics
gradients and exact Metropolis correction. A mode and observed Hessian provide
proposal coordinates only. The protocol's stationarity and finite-chain gates,
paired joint/marginal mean screen, and independent raw replay precede ranking.
None establishes physical correctness or repeated-dataset uncertainty calibration.
