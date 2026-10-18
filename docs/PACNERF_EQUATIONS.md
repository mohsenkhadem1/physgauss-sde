# Equations for the frozen PAC-NeRF transfer model

This describes the corrected numerical model in
`src/physgauss/pacnerf_model.py` and exact-innovation worker `219665b`.
It is a reduced graph model, not the authors' MPM or the earlier cube's
mechanics. The [corrected protocol](PACNERF_EXACT_PROTOCOL.md) defines the
observations, source pins and diagnostics. The historical four-point
quadrature results are superseded.

## Motion and rendering

There are 32 mechanical handles and 12 orthonormal displacement modes, the
lowest modes of the initial central-spring stiffness matrix. For handle n,

$$
x_n(t)=X_n+\Phi_n q(t),\qquad \dot x_n(t)=\Phi_n v(t),\quad q,v\in\mathbb R^{12}.
$$

Six modes describe infinitesimal rigid motions; six describe deformation.
This linear subspace does not implement finite-rotation-invariant elasticity.
The 128 rendering centers use fixed RBF interpolation weights W:
$\mu_g(t)=\mu_g(0)+(W\Phi)_gq(t)$. Their isotropic radius, opacity and
initially fitted colors stay fixed. Continuous depth extinction/emission
composites them over supplied background images.

Let $u=(q,v)$ and standardized material parameters be
$\theta=(\theta_k,\theta_c)\sim N(0,I_2)$. The linear generator is

$$
A_\theta=\begin{bmatrix}0&I\\-e^{0.7\theta_k}\Lambda&-0.8e^{0.7\theta_c}I\end{bmatrix},
\qquad F_\theta(h)=\exp(hA_\theta).
$$

Lambda is the fixed graph spectrum. Its scale is not a continuum Young's
modulus estimate. Initial q is zero and initial modal velocity has independent
N(0,0.5²) components.

On each h=1/24 s interval, the known modal acceleration b(q_i) includes gravity
and the preceding-pose ground reaction. At each handle this is
$[0,-9.8+6\,\operatorname{softplus}(-x_{n,y}/0.01),0]$, projected by Phi.
Holding contact over a whole interval is a material integration approximation,
as the separate post-hoc contact audit shows.

## Structured diffusion and correlated unknown loads

The modal velocity-diffusion covariance at the preceding state is

$$
C_i=H_iH_i^T+0.15^2I.
$$

Each column of H is the modal projection of an equal/opposite central edge
impulse, multiplied by $0.20/\sqrt{E/32}$ and an amplitude ratio. E is the
edge count. The constant control uses ratio 1. The neural arm uses
$h_\psi(\epsilon_e,s_e)/\sigma_\text{old}$, where strain and radial edge
speed feed the frozen 33-parameter network:

$$
h_\psi(f)=10^{-3}\operatorname{softplus}
 \left(W_2\tanh\left(W_1\frac{\sqrt{f^2+10^{-6}}}{[0.10,0.50]}+b_1\right)+b_2\right)+10^{-4}.
$$

The old constant MLE is about 0.00180739. Both arms retain geometry-dependent
edge directions and modal projection; "constant" means constant **edge
amplitude**, not a constant isotropic state covariance. The 0.15 modal floor is
additional full-rank discrepancy and is not itself a balanced internal edge
force. All these amplitudes are transfer assumptions, not PAC-calibrated truth.

Unknown applied acceleration at handle n is

$$
f_n(t)=2\sum_{k=1}^4\sum_{p=1}^2\psi_k(t)w_p(n)a_{k,:,p},
\qquad a\sim N(0,I_{24}).
$$

The four temporal Gaussian RBFs have length 0.14 s and are normalized so
$\sum_k\psi_k(t)^2=1$. The two fixed spatial profiles are uniform and
lower-rest-region, each with handle mean one. This finite-rank prior induces
correlations across space and time; it cannot describe arbitrary localized
contacts. Its output has acceleration units, not calibrated Newton forces.

With $J=[0;I]$, define

$$
D_\theta=\int_0^h F_\theta(s)J\,ds,\quad
Q_i=\int_0^h F_\theta(s)JC_iJ^TF_\theta(s)^Tds.
$$

The implementation uses exact analytic F and balanced augmented matrix
exponentials for these integrals: a 3x3 affine block for each modal input and
a stable 5x5 Kronecker-sum block for every mode pair's covariance integral.
All off-diagonal entries of C_i are retained. Fixed four-point quadrature was
inaccurate at high stiffness and is superseded. These exact linear integrals
do not remove the frame-held contact approximation. Let P_i map a to modal
applied acceleration at time i. Then

$$
r_i=u_{i+1}-F_\theta(h)u_i-D_\theta b(q_i),\qquad
r_i\mid a\sim N(G_i a,Q_i),\quad G_i=D_\theta P_i.
$$

Diffusion depends on the **previous** state, never the next innovation or a.
The state-to-residual Jacobian is triangular with unit diagonal in physical
coordinates. Fixed latent-coordinate scales contribute only a constant.

## Exact elimination of the load coefficients

Stack all 13 residual blocks into r, use block-diagonal Q, and stack G. Set

$$
L=I+G^TQ^{-1}G,\qquad m=L^{-1}G^TQ^{-1}r.
$$

The joint dynamics density keeps a and has negative log density

$$
\tfrac12\left[(r-Ga)^TQ^{-1}(r-Ga)+a^Ta+\log|Q|+(n+24)\log(2\pi)\right].
$$

Integrating a gives

$$
\tfrac12\left[(r-Gm)^TQ^{-1}(r-Gm)+m^Tm+\log|Q|+\log|L|+n\log(2\pi)\right].
$$

Here n=13×24. Every normalizer is retained. Joint and marginalized arms have
the same integrated motion posterior; they differ in sampled dimension
(350 versus 326), parameterization and computational cost. Their path MAPs
can differ because maximizing and integrating a density are different operations.

Given a motion sample, $a\mid u,\theta,\mathrm{RGB}\sim N(m,L^{-1})$.
Reported coefficient uncertainty uses total covariance
$E[L^{-1}]+\operatorname{Cov}(m)$ in both arms; joint raw sample moments
are saved separately. These are model-conditioned distributions, not validated
physical force uncertainty.

## Image posterior and sampling

Independent RGB residuals have declared sigma 0.08. The target combines their
squared error with the dynamics density and the same parameter/initial-velocity
priors. All eleven initial images fit appearance; sparse later images fit motion.
Future geometry enters only scoring after sampling.

Despite Gaussian load elimination, the motion posterior is **not Gaussian**:
rendering, contact and state-dependent diffusion are nonlinear.

The sampler uses an exact triangular change of coordinates. For explicit
loads, independent standard innovations give

$$
u_{i+1}=F_\theta u_i+D_\theta b(q_i)+G_i a+\operatorname{chol}(Q_i)\xi_i.
$$

For marginalized loads, maintain the conditional Gaussian distribution
$a\mid u_{0:i},\theta\sim N(m_i,S_i)$. Then

$$
\begin{aligned}
\bar u_{i+1}&=F_\theta u_i+D_\theta b(q_i)+G_i m_i,\\
V_i&=Q_i+G_iS_iG_i^T,\\
u_{i+1}&=\bar u_{i+1}+\operatorname{chol}(V_i)\xi_i,\\
S_{i+1}^{-1}&=S_i^{-1}+G_i^TQ_i^{-1}G_i,\\
S_{i+1}^{-1}m_{i+1}&=S_i^{-1}m_i+G_i^TQ_i^{-1}r_i.
\end{aligned}
$$

Start with $m_0=0,S_0=I$. Covariance and contact use the preceding state,
so each transform is causal and its Jacobian is triangular. The log absolute
Jacobian is the sum of Cholesky log diagonals, minus fixed state-coordinate
log scales. It cancels the transition Gaussian normalizers exactly. With
$\eta=(\theta,v_0/0.5,\xi[,a])$, the transformed negative log density is

$$
U(\eta)=\tfrac12\|\eta\|^2+
\tfrac12\|[R(u(\eta))-y]/0.08\|^2+\text{constant}.
$$

The mapping $u(\eta)$ remains fully nonlinear. Unit tests and saved-state
replays compare this expression to physical density minus its log Jacobian.
The physical path posterior and force priors are unchanged.

Fit a reference center c on observed images only. Its positive-definite
Gauss-Newton precision is $P=I+J_{\rm RGB}^TJ_{\rm RGB}$, assembled using
sensitivities of observed poses and **all observed pixels**. It is fixed before
sampling. If $BB^T=P^{-1}$ and $\eta=c+Bw$, split HMC evaluates the exact
log-density correction

$$
\log\frac{\pi_w(w)}{\phi(w)}=-U(c+Bw)+U(c)+\tfrac12\|w\|^2.
$$

The Gaussian part admits exact rotations; nonlinear kicks and Metropolis
correction retain the full target. Neither Gauss-Newton nor the Gaussian
reference substitutes a linearized image likelihood. A coordinate-dependent
reference mode is not a physical-path MAP and is not a posterior mean.

Four chains monitor both physical and innovation variables, force conditional
means, log target and covariance normalizers. The protocol's finite-chain gates,
paired joint/marginal mean screen and independent raw replay precede ranking.
Using observations after a missing interval is smoothing, not future-video
forecasting. None of these checks establishes physical correctness or
repeated-dataset uncertainty calibration.
