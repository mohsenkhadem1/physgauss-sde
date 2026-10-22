# Fixed-mesh surface quadrature: negative result

The Q16 run completed and its original conservation/J/mass guards passed, but the declared Q8→Q16 field and velocity gates failed. The same Warp backend, 96 T10 tetrahedra, exact full consistent mass, stable-NH material, finite compliant reference-area plane law (κ=10⁹ N/m³), analytic freefall initializer, adaptive tolerances and maximum step were retained. Only the positive Duffy surface rule changed from 8×8 to 16×16; the qualified Q8 capture was imported without another simulation.

Maximum common-output differences were **121.710 µm at nodes**, **110.192 µm at the 81 fixed queries**, and **0.308995 m/s in nodal velocity**, versus gates of 20 µm, 20 µm and 0.0001 m/s. The energy difference was 1.526 nJ and passed. Q16 accepted-state energy drift was 1.499 nJ (470.88 nJ limit), momentum consistency error 7.77×10⁻¹⁵ N s, minimum guarded J 0.925262, mass-solve residual 2.00×10⁻¹³ N, and maximum analytic quadratic-face penetration 166.710 µm.

The new run used 29,566 accepted steps / 552,551 RHS evaluations and 737.01 s of integration (743.23 s controller wall time). Both owned processes were reaped and GPU1 released; all recorded resource guards passed. Completed case metadata was cached externally by exact SHA after termination, without changing the frozen driver.

This is failed surface-quadrature convergence, not drop-data qualification. Conservation does not establish field accuracy, Q16 itself has not been established as exact, and this finite compliant floor is not a hard-contact limit. No Q32 or other follow-up was launched. Raw trajectories, checkpoints, logs and metadata cache remain local.
