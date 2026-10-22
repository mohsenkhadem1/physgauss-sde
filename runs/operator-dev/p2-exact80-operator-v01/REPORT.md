# Exact80 operator assessment at the completed Q3 field

The bounded operator-only GPU2 calculation completed in **33.88 s**, with no nonlinear or linear solves and no new trajectory. Peak CUDA allocation was **1.95 GB**. The same immutable graded48 field, material, load and clamp were supplied to Q3 and exact80 operators.

Q3 free residual was 9.86e-13 N; the exact80 residual is **4.83e-8 N**. The supplied Q3 field therefore does not meet strict exact80 equilibrium acceptance. Energy changes by 2.05e-12 J. Gradient difference is 6.14e-9 N maximum and 1.80e-6 relative L2. The prescribed public smooth HVP differs by 2.58e-11 N maximum and 6.86e-8 relative L2; node-block relative L2 difference is 4.82e-7. These are operator diagnostics, not a displacement error certificate.

All 80 quadrature points were streamed with the original FP64 polynomial stable-NH law. No clipping, material change, force change, response fitting or convergence threshold relaxation occurred. No exact80 displacement/spatial/time/inference qualification is claimed.

Session 49199 was reaped with exit zero. Controller 2226806 and worker 2226892 were absent; GPU2 was released. Raw operator differences remain local.
