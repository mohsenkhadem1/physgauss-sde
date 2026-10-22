# Higher-rate compatible P2 contact: time convergence still unresolved

All four 96-tetrahedron P2 cases completed with unchanged original residual/J criteria and fixed E300 kPa, ν=.35, ρ625 kg/m³, g=−9.81z, κ=10⁹ N/m³, Q8 surface quadrature, friction/damping zero. GPU1-only runtime was254.3 s. Full accepted-state checkpoints were saved every32steps and at each completed case.

A separately declared exact analytic freefall initializer starts at t=1/12 s with 15.9375 mm positive clearance and zero contact energy. Current and previous displacements/velocities are the exact gravity polynomials for each chosen step. Its independent full-M next-step check differs from analytic position by1.39e-17 m. This is a new numerical startup, not a relabeling of the older BE-start captures.

The common120Hz/81-fixed-material-query differences are:

|Rates (Hz)|Maximum difference|Contraction|20 µm criterion|
|---|---:|---:|---|
|1920→3840|1.3643 mm|—|FAIL|
|3840→7680|1.4369 mm|1.0533|FAIL|
|7680→15360|0.85967 mm|0.5983|FAIL|

No rate met both the accuracy and contraction conditions; the conditional 768-tetrahedron solve was therefore not launched. No drop-data or rigid-floor qualification is claimed. A small force residual establishes algebraic accuracy, not time resolution. The stiff undamped contact remains under-resolved. Materials, floor stiffness and damping were not altered to hide the discrepancy.

The compact result includes energy and full-consistent-mass impulse bookkeeping. For this fully BDF2 analytic initialization, the telescoped momentum is (3P_N−P_N−1−3P_0+P_minus1)/2. Energy changes are numerical-method diagnostics, not an unconditional BDF2 conservation theorem. Raw trajectories/checkpoints and per-step ledgers remain local; both owned processes terminated.
