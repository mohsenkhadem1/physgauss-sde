# Long supported-block static prerequisite

The new .48 × .08 × .08 m, E300 kPa, rho625 kg/m³, nu=.35 block completed both body-gravity cases in260.75 s. All four fresh fine fields passed unchanged Keast24/exact80 original residual, gradient/energy, force/moment balance, exact clamp and whole-element positive-J checks. New96-tet independent CPU/GPU equilibrium controls agree within6.36e-17 m. All owned processes and groups are absent; GPU0/1 released without a watchdog workaround.

| Finest graded40 result | Both end clamps | Separated bottom patches |
|---|---:|---:|
| Conditional fixed63 query indicator |14.429 µm|12.224 µm|
| Bulk nonrigid coordinate RMS |1.37428 mm|.291858 mm|
| Maximum displacement |7.00738 mm|1.33917 mm|
| Maximum local principal strain |9.7367%|23.3413%|
| Volume-weighted99th percentile strain |2.2153%|.8215%|
| Whole-element Jacobian lower bound |.919637|.819672|

Both fixed-query spatial checks pass20 µm. The conditional p>=1 indicator is an engineering refinement estimate, not a rigorous continuum bound. Both original2–8% maximum-strain goals fail, and the bottom-patch case also fails the1 mm nonrigid goal. Large fixed-edge peaks remain explicit and do not establish converged stress or physical material validity.

This is a STATIC prerequisite only. No actuator pulse, dynamic trajectory, observation, inference or training was run. The new midpoint implementation passed independent manufactured q/v order, Cayley/linear-energy, nonlinear energy-gradient/work-defect, exact-constraint/reaction and endpoint-J controls. Nonlinear midpoint is not exactly energy-conserving: potential quadrature defect is separate from original stationarity work. A representative fine midpoint capacity/parity stage must be separately reviewed before any trajectory schedule; fine solves can still be costly.

Frozen protocol: [protocol-v02.json](protocol-v02.json). Scalar evidence: [compact-summary.json](compact-summary.json), [terminal-proof.json](terminal-proof.json). Source and design: [PLAN.md](PLAN.md). All raw fields, iterative checkpoints and full case metadata remain local. Unexecuted v01 preparation metadata and its source snapshot are retained. No sealed C38/C39 source was changed.
