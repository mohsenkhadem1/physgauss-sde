The numerical mass-solver comparison passed all five declared RHS controls on 96, 12000 and 20736 tetrahedra. No ODE, Newton or response solve was run. Each control used the exact analytic full consistent mass matrix; the original diagonal was used only as a preconditioner. The physical matrix, kinetic energy, loads and fixed reactions were unchanged.

Independent SciPy residuals were at most 6.83e-13 N, below the frozen 1e-12 N limit. The optional relative condition only tightened this limit to min(1e-12 N, 1e-11 times the RHS norm). Maximum relative acceleration error versus LU was 2.63e-11, maximum component error 2.22e-9 m/s², and fixed M_cf reaction error 1.28e-13 N. The separately labeled manufactured kinetic/work checks also passed. Near-equilibrium RHS controls converged to approximately 2e-23 N residual, rather than being rounded to zero. All zero-RHS checks returned zero exactly.

| Mesh | Repeated GPU PCG mass solve | Repeated CPU LU mass solve | Fine PCG iterations |
| --- | --- | --- | --- |
| 96 tets | 7.6–7.7 ms | 0.043–0.139 ms | 32 |
| 12000 tets | 25.1–28.9 ms | 43.1–43.6 ms | 46 at rest / 52 at static state |
| 20736 tets | 32.4–36.0 ms | 97.4–102.5 ms | 46 at rest / 52 at static state |

Timings include host/device transfers. GPU PCG initialization took 0.040/0.009/0.016 s; the independent LU initializations took 0.002/2.866/10.130 s. The GPU option is slower for the tiny mesh. Its fine-mesh mass solve is about 1.6–2.9 times faster, but this is not a trajectory speed claim. With force, telemetry and mass-action costs, four fine trajectories still have a roughly 3000-second minimum forecast before adaptive refinement and checkpoint/output overhead.

The worker completed in 21.06 s, the owned controller in 24.00 s; observed GPU/RSS maxima were 0.825/1.674 GB. All states and outcomes were retained. Both PIDs were reaped and GPU0 was released. Sources and all JSON evidence are hash-bound by the external terminal receipt. No temporal, spatial, continuum, dataset or uncertainty qualification follows from this numerical control.
