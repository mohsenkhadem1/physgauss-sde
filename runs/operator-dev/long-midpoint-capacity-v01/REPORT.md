# Long-block midpoint capacity benchmark

The frozen one-step algebra benchmark completed all controls and both fine steps in 132.30 s. GPU0 is released; controller 2295270 and worker 2295352 are absent, the owned group is drained, and the execution session reaped exit 0. All source/static-input/output-capture hashes and the exact metadata cache pair verified. No retry or source change occurred.

The coarse 96-tet independent sparse CPU/GPU midpoint comparison differs by 7.12e-17 m in q and 7.11e-15 m/s in v. Full mass, M_cf reactions, original 24/80 energy/gradient/residual, exact clamps, midpoint+endpoint whole-element positive J, wrench and explicit work-defect controls pass.

| Fine mesh | Construction + solve | Final checks overhead | Whole field elapsed | Newton / CG | Original midpoint R80 |
|---|---:|---:|---:|---:|---:|
| graded 32 |27.97 s|15.72 s|43.69 s|4 /1148|2.64 pN|
| graded 40 |52.91 s|28.97 s|81.88 s|4 /1459|3.79 pN|

Constructor-only cost was not independently instrumented; construction and solve are combined. Final checks include exact 80 energy/gradient/original physical residual and whole-element midpoint/endpoint certificates. The finest whole-J lower bound is 0.916575. Peak Torch allocation 1.227 GB; watchdog peak RSS 1.989 GB. All physical/quadrature/wrench checks passed their original thresholds.

The finest external midpoint work is 4.20442 mJ and the nonlinear potential quadrature defect is -48.0842 nJ. The work-ledger identity defect is below 4e-19 J. Midpoint is NOT exactly nonlinear-energy conserving; the nonzero potential defect is retained separately from stationarity work. There is no physical damping, contact, filtering or rigid projection.

The static equilibrium at nominal time 0.49 s, one force value at 0.5 s and reconstructed endpoint 0.51 s are an ALGEBRA/COST experiment. They do not constitute physical prehistory, a C3 pulse trajectory, a forecast or a temporal/spatial convergence study. The saved 32/40 q/v point values have no convergence gate here. The source static maximum-strain goal remains failed 9.74%; this benchmark cannot rehabilitate it.

A naive 50-call use of the measured finest construction+solve cost is about44 minutes, or68 minutes including these final checks; it is only a planning multiplication, not measured trajectory throughput. Reusing a constructor and smoother successive states could change cost, and the present instrumentation cannot quantify those savings. A full dynamic protocol needs separate reviewed resource estimates and q AND v time/spatial tests before admission. No further physics is scheduled automatically.

Frozen [protocol](protocol.json), [scalar summary](compact-summary.json), [terminal proof](terminal-proof.json), and [prospective plan](PLAN.md). Raw q/v fields and full iterative linear records remain LOCAL.
