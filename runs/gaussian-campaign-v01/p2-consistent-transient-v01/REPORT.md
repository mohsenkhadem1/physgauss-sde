# Consistent-mass P2 transient CPU reference

Nine of the ten prospective trajectories completed. The tenth (96-tetrahedron smooth ramp at 960 Hz) was stopped by the frozen 300 s wall cap after 53 accepted steps. No automatic retry, redraw or continuation occurred. The incomplete case, accepted ledger and last state are preserved; the overall matrix remains incomplete.

These use the unchanged stable Neo-Hookean law, E=5000 Pa, nu=0.35, rho=625 kg/m3, a 0.24 x 0.08 x 0.08 m block, exact full left clamp, gravity/damping zero, no contact, and normalized 1 N full right-face axial traction. The full analytic consistent T10 mass matrix is assembled as sparse CSR with all node couplings. Negative vertex row sums are legitimate consistent integration weights; the mass matrix is SPD and is never row-sum lumped.

The numerical method is backward Euler startup followed by BDF2 on (displacement, physical velocity). Both the abrupt step and 0.05 s sine-squared ramp-to-hold are fixed known controls. Force is evaluated at the implicit right endpoint, with exact command times/arrays saved. All runs start from rest and integrate 0.1 s. Full u/v/step increments and mass CSR are saved, together with 25 common 240 Hz physical times and 122 material-probe positions/velocities. These are fixed-discretization CPU references for same-rate GPU checks; they do not provide spatial qualification.

| Mesh / control | Rates (Hz) | Full-trajectory adjacent maximum query differences (um) | Status |
|---|---|---|---|
| 6 / step | 240, 480, 960 | 213.540704, 95.255904 | FAIL 20 um |
| 6 / smooth_ramp | 240, 480, 960 | 99.721830, 29.570220 | FAIL 20 um |
| 96 / step | 480, 960 | 244.026804 | FAIL 20 um |
| 96 / smooth_ramp | 480 | unavailable | incomplete |

Across the nine complete runs, maximum objective residual is 8.36e-11 N and independently reconstructed physical momentum residual is 8.36e-11 N. Every accepted endpoint satisfies J>0.1 and exact zero clamp displacement/velocity. No force tolerance was relaxed.

Four tests passed in 1.86 s: independent positive order-4 quadrature of the full mass matrix and SPD/negative-row-sum checks; direct-displacement energy/gradient/tangent parity with the frozen physical assembler plus finite differences; independent coupled first-order oscillator BDF2 oracle; and actual consistent-P2 force/constraint checks.

The stage took 300.17 s and the watchdog observed 586,985,472 bytes peak RSS, below its 8 GiB guard. The worker and watchdog are absent, and the exec session was reaped with exit 1 for the preserved wall-limit failure. All frozen source hashes and all nine complete capture hashes match.

Evidence: [prospective protocol](../../operator-dev/p2-consistent-transient-v01/protocol.json), [summary](summary.json), [available full-trajectory comparisons](available-comparisons.json), [terminal receipt](../../operator-dev/p2-consistent-transient-v01/terminal.json). Raw NPZ fields and full ledgers remain local.
