All four declared pure-gravity trajectories completed and passed the original mechanics checks. Joint numerical time refinement passed on each fixed mesh, with genuinely different accepted partitions. The conditional spatial displacement and velocity criteria both failed. The overall qualification flag remains false. No new simulation, smoothing, startup exclusion or gate change followed this result.

The case was fixed E=100 kPa, nu=.35, density 625 kg/m³, g=9.81 m/s², full clamps at both ends, fresh unloaded rest geometry/zero velocity, no surface force, damping or contact. Meshes were 12,000/20,736 straight T10 tetrahedra. The physical operator used the degree-six-exact Keast24 bulk rule, original 80+10 sampled determinant guards and analytic full consistent mass. Strict GPU PCG changed only the mass-equation solver; its diagonal remained a preconditioner rather than a physical mass lump.

| Fixed-mesh joint time comparison | Maximum all-node position difference | Maximum all-node velocity difference | Result |
| --- | --- | --- | --- |
| 12000 tets | 0.0000576 µm | 0.001206 mm/s | Both pass |
| 20736 tets | 0.0001381 µm | 0.003878 mm/s | Both pass |

The position/velocity time limits were 20 µm / 0.1 mm/s. Accepted steps changed 1,000 → 2,001 and 1,003 → 2,001, respectively. Both tolerance and step cap were refined together; no isolated convergence-order claim follows.

| Spatial comparison at all 25 clocks / 63 fixed material queries | Raw maximum position difference | Conditional position indicator | Raw maximum velocity difference | Conditional velocity indicator |
| --- | --- | --- | --- | --- |
| Coarse time controls | 11.0937 µm | 55.4686 µm — FAIL | 9.57334 mm/s | 47.8667 mm/s — FAIL |
| Refined time controls | 11.0937 µm | 55.4686 µm — FAIL | 9.57348 mm/s | 47.8674 mm/s — FAIL |

Each indicator divides the raw maximum by the actual coarse/fine maximum-element-edge ratio minus one (factor approximately 5). This is a conditional engineering estimate under assumed spatial order at least one, not a rigorous continuum bound. The unchanged spatial limits were 20 µm / 0.1 mm/s. Raw displacement disagreement alone being below 20 µm does not rescue the failed declared indicator. The largest velocity disagreement occurred at 70.833 ms at material point [.12,.04,.04] m; displacement disagreement peaked at 100 ms at [.12,-.04,-.04] m. Full unfiltered traces remain available. These comparisons locate the discrepancy but do not uniquely identify its physical/numerical cause.

For clarity, refined-space S plus maximum observed temporal T is descriptively 55.468712 µm for position and 47.871257 mm/s for velocity. These sums are not a combined error bound, and the protocol never established a combined 20 µm bound. Position and velocity, time and space criteria remain separate.

All original physical mass-equation residuals were at most 9.997e-13 N (guard 1e-9 N); mass PCG also met its stricter original-residual target. Maximum mechanical-energy drift was 4.279e-13 J (guard 1e-8 J). Minimum sampled determinant was .93912 and the minimum whole-element Bernstein lower bound at saved clocks was .93917. Exact supports were retained. This does not certify continuous-time determinant bounds or arbitrary material/support/load cases.

The case runtimes were 459.66/894.03/695.28/1349.09 s, within the predeclared 1200/2400/1800/3600 s caps. Total owned-controller elapsed time was 3413.18 s (56.9 minutes). All completed case JSONs were independently copied by content SHA before comparison. Controller, four workers and cache monitor are terminal/reaped; GPU0 is released. All outcomes are retained and raw states/ledgers/checkpoints stay local. No continuum dynamic dataset, broad E/BC, contact, image inference or active-policy admission follows.
