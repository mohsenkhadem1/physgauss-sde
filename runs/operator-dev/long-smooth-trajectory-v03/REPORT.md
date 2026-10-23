# Actual C3 trajectory pilot: completed numerical negative

All4 planned actual trajectories completed all original physical/numerical step checks:352 accepted steps and36 fixed-clock captures, supervised wall6758.61s (1.877h). The predeclared POSITION temporal contraction gate FAILED. This pilot has no temporal/spatial trajectory qualification, dataset admission or forecasting claim. No thresholds changed, no extra trajectories or retries ran.

The experiment starts from each mesh's own hash-bound gravity equilibrium at real time0 with v0=0. Domain .48×.08×.08m, stable polynomial NH/E300kPa/nu.35/rho625, exact full end clamps and full signed consistent P2 mass. Known force is gravity plus worldDOWN2N times256t^4(1−t)^4 on0<t<1; T1s. No damping, contact, filtering or projection.

| Mesh / timestep | Accepted steps / clocks | Worker elapsed | Median accepted step | Physical checks |
|---|---:|---:|---:|---|
| fine40_h32 | 32 / 9 | 1261.35s | 39.52s | PASS |
| fine40_h64 | 64 / 9 | 3601.85s | 61.86s | PASS |
| fine40_h128 | 128 / 9 | 6755.62s | 50.58s | PASS |
| coarse32_h128 | 128 / 9 | 2418.87s | 19.20s | PASS |

Sum of elapsed worker time3.899h; this is elapsed time while leases were owned, not measured GPU compute time. Fine constructors were measured separately at16.69–17.04s (coarse8.80s). Timing includes original checks and durable accepted-state writes; no universal reuse speedup is inferred.

| Common63 queries at9 clocks | Position | Velocity |
|---|---:|---:|
| fine40 timestep delta32→64 | 3.985332µm | 269.743899µm/s |
| fine40 timestep delta64→128 | 2.625618µm | 80.485310µm/s |
| Observed temporal order | 0.602043: FAIL floor1 | 1.744793: PASS floor1 |
| Conditional actual-h spatial p>=1 indicator | 16.600909µm | 9.283871µm/s |
| Descriptive temporal+spatial sum | 19.226528µm | 89.769182µm/s |
| Frozen combined threshold |20µm|100µm/s|

Position's descriptive sum is below20µm, but its required contraction premise fails, so it is NOT a qualified error bound or a position PASS. The allnode position order also fails (0.602016); allnode velocity order is1.744937. Independent direct NumPy recomputation reproduces every stored delta. Position pair maxima occur at t1s for both pairs. Velocity pair maxima occur at t.625/.75s. Two spatial levels only support the explicitly conditional p>=1 indicator; no spatial order was measured.

Every accepted midpoint/endpoint retained exact clamps, original residual/CG/24vs80 checks, whole-element positive-J guards, full-Mcf reactions and force/moment/work/linear-angular impulse ledgers. Across352 steps: original R80<=8.95182e-11N, 24/80 gradient<=2.73445e-12N, energy<=5.55112e-17J, whole-J lower>=0.907803026. Maximum force/moment defects2.30348e-10N/5.39142e-11Nm. Maximum work-identity defect3.66045e-19J. Cumulative linear/angular impulse residual maxima4.25387e-12Ns/1.27501e-12Nms. All retain their unchanged original limits.

| Job | Signed potential midpoint defect sum | Sum absolute defects | Stationarity work sum |
|---|---:|---:|---:|
| fine40_h32 | -3.76746098e-13J | 1.39477492e-09J | 3.47033532e-17J |
| fine40_h64 | 4.50994842e-14J | 3.408603e-10J | -3.63110355e-18J |
| fine40_h128 | 7.07427526e-15J | 8.48401906e-11J | -3.20948483e-16J |
| coarse32_h128 | 6.96145711e-15J | 8.43986832e-11J | -8.44139752e-16J |

Signed cancellation is not exact nonlinear-energy conservation. The actual external midpoint work and both defect sums are retained separately in the scalar summary. Original CSR is unchanged: reported machine-scale skew and angular cross-velocity terms remain inside the physical impulse ledger; no matrix symmetrization or mass lumping occurred.

For the finest run, the maximum recorded63-query excursion from initial gravity equilibrium is1.201670mm and recorded maximum query speed4.035321mm/s, over9 clocks. These are query maxima. The earlier static local-strain goal remains FAILED and local peaks/stress/real physics are not certified.

All4 workers exited0 and their owned groups are empty; detached controller PID is absent (its exit code was not independently observed). GPU0..2 are released, GPU3 untouched. All52 frozen pins, all36 clock hashes/finite states/clamps, final+previous checkpoints, complete accepted ledgers and exact external metadata caches verified after completion. Raw clocks, checkpoints, fields and media stay LOCAL. Old v02 zero-step technical guard negative and all frozen source/results remain unchanged.

Frozen protocol, original aggregate, independent terminal audit, compact summary and prior technical-negative lineage are linked by SHA256 in compact-summary.json. Further refinement needs a new reviewed protocol; none is scheduled here.
