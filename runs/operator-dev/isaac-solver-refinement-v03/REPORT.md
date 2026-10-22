The bounded native refinement completed all four applications in 186.67 seconds. The highest-rate iteration check passes, but temporal accuracy remains unqualified. No further Isaac jobs were launched.

| Comparison, same physics and material queries | Maximum position difference | 20 µm gate |
|---|---:|---|
| 480→960 Hz, 128 iterations; prior context | 5.210 mm | FAIL |
| 960→1920 Hz, 128 iterations | 3.448 mm | FAIL |
| 1920→3840 Hz, 128 iterations | 0.993 mm | FAIL |
| 128→255 iterations at 3840 Hz | 9.739 µm | PASS |

All 325 material nodes and 19 common frames were compared without registration or interpolation. The 1920→3840 maximum occurs at 0.5 s on node300, a free-end corner. The separate mechanics-context JSON gives the exact world XYZ error components, locations and times. The highest-rate comparison controls iteration sensitivity at that rate; it does not prove smaller-rate solves converged, provide a physical residual, or bound continuum error.

The response is deforming rather than moving as a rigid block. At3840 Hz/255 iterations, the mass-weighted best-rigid-fit residual reaches5.215 mm vector RMS and ends at2.958 mm. Principal stretches span approximately0.795–1.234, native-rest-relative J spans0.850–1.154, and best-fit body rotation reaches22.2°. Fixed nodes remain exact and all native geometry, material, gravity and clock guards pass.

The COM down-peak period changes from0.34896 to0.34635 to0.34609 s across960/1920/3840 Hz, about2.89 Hz at the finest rate. A rough Euler–Bernoulli small-deflection cantilever estimate using the authored parameters gives4.92 Hz and15.9 mm static tip deflection. It is context only: this block has L/h=3, large finite bending, and an unidentified native constitutive law. The discrepancy is not evidence that permits fitting a different E or density.

Nominal-mass kinetic energy and gravity-potential trends are saved separately. At480 Hz, first/second-cycle sampled kinetic peaks are0.0829/0.0759 J; at3840 Hz/255 they are0.0840/0.0853 J. This is consistent with decreasing numerical attenuation, but is not a native energy-balance proof. The conditional log-J Neo-Hookean energy proxy varies strongly (roughly0–0.80 J total), so it cannot be used as a conservation diagnostic without identifying native stored energy and masses. The proxy is explicitly labeled throughout; no native conservation violation is claimed.

The temporal pair ratios give unstable apparent orders, approximately0.60 then1.80. They do not establish an asymptotic error law. Reducing the last0.993 mm discrepancy to20 µm under a naive first-order assumption would require about50 times the rate, with substantial runtime and no guarantee. The current four-app cost was186.67 s; escalating rates for twenty minutes is not a justified accuracy fix. The native dynamic/contact route remains unqualified while the independently aligned static reference is investigated separately.

One initialization limitation remains explicit. Installed SimulationManager source lines759–760 call `simulate(dt,0)` during initialization before tensor views are constructed. The worker then restores and verifies rest positions and zero velocities before applying the fixed-node targets. Hidden warm-start state is not observed. This is a possible confound, rather than a demonstrated cause; no initialization or physical setting changed in this refinement.

The [protocol](protocol.json), [compact results](result-summary.json), [terminal receipt](terminal-receipt.json), and [posthoc mechanics context](mechanics-context.json) bind this run. All controller/worker PIDs are absent, the session was reaped with exit0, and all13 source hashes remain unchanged. Execution success is separate from the failed temporal gate. Raw trajectories and full logs remain local. The original [four-cell ablation and resource failure](../isaac-timestep-forensic-v01/REPORT.md) remain preserved.
