# Root128 contact: same-mesh time control passes

One fresh loose trajectory (rtol1e-9, displacement atol1e-12 m, velocity atol1e-10 m/s) completed in 90.18 s and was compared to the exact hash-bound completed root128 tight trajectory (rtol1e-10/1e-13 m/1e-11 m/s). The physical 96-tet model, root128 compliant plane, Keast24 bulk, full consistent T10 mass, all141 Jacobian guards, analytic freefall initialization, maximum time step1/30000 s and nine output clocks are unchanged.

The maximum differences are **7.727 nm position** and **32.384 µm/s velocity**, below 20 µm and100 µm/s. Output-energy difference15.48 nJ is below470.88 nJ. Actual accepted partitions differ. The loose run took2900 accepted steps/43475 RHS evaluations; original mass residual≤1.93e-13 N, accepted energy error≤15.85 nJ, momentum defect≤2.47e-15 Ns and minimum RHS J=.925348. All four same-geometry CPU/GPU controls passed before integration.

Session51544 was reaped with exit0. Controller2273012 and worker2273018 are absent; GPU0 was verified idle and released. All source/evidence hashes remain unchanged. The terminal case JSON was cached externally by its exact bytes.

This closes the **root128-only same96mesh time gap**. It does not reuse the insufficient old root64-time plus64→128-surface velocity-error sum as a pass. No spatial, new-mesh surface-order, hard-floor, contact-population or dataset admission follows. No retry, physical retuning, tolerance relaxation or old-source edit was made.
