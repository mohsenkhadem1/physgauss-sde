# Finite-plane drop: time and spatial checks remain separate

All four fresh trajectories completed, including their mandatory four geometry-specific CPU/adaptive-plane/GPU/full-mass controls. The original force law, E300 kPa/nu.35/rho625, full consistent T10 mass, Keast24 bulk, root128 surface operator, all141 Jacobian guards, analytic freefall initial condition and physical output clocks are unchanged. No retries, material retuning or tolerance relaxation occurred. The two phase sessions were reaped with exit zero; all controller/worker PIDs are absent, all four terminal case JSONs are cached by exact bytes, and GPUs1/2 were verified idle at release.

## Same-mesh time comparisons

| Mesh | Maximum node position difference | Maximum node velocity difference | Original time gate |
|---|---:|---:|---|
| 96 tets, separately completed control | 7.727 nm | 32.384 µm/s | PASS |
| 324 tets | 5.025 nm | 31.151 µm/s | PASS |
| 768 tets | 17.326 nm | 192.421 µm/s | **Velocity FAIL** |

The original limits remain20 µm position and100 µm/s velocity. Actual accepted partitions differ. The768 displacement pass does not repair its velocity failure. Phase wall times were288.86 s and644.37 s; all accepted/common energy, momentum and full-mass checks passed. Jacobian checks are the declared141 samples per element; no whole-element/whole-time positivity certificate is asserted here.

## Fixed root128 operator, common-material spatial comparison

The prospective CPU comparison loaded identical nine clocks and81 material points. Positions and physical velocities were interpolated with the authored compatible P2 basis. The old96 capture lacks query velocities, so its saved physical nodal velocity was interpolated through its exact original P2 mesh. No all-node arrays were compared across different meshes.

| Pair | Raw maximum position difference | Conditional position indicator | Raw maximum velocity difference | Conditional velocity indicator |
|---|---:|---:|---:|---:|
| 96→324 | 2.50050 mm | 5.00100 mm | 3.42379 m/s | 6.84757 m/s |
| 324→768 | 1.56223 mm | 4.68668 mm | 2.37502 m/s | 7.12506 m/s |

The factors are2 and3, respectively, under the explicitly assumed asymptotic order at least one. They are engineering indicators, not rigorous continuum error bounds. Raw position and velocity differences contract, but the finest indicators fail the unchanged20 µm/100 µm/s criteria by large margins. These failures are separate from the768 temporal velocity failure. Numerical time refinement alone cannot establish spatial accuracy.

**The contact dataset remains closed.** The comparison uses one fixed root128 surface operator. Manufactured64/128/adaptive E/g agreement is not trajectory surface-order proof at the new deformed states. The model is a finite reference-area compliant plane, not a rigid floor, frictional contact or CCD certificate. No population/inverse/image/learned-policy qualification follows.

The new CPU adaptive reference controls energy and gradient only at the original E1e-12 J/g1e-10 N/rtol1e-10 tolerances. Original strict E/g/H status2 failures are retained explicitly. No Hessian qualification or status2 waiver is claimed. A separately authorised768 numerical refinement may diagnose its time error; it must preserve these initial results and does not change the spatial verdict.
