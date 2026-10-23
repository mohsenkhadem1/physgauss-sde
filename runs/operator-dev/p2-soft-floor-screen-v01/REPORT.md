# Softer compliant-floor screen: completed, spatially unqualified

This is a new physical boundary experiment: floor stiffness κ = 10⁶ N/m³, 1000 times softer than the previous 10⁹ N/m³ experiment. The old failed results remain unchanged. Material, gravity, gap, freefall initialization, full consistent P2 mass, Keast24 bulk law, root128 surface operator, 141 Jacobian guards, absence of friction/damping, and the 0.15 s endpoint were preserved. The [protocol](protocol.json) was frozen before all three fresh trajectories.

All 12 independent CPU manufactured E/g controls passed before execution, and every GPU worker passed its own four geometry controls before integrating. CPU64/root128 agreement here is a manufactured-state check, not a trajectory surface-order qualification. The prior strict full-E/g/H integration failure remains preserved; the new independent reference certifies E/g only and does not waive unsuccessful integration status.

| Base tetrahedra | Total / integration time | Accepted steps | Maximum energy defect | Minimum accepted J | Maximum compliant penetration |
|---|---:|---:|---:|---:|---:|
| 96 | 72.17 / 36.76 s | 2006 | 0.145 nJ | 0.98876 | 7.195 mm |
| 324 | 87.53 / 51.69 s | 2009 | 0.378 nJ | 0.98589 | 7.180 mm |
| 768 | 127.68 / 89.53 s | 2010 | 0.0198 nJ | 0.98267 | 7.184 mm |

All mechanics and conservation gates passed. The maximum original full-mass solve residual was 6.33×10⁻¹⁵ N. Penetration is the response of a finite compliant floor; this is not a hard-floor accuracy claim. The global three-GPU controller finished in 131.50 s, with no retry, timeout, or omitted case.

The original common81 displacement/velocity gates still fail:

| Mesh pair | Raw maximum position difference | Raw maximum velocity difference | Conditional p ≥ 1 position indicator | Conditional velocity indicator |
|---|---:|---:|---:|---:|
| 96 → 324 | 205.86 µm | 0.15393 m/s | 411.72 µm | 0.30785 m/s |
| 324 → 768 | 152.12 µm | 0.13311 m/s | 456.36 µm | 0.39934 m/s |

Both raw errors contract. Neither finest indicator meets 20 µm / 0.0001 m/s. These are conditional engineering indicators under an assumed order of at least one, not rigorous continuum error bounds. No temporal pair was run, and root128 remains a fixed operator pending trajectory surface-order checks. No contact dataset is admitted.

The prospectively specified [read-only field diagnosis](field-diagnosis.json) uses positive exact-volume quadrature for rigid fits and the original consistent mass for momentum. At the finest pair, velocity discrepancy has coordinate RMS 13.853 mm/s; its rigid component is only 0.127 mm/s (0.916% of the total L2 norm), while the nonrigid maximum is 0.133104 m/s. The bulk nonrigid position RMS peaks at 0.436 / 0.435 / 0.434 mm across the three meshes; corresponding velocity RMS peaks are 0.0800 / 0.0789 / 0.0784 m/s. Momentum versus saved net impulse defects are below 5.23×10⁻¹⁵ Ns. The unresolved discrepancy remains predominantly elastic rather than a global rigid bounce mismatch. This decomposition does not uniquely identify individual modal frequencies or causes.

All owned processes were reaped; the [terminal receipt](terminal-verification.json) checks source/evidence pins, capture hashes, external case-JSON caches, and PID absence. Peak observed GPU residency was 0.300 GiB, RSS 1.46 GiB, and minimum global free space 104.8 GiB. [Raw trajectories](../../gaussian-campaign-v01/p2-soft-floor-screen-v01/summary.json) remain local. No additional physics, time refinement, damping, projection, or material changes were executed.
