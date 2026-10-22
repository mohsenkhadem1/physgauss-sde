# Static supported block: gravity and broad traction

All 18 predeclared static solves completed in 6.69 s on two CPU threads, with exact constraints, positive tetrahedral Jacobians, and accepted original equilibrium residuals. No case yet passes every frozen qualification check. These are independent compressible Neo-Hookean calculations; they are not Isaac Sim dynamics, dropping/contact, a learned model, or a posterior.

The block is 240 × 80 × 80 mm, density 625 kg/m³ (mass 0.96 kg), Poisson ratio 0.35. Meshes contain 96, 768, and 6,144 tetrahedra. Every case starts from rest. Supports constrain all three displacement components on physical faces or two bottom quarter-length patches. Gravity is 9.81 m/s² with E = 300 kPa. The separate axial case uses E = 5 kPa and a uniform 1 N reference-face traction. No rigid handles, point forces, artificial fixture springs, time-step inertia, stiffness repair, or output filtering are used.

| Finest case | Peak strain | Volume-weighted 99th percentile | Volume strain RMS | Rigid-removed volume coordinate RMS | Largest 768→6,144 query error | Frozen outcome |
|---|---:|---:|---:|---:|---:|---|
| Left clamp, broad axial traction | 3.794% | 3.288% | 2.020% | 1.291 mm | 365.3 µm | Excitation passes; spatial accuracy fails |
| Left clamp, gravity | 4.333% | 3.352% | 0.722% | 0.450 mm | 3,144.8 µm | Bulk excitation and spatial accuracy fail |
| Right clamp, gravity | 4.204% | 3.282% | 0.719% | 0.450 mm | 3,204.7 µm | Bulk excitation and spatial accuracy fail |
| Both end clamps, gravity | 0.817% | 0.661% | 0.190% | 0.132 mm | 94.0 µm | Restrictive control; spatial accuracy fails |
| Bottom face clamp, gravity | 0.194% | 0.148% | 0.050% | 0.0117 mm | 3.85 µm | Spatial accuracy passes; force balance narrowly fails |
| Two bottom patches, gravity | 0.623% | 0.456% | 0.094% | 0.0454 mm | 47.6 µm | Restrictive control; spatial accuracy fails |

The gravity cantilevers reach 15.74 and 15.68 mm maximum displacement. That visible motion does not satisfy the prospectively defined 1 mm *nonrigid volume coordinate RMS after a best rigid fit*. The original 2–8% peak-strain and 1 mm bulk goal remains unchanged. Both-end, bottom, and separated-patch controls were declared restrictive controls and are not required to reach that excitation target.

Fixed probes include 36 material-interior points, 66 face-interior points, 12 edge midpoints, and all eight corners. Every group's maximum Euclidean position discrepancy must be at most 20 µm, corresponding to 0.1 of the stated 200 µm reference sensor scale. All corners and edges remain included. These are spatial material-point/surface-position diagnostics, not rendered depth observations or a certificate over every continuum point.

For the finest bottom-clamp case, the original free-force residual is 5.20 × 10⁻¹⁰ N, below its 9.42 × 10⁻¹⁰ N acceptance tolerance. Its summed reaction-plus-load defect is 1.1535 × 10⁻⁸ N, slightly above the separate frozen 10⁻⁸ N global balance gate. The failure is retained; no threshold relaxation or tighter rerun was used. Other finest cases pass their reaction force/moment checks. The minimum finest Jacobian over all cases is 0.9683; all exact support defects are zero.

Sharp fully clamped/free-face intersections can concentrate strain and stress. Peak strain, the volume-weighted 99th percentile of each tetrahedron's largest absolute principal stretch strain, and volume-weighted strain RMS are reported separately. The auxiliary volume measures do not replace the peak-strain gate. Positive Jacobians and equilibrium acceptance are independent of the 8% excitation upper target. Large spatial errors occur at interior and face-interior probes too, so omitting corners would not qualify the gravity cases.

Three implementation tests passed in 1.78 s: nested physical support membership and exact known mass/load resultants; energy-gradient, exact-constraint and reaction checks; and mesh-independent affine volume RMS plus passive frame transformation. Static spatial qualification, excitation quality, and nonlinear solver acceptance are deliberately separate. No inverse or network stage is authorized by this result, and no dynamic/freefall/contact claim is made.

Sources, the frozen pre-run protocol, and all retained result records are indexed in [the publication inventory](../../operator-dev/static-supported-block-v01/publication-inventory.json). See the [compact summary](compact-summary.json), [predeclared protocol](../../operator-dev/static-supported-block-v01/protocol.json), and [terminal receipt](../../operator-dev/static-supported-block-v01/terminal.json). Raw nodal fields and per-solve ledgers remain local.
