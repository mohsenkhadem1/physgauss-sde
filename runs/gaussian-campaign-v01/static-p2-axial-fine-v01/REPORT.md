# A qualified static axial block benchmark

The nonlinear rectangular block now passes the declared static fixed-probe, excitation, equilibrium, quadrature, and positive-Jacobian checks for **one exact axial loading case**. The final two solves completed in 532.03 s on two CPU threads, with a monitored peak process RSS of 2.55 GiB, below the frozen 8 GiB guard. Both quadrature orders start from rest. No additional mesh, force, material, or support search was performed.

The case is a 240 × 80 × 80 mm block with E = 5 kPa, Poisson ratio 0.35, density 625 kg/m³, an exact all-component left-face clamp, and 1 N uniform reference-right-face traction. It has no rigid handles, Gaussian spring approximation, gravity, contact, dynamic time step, learned model, or fitted physical parameter. The 12,000 affine-rest tetrahedra use ten-node quadratic displacement bases, 18,081 shared global nodes, and consistent continuum force integrals.

| Check | Result | Frozen criterion |
|---|---:|---:|
| 6,144→12,000 maximum fixed-query difference | 3.522028 µm | ≤20 µm |
| Conditional first-order fine-error indicator | 14.088114 µm | ≤20 µm |
| 27→64-point quadrature maximum query difference | 0.000101 µm | ≤20 µm |
| Original free-force residual, both orders | 8.688 × 10⁻¹² N | ≤10⁻¹¹ N |
| Global force balance defect | 3.67 × 10⁻¹² N | ≤10⁻⁸ N |
| Actual-position moment balance defect | 7.24 × 10⁻¹² Nm | ≤10⁻⁸ Nm |
| Exact support displacement defect | 0 | 0 |
| Minimum sampled Jacobian | 0.989608 | >0.1 |
| Sufficient global elementwise Jacobian lower bound | >0.758 | >0 |
| Sampled peak absolute principal stretch strain | 6.4873% | 2–8% |
| Rigid-removed material-volume coordinate RMS | 1.294959 mm | ≥1 mm |

The maximum displacement is 7.60858 mm. The volume-weighted 99th percentile of local maximum absolute principal stretch strain is 3.3272%; volume-weighted principal-strain RMS is 2.0321%. These auxiliary values remain separate from the peak excitation criterion. Peak strain is sampled at all quadrature points and ten reference element nodes; it is not a certified continuous maximum near the sharp clamp edges.

All 122 fixed physical probes remain included: 36 volume-interior, 66 surface-interior, 12 edge-midpoint, and eight corner points. The 20 µm displacement criterion is one tenth of the declared 200 µm sensor scale. It is a finite material/surface-position diagnostic, not rendered depth or a bound at every continuum point.

The factor four in the fine-error indicator comes from h-coarse/h-fine = 20/16 = 1.25 and assumed minimum convergence order p = 1. Under an asymptotic power-law regime with p ≥ 1, the fine-error multiplier is at most 1/(1.25−1) = 4. The resulting 14.09 µm is a **conditional engineering convergence indicator**, not a rigorous continuum-error bound or proof that asymptotic order has been reached. The independent quadrature comparison rules out a material quadrature contribution at that scale for this case.

The positive-Jacobian certificate is separate: P2 displacement gradients are affine over each affine-rest tetrahedron. The spectral norm is convex, so the largest vertex ||F−I||₂ bounds it throughout the element. Here that maximum is approximately 0.08822, below one; hence every element has positive determinant with J ≥ (1−max||F−I||₂)³ > 0.758. No inversion repair, stiffness modification, or residual relaxation was used.

The result preserves the failed history. The initial P1 768→6,144 axial query discrepancy was 365.27 µm. The first P2 comparison gave 22.353 µm and failed 20 µm. A separately declared intermediate 2,592-tetrahedron solve reduced the adjacent difference to 7.9971 µm, but its three-times first-order indicator was 23.9912 µm and still failed. The final one-level continuation was authorized prospectively with unchanged physics and gates. A resource-only 900 s amendment was frozen before this agent read the intermediate result; the old 600 s source and its budget-limited outcome remain intact. Both final quadrature orders independently converge from rest.

The implementation tests cover P2 partition and gradient identities, affine compatible deformation, consistent load resultants/moments, finite rigid rotations, energy-gradient and tangent finite differences, invalid-J rejection, an independent homogeneous uniaxial oracle, and query interpolation. Two additional gate tests ensure an adjacent error below 20 µm cannot bypass the stricter first-order estimate and that Euclidean vector error is used. The frozen initial three-test receipt is 3 passed in 2.01 s; the two continuation guard tests passed in 1.62 s.

This is a successful static axial development benchmark with informative strain. **Gravity remains spatially unresolved**: its initial P2 768→6,144 query discrepancy is 163.344 µm, and its rigid-removed bulk RMS misses the 1 mm target. Earlier Gaussian-fixture failures, native Isaac temporal failures, dropping/contact fidelity, dynamic forecasting, unknown-boundary inference, and uncertainty calibration are not qualified by this result. No inverse or neural training was launched.

See the [final compact summary](compact-summary.json), [frozen final protocol](../../operator-dev/static-p2-axial-fine-v01/protocol.json), [terminal receipt](../../operator-dev/static-p2-axial-fine-v01/terminal.json), [intermediate retained failure](../static-p2-axial-continuation-v01/REPORT.md), and [initial P2 report](../static-p2-block-v01/REPORT.md). Raw nodal fields and detailed histories remain local. A static deformation ramp in a visualization would be interpolation, not a simulated time trajectory.
