# Independent finite-strain static bar benchmark

The sparse compressible Neo-Hookean energy, gradient, and tangent pass an independent analytical uniaxial benchmark at **2.12% and 6.58% axial extension**, on 48- and 384-tetrahedron meshes. This checks the bulk elastic operator and new static boundary assembly. It does not qualify the separate localized Gaussian-boundary simulations or resolve their spatial-convergence failures.

The prism is 0.24 × 0.12 × 0.08 m, E=5000 Pa and ν=0.35. A known uniform nominal traction gives total right-face forces of 1 N or 3 N. The left face has a finite axial spring with total stiffness 200 N/m; lateral faces remain traction free. Exact transverse symmetry conditions remove rigid modes while allowing Poisson contraction. All axial degrees of freedom remain unknown. The static solver eliminates prescribed degrees of freedom and solves the original elastic-plus-spring-minus-dead-load potential with sparse Newton, positive-J checks, and Armijo acceptance. There is no mass, artificial time step, stiffness shift, or inverse fit.

For homogeneous F=diag(s,t,t), write y=t². Independent scalar root solves enforce

```
y - 1 + (lambda/mu) log(s*y) = 0
A * mu * (s - y/s) = applied_force
left_translation = applied_force / spring_stiffness
```

The first equation imposes zero lateral Piola traction. The second uses the reference end-face area A, consistently with a known nominal dead traction. It does not substitute current-area Cauchy stress for nominal stress.

| Known force | Axial strain | Left spring translation | Newton iterations, coarse/fine |
| --- | ---: | ---: | ---: |
| 1 N | 2.119385% | 5 mm | 3 / 3 |
| 3 N | 6.580441% | 15 mm | 4 / 4 |

All four cases start from rest. The exact consistent-face variant has maximum position error 5.85×10⁻¹⁵ m and maximum Piola stress error 1.53×10⁻¹⁰ Pa against the independent analytical solution. Original free-force residuals remain below 1.2×10⁻¹² N. The preserved area-lumped variant gives the same homogeneous solution and coarse/fine common-node differences below 3.5×10⁻¹⁷ m.

Version 1 uses vertex-area quadrature for the left spring, which is exact for the constant left-face displacement in this oracle. Version 2 additionally assembles the fully consistent P1 surface spring matrix, integrating NiNj exactly as triangle-area/12 × (ones+identity). A separate test integrates the square of a nonuniform affine left-face displacement and distinguishes that exact integral from vertex lumping. Both versions and their frozen protocols/results remain intact.

Nine tests cover independent force/lateral balance, zero static rest, finite-difference energy gradients and tangents, finite rigid-frame covariance including loads and constraints, analytical stress/energy/nodal equilibrium, invalid determinant and incompatible constraints, and exact consistent face integration. The four solves take about 0.11 seconds per version on two CPU threads; this is not a controlled performance comparison.

The exact homogeneous field is representable by P1 tetrahedra, so this is a patch verification across two meshes, not a general continuum mesh-convergence certificate. Gaussian patches, contact, dynamic time integration, uncertain parameters, rendering, and physical calibration are outside this benchmark.

Sources: `experiments/campaign_static_uniaxial_bar.py`, `experiments/campaign_static_uniaxial_consistent.py`; frozen protocols: `runs/operator-dev/static-uniaxial-bar-v01/protocol.json`, `runs/operator-dev/static-uniaxial-consistent-v02/protocol.json`. Compact results use matching directories under `runs/gaussian-campaign-v01`. Raw displacement/stress arrays remain local.
