# Newton VBD material and static-reference alignment

Newton 1.7.0 VBD and the previous C36 CPU reference use different finite-strain energies. Their small-strain Young's modulus and Poisson ratio agree, but using C36's log-J response directly as Newton ground truth would confound material and solver errors. A separate matching-law CPU reference now provides six converged equilibria; this does not yet qualify Newton's GPU solver or the stable-law continuum solution.

## Installed-source derivation

The installed source is `newton/_src/solvers/vbd/particle_vbd_kernels.py`, function `evaluate_volumetric_neo_hookean_force_and_hessian`, lines 179 onward. Its SHA256 is `79a5075c62d53b2888858c3380bdae4a1b4f5c9a284a09206c88e37918c1241e`. `ModelBuilder.add_tetrahedron` stores `(k_mu, k_lambda, k_damp)` unchanged, and the VBD kernel consumes those columns as the Lamé shear modulus μ and first Lamé modulus λ. It sets κ = λ + μ and α = 1 + μ/κ for the declared material range.

Subtracting the constant rest energy, the literal kernel's stress integrates to

W(F) = μ/2 (||F||² − 3) − μ(J − 1) + κ/2 (J − 1)²,

P(F) = μF + [κ(J − 1) − μ] cof(F).

Thus P(I) = 0. The usual authored conversion μ = E/[2(1+ν)] and λ = Eν/[(1+ν)(1−2ν)] is correct: expansion about F = I gives W = μ||sym G||² + λ/2(tr G)² + O(G³). No extra E/ν adjustment is needed. The VBD kernel already applies the required λ→λ+μ conversion internally; supplying λ+μ as its input would count it twice.

There is a range qualification: the native kernel uses `max(κ, 10^-6)` in α for positive κ. The formulas above and the new CPU variant exactly match the installed elastic force for κ ≥ 10^-6 Pa, including every campaign material. Here E = 5,000 Pa and ν = 0.35 give μ = 1,851.85185 Pa, λ = 4,320.98765 Pa, and κ = 6,172.83951 Pa. At hypothetical sub-micro-Pascal κ the native guard changes the law and can introduce rest stress; no general equivalence claim is made there.

The full continuum tangent is μI₄ + κ cof(F)⊗cof(F) + s D²J, where s = κ(J−1)−μ. VBD's per-P1-vertex 3×3 block omits the last term. This is exact for that diagonal contraction: a single vertex changes F by a rank-one matrix, so J is affine in that vertex. The cross-vertex blocks and the full material Hessian retain D²J. The new CPU P1/P2 reference does so; the test explicitly detects nonzero cross-block contributions.

This stable law is polynomial and remains finite at J = 0; it is not the old log-J energy and has no logarithmic inversion barrier. The CPU equilibrium reference separately restricts its branch to J > 0.1. It does not claim that Newton enforces this restriction. Native VBD element damping acts on the rate of C = FᵀF, not on the earlier body's linear drag. The primary matching comparison therefore uses zero element damping and no gravity/contact.

## Independent tests and equilibrium captures

Three tests passed in 1.80 s. They check rest stress and the independently derived isotropic small-strain tangent, finite rigid-rotation objectivity, force-energy and full tangent finite differences, the exact P1 diagonal-block identity, full tetrahedron force/tangent finite differences, and an independent homogeneous uniaxial oracle solved by a P2 model at about 7% extension. For axial stretch s and transverse squared stretch y, that oracle uses y = [(κ+μ)s−μ]/(κs²) and P11 = μ(s−y/s), obtained directly from zero transverse stress. No log-J oracle is reused.

All six predeclared static solves completed in 211.53 s on two CPU threads. Geometry and dead load are the 240 × 80 × 80 mm block, exact left-face clamp, 1 N uniform right-reference-face traction, E = 5 kPa, ν = 0.35, and density 625 kg/m³. P1 meshes have 96, 768, and 6,144 tetrahedra. P2 meshes have 768 and 6,144 tetrahedra, with a separate 64-point quadrature solve on the finest mesh. Every case starts from rest. All pass the fixed 10^-11 N original free-force residual, exact constraints, positive-J branch, and 10^-8 N / 10^-8 Nm resultant/moment balance checks. No stiffness repair, mass-as-time-step static surrogate, damping, material retuning, or retry is used.

| Same-mesh stable-law versus old log-J query difference | Maximum Euclidean difference |
|---|---:|
| P1, 96 tetrahedra | 47.281 µm |
| P1, 768 tetrahedra | 26.319 µm |
| P1, 6,144 tetrahedra | 16.883 µm |
| P2, 6,144 tetrahedra, 27-point rule | 13.261 µm |

These are constitutive-law differences, not Newton solver errors. All fixed volume, surface, edge, and corner probes are retained. Within the new stable law, the P1 768→6,144 spatial difference remains 376.840 µm; P2 reduces it to 22.823 µm, still above the unchanged 20 µm gate. P2 27→64-point quadrature changes the maximum query by only 0.0000485 µm. The stable-law P2 spatial gate fails; the C36 log-J qualification is not transferred to this different law, and no extra fine continuum study was launched.

## How the GPU comparison must be interpreted

The primary reference is the **same-discretization** P1 equilibrium: `P1-tets96.npz`, then a prospectively chosen finer shared mesh if authorized. Each local capture contains rest positions, tetrahedra, surface triangles, equilibrium positions, exact known nodal loads, full elastic forces, the exact constraint mask, and the common 122 material probes with reference values. Native input masses must use the same physical P1 tetrahedron volume weights, and boundary flags must be recorded.

A transient backward-Euler residual or visual settling is not proof of static equilibrium. At the final GPU state, independently assemble the original native elastic forces plus the known external force on free nodes, record velocities and constraint error, and compare positions to the matching CPU equilibrium. The native kernel can be called in a separate force-audit kernel; state `particle_f` need not be an internal-force getter. Float32 rest positions, inverse rest matrices, materials, and masses should be saved, and the elastic force floor at rest should be reported. GPU precision need not attain the CPU's 10^-11 N threshold, but a separately declared GPU residual tolerance must be met and raw residuals retained.

No GPU accuracy, speed, static convergence, dynamic forecasting, contact, inverse inference, or uncertainty-calibration result is claimed by this CPU alignment stage. It removes a concrete material mismatch and supplies the reference needed to evaluate a bounded solver correction. All old C36 fields and successes/failures remain unchanged.

See the [compact results](compact-summary.json), [pre-run protocol](../../operator-dev/newton-material-alignment-v01/protocol.json), and [terminal receipt](../../operator-dev/newton-material-alignment-v01/terminal.json). Raw reference NPZ arrays and detailed histories remain local.
