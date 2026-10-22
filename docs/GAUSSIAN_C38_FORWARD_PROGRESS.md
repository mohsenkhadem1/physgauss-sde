# C38: forward mechanics milestone, campaign continuing

The higher-order static reference now passes the declared spatial indicator,
and a new FP64 GPU implementation reproduces independent CPU results. Dynamic
mesh qualification, broader supports and loads, contact, dataset admission and
prediction remain in progress. No learned prediction or probing superiority is
established by this milestone.

## Checked forward results

| Check | Result | Meaning |
|---|---|---|
| CPU stable-NH P2 static, 6,144 to 12,000 tetrahedra | Maximum fixed-query change 3.642 µm; conditional four-times indicator 14.568 µm | Passes the prospective 20 µm engineering indicator under assumed order at least one; not a rigorous continuum bound |
| Independent 12,000-tet integration orders 3 and 4 | Query change 0.000031 µm | Quadrature agreement for this static pulling case |
| GPU P2 against both fine CPU solutions | Maximum nodal difference below 0.000001 µm; free residual below 1e-12 N | Implementation parity, not an independent continuum proof |
| Seven full consistent-mass transient CPU/GPU comparisons | Maximum all-state nodal difference below 3.6e-13 m | Same discrete BDF2 equations agree, including velocities and mass action |
| Smooth load, 96-tet GPU temporal refinement | 960→1,920 Hz: 8.173 µm; 1,920→3,840 Hz: 2.262 µm | Passes the direct temporal-difference target on this mesh and history |
| Same smooth load, 96→768 tets at 3,840 Hz | 87.391 µm | Fails the spatial target; these trajectories are not admitted as resolved truth |

The static block has E=5,000 Pa, Poisson ratio 0.35, a full left clamp and a
known 1 N right-face pull. Its fine CPU solution has about 6.45% peak strain
and 1.293 mm volume-weighted nonrigid coordinate RMS. Thus this particular
experiment contains substantial deformation after rigid motion is removed.
These figures do not describe a population of objects.

The GPU code uses the same polynomial stable Neo-Hookean law as the pinned
Newton 1.7 comparison, including the complete determinant Hessian. It uses
quadratic ten-node tetrahedra and the exact consistent mass matrix. The
matrix is positive definite although some entries and vertex row sums are
negative; it is never converted to row-lumped particle masses. There is no
added physical damping or material clipping. BDF2 uses backward-Euler startup,
and each step checks the original full-mass momentum equation.

## Gravity and contact findings

Five separate no-contact Newton static gravity controls agree with independent
CPU reference queries within 2.11 µm. This checks static equilibrium, not
physical-time falling.

The stock backward-Euler Newton drop experiments remain negative. Their
tested time-step field difference is 4.818 mm, spatial difference 0.7395 mm,
and floor-stiffness difference 30.18 µm. Increasing solver iterations changes
the field by only 1.095 µm. Accurate iteration convergence therefore does not
resolve the time, space or floor-law errors.

A new FP64 P2 contact component integrates a finite-compliance, frictionless
fixed plane over the quadratic boundary trace with positive surface
quadrature. Its CPU force, tangent and penetration controls pass. The plane is
not a demonstrated rigid floor: temporal, spatial, surface-quadrature and
stiffness refinement remain necessary before dropping data can be admitted.
Earlier failed captures and resource-limited runs remain recorded.

## Evidence and limits

- [Independent fine CPU spatial report](../runs/gaussian-campaign-v01/newton-stable-p2-spatial-v01/REPORT.md)
- [GPU fine static comparisons](../runs/gaussian-campaign-v01/p2-torch-spatial-v01/publication-summary.json)
- [GPU full-mass transient parity](../runs/gaussian-campaign-v01/p2-torch-time-v01/publication-summary.json)
- [GPU coarse temporal/spatial refinement](../runs/gaussian-campaign-v01/p2-gpu-refinement-v01/publication-summary.json)
- [Contact diagnostic inventory](../runs/operator-dev/newton-contact-validation-v01/publication-inventory.json)
- [CPU transient terminal report](../runs/gaussian-campaign-v01/p2-consistent-transient-v01/REPORT.md)

One of ten CPU reference trajectories hit its fixed 300 s resource cap; nine
completed with strict residual and constraint checks. This is recorded as an
incomplete stage, not a numerical pass. The seven GPU parity cases used the
first available completed CPU prefix selected before their GPU outcomes.

Raw fields and media stay local. Only source and compact reports are pushed.
The next stages refine the dynamic mesh and test new force, material and
support combinations, then admit qualified data for strictly causal prediction.
The small material/support posterior study is a separate reference problem;
neither numerical parity nor that reference alone establishes general learned
boundary conditions or useful uncertainty-guided probing.
