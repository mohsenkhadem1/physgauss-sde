# C38: forward mechanics milestone, campaign continuing

The controlled smooth pulling motion now passes the declared conditional
space/time indicator at 12.920 µm. A new FP64 GPU implementation reproduces
independent CPU results. Broader supports and loads, contact, dataset admission
and prediction remain in progress. No learned prediction or probing superiority
is established by this milestone.

The subsequent [mechanics-control report](GAUSSIAN_C38_MECHANICS_CONTROLS.md)
records the strict Warp/Torch trajectory pass, the capped larger cantilever,
and the preconditioner timing tradeoff. Varied supported-block qualification
continues separately; its preparation artifacts do not admit a dataset.
The completed [supported-family matrix and graded pilot](GAUSSIAN_C38_SUPPORTED_FAMILY_PROGRESS.md) now record seven uniform spatial passes, seven failures and two further graded failures. No varied cohort is admitted.

## Subsequent bending and impact checks

The same informative bending case (E=4,500 Pa, 0.12 N transverse end force,
full left clamp) has now completed graded 96,000- and 165,888-tet solves.
Both pass the original residual, force/moment balance and fixed-node checks.
Their 63-query difference is 5.063 µm, giving a conditional spatial indicator
of **25.591 µm: still above the unchanged 20 µm gate**. Earlier uniform and
graded failures remain published. Clamp-edge peak strain grows under refinement;
it is not a converged peak-stress result or a qualified moderate-strain dataset.

A separate whole-element Jacobian diagnostic evaluates the 20 cubic Bernstein
coefficients of det(F) for straight-reference quadratic elements. It gives
positive lower bounds of 0.5563 and 0.4511 on these saved fields, compared with
the earlier, more pessimistic norm bounds of 0.02965 and 0.00917. This checks the
whole computed affine-F polynomial with FP64 arithmetic margins and supplied
reference gradients; reference-geometry inversion is not interval enclosed.
A nonpositive Bernstein bound would be inconclusive, not proof of inversion.
No physical solution or old diagnostic was changed.

The streamed quadrature implementation matches the frozen 12,000-tet GPU field
to 3.90e-15 m from rest, with a free residual of 9.28e-13 N. It reduces memory
by recomputing quadrature chunks and preserves the full original operator and
consistent mass. The subsequent 393,216-tet Q3 bending solve reached its
40-minute wall limit during a linear solve without a usable converged field.
It provides no finer spatial comparison. A stronger preconditioner is being
benchmarked on the same smaller linear system before another large solve.
The separately checked positive 24-point degree-six rule preserves the exact
constant-material polynomial bulk operator and the original 80-point Jacobian
guards; this does not qualify contact surface quadrature or spatial error.

The high-order impact integrator completes two accuracy settings on the same
96-tet model and finite compliant plane. Their position difference is only
0.154 µm, and both energy checks pass. Their maximum nodal velocity difference
is **0.516 mm/s, failing the unchanged 0.1 mm/s gate**. A subsequent third
accuracy setting completes the next comparison: position difference 4.292 nm,
maximum nodal velocity difference 0.02475 mm/s, and pair energy difference
13.20 nJ. All unchanged temporal gates pass for this fixed mesh, Q8 surface
rule and finite compliant plane. The earlier failed pair remains. Contact
surface-quadrature and spatial refinement are still required.

Separately, a fixed learned-posterior control has been frozen for the validated
small logarithmic-NH spring-bar family: 4,096 training objects, 256 development,
128 calibration and 128 test objects, with three fixed training seeds. Test
decisions and predictions must freeze before test labels; development and
calibration gates must pass before any test release. This is an ideal endpoint
sensor experiment, not RGB-D inference or dynamic forecasting. Training and
validation were subsequently completed: all three learned posteriors failed
the joint and field-prediction gates in both development and calibration.
All 32 continuous reference cases passed numerical checks, identifying a
learned approximation error. The test set remains closed. The [complete learned
control report](GAUSSIAN_C38_LEARNED_POSTERIOR.md) retains every seed.

The exact80 bulk rule has also been used to correct the 165,888-tet bending
equilibrium from its frozen Q3 field. One Newton step reduces the exact80 free
residual from 4.83e-8 to 1.06e-12 N. The maximum all-node change is 1.20e-9 m;
the maximum change at the 63 comparison queries is 1.03e-10 m. This is a
quadrature-continuation check, not an independent basin test. The previous
spatial failure and unconverged clamp-edge peak strain of 61.6% remain.

Four new 96-tet gravity controls use right-face, both-end, bottom-face and two
separated bottom-patch supports. Independent sparse CPU and streamed CPU
solutions agree to 1.22e-15 m, with original residuals below 2.4e-12 N and
positive whole-element Jacobian bounds. They use E=100,000 Pa, density 625 kg/m³
and gravity 9.81 m/s². The both-end case has 0.404 mm nonrigid coordinate RMS and
2.32% peak strain on the coarse mesh. Subsequent exact80 refinement through
2,592, 6,144, 12,000 and 20,736 tetrahedra passes the conditional spatial
indicator at **14.238 µm**. The fine field has 0.436 mm nonrigid coordinate RMS,
4.53% peak strain and whole-element Jacobian lower bound 0.9690. Its separate
96-tet independent sparse CPU/GPU exact80 comparison differs by 9.63e-16 m.
This qualifies the declared static displacement indicator for one gravity
case; it does not admit a varied or dynamic training cohort.

An additional geometry control rejects a node mask when triangles cross the
declared edges of finite support patches. It checks the complete fixed area
and straight quadratic reference edges. It preserves the supplied mesh and
mask; a plausible set of fixed nodes alone does not prove the intended
continuous boundary condition.

## Checked forward results

| Check | Result | Meaning |
|---|---|---|
| CPU stable-NH P2 static, 6,144 to 12,000 tetrahedra | Maximum fixed-query change 3.642 µm; conditional four-times indicator 14.568 µm | Passes the prospective 20 µm engineering indicator under assumed order at least one; not a rigorous continuum bound |
| Independent 12,000-tet integration orders 3 and 4 | Query change 0.000031 µm | Quadrature agreement for this static pulling case |
| GPU P2 against both fine CPU solutions | Maximum nodal difference below 0.000001 µm; free residual below 1e-12 N | Implementation parity, not an independent continuum proof |
| Seven full consistent-mass transient CPU/GPU comparisons | Maximum all-state nodal difference below 3.6e-13 m | Same discrete BDF2 equations agree, including velocities and mass action |
| Smooth load, 96-tet GPU temporal refinement | 960→1,920 Hz: 8.173 µm; 1,920→3,840 Hz: 2.262 µm | Passes the direct temporal-difference target on this mesh and history |
| Same smooth load, 96→768 tets at 3,840 Hz | 87.391 µm | Fails the spatial target; these trajectories are not admitted as resolved truth |
| Same smooth load, 6,144→12,000 tets at 3,840 Hz | 3.081 µm | Conditional four-times spatial indicator 12.324 µm |
| Same smooth load, 12,000 tets, 3,840→7,680 Hz | 0.596 µm | Direct temporal-difference target passes |
| Combined conditional indicator | 12.920 µm | Four times spatial difference plus temporal difference; passes 20 µm for this case only |

The static block has E=5,000 Pa, Poisson ratio 0.35, a full left clamp and a
known 1 N right-face pull. Its fine CPU solution has about 6.45% peak strain
and 1.293 mm volume-weighted nonrigid coordinate RMS. Thus this particular
experiment contains substantial deformation after rigid motion is removed.
These figures do not describe a population of objects.

The dynamic test lasts 0.1 s with a sin-squared force ramp over its first
0.05 s, then a 1 N hold. Comparisons use 63 fixed material queries at 25 shared
times. Captured fine states have a sufficient whole-element positive-J lower
bound of at least 0.882. This certificate applies at recorded frames; unsaved
integration states and continuous time are not certified. The combined
indicator assumes spatial order at least one; it is not a rigorous error bound.

The final 7,680 Hz case received its own declared resource phase. The previous
phase was explicitly stopped during that incomplete case after the two completed
mesh captures were saved; those captures were reused without rerunning them.
Both the resource preemption and the successful continuation remain recorded.

## New materials, forces and supports

Seven prospective static combinations include bending, twisting, a top load,
four material stiffnesses and supports at several locations. Four restrictive
or weak-motion controls pass the displacement indicator. The informative
cantilever bending cases fail it at 185 and 87.7 µm. A twisting solve initially
hits the linear iteration cap; an exact nodal-block preconditioner repairs that
numerical failure while preserving the force law. The resulting 1.93 mm
nonrigid twisting solution still fails the spatial indicator at 167.8 µm.

The stronger transverse case has 1.48 mm nonrigid RMS and is being refined
further. Peak clamp strain increases with refinement; no converged peak-stress
claim is made. Failed cases are retained and do not become qualified training
data because their visual deformation looks plausible.

A two-level compatible preconditioner reduces the fine 49,152-tet solve to
49.23 s while retaining the full original FP64 operator and force checks. The
uniform-mesh spatial indicator still fails at 97.34 µm. A straight-reference
mesh graded toward the clamp and edges improves the tested indicator to
46.56 µm, which also remains a failure. Its physical geometry, force and support
are unchanged. Larger graded meshes are in progress. Clamp strain peaks grow
to about 44% on the graded mesh, while volume-weighted nonrigid motion stays
near 1.48 mm; a local clamp singularity is not a converged bulk strain estimate.

An optional mixed-precision linear correction also passes three independent
fine-solution comparisons (pulling, bending and twisting), with nodal differences
below 7e-13 m and original FP64 force residuals below 6.6e-12 N. It uses FP64
residual refinement and a full FP64 fallback. This changes numerical work, not
material, mass, damping or boundary conditions. These static checks do not
qualify it for arbitrary dynamic/contact cases.

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

An independently assembled CPU contact trajectory agrees with the GPU at all
73 recorded times to 2.11e-14 m. Thus matching the same discrete equations does
not remove resolution error. Even analytic-freefall-initialized BDF2 rates up
to 15,360 Hz fail the 20 µm refinement target. A separate high-order adaptive
ODE check is underway with the same mass, material and floor stiffness.

Its first accuracy setting completes with energy drift below the prospective
limit. The tighter setting hits the declared 900 s resource limit before the
physical interval ends. A separate tight-only continuation keeps the same
tolerances and imports the completed first case. No temporal qualification is
claimed until both settings complete and agree. The new driver checks every
accepted state and checkpoints the committed trajectory prefix.

## Evidence and limits

- [Independent fine CPU spatial report](../runs/gaussian-campaign-v01/newton-stable-p2-spatial-v01/REPORT.md)
- [GPU fine static comparisons](../runs/gaussian-campaign-v01/p2-torch-spatial-v01/publication-summary.json)
- [GPU full-mass transient parity](../runs/gaussian-campaign-v01/p2-torch-time-v01/publication-summary.json)
- [GPU coarse temporal/spatial refinement](../runs/gaussian-campaign-v01/p2-gpu-refinement-v01/publication-summary.json)
- [Fine dynamic space/time continuation](../runs/gaussian-campaign-v01/p2-gpu-refinement-v03/publication-summary.json)
- [Varied static controls](../runs/gaussian-campaign-v01/p2-varied-static-v01/publication-summary.json)
- [Optional mixed-solver comparisons](../runs/gaussian-campaign-v01/p2-mixed-benchmark-v01/publication-summary.json)
- [Contact diagnostic inventory](../runs/operator-dev/newton-contact-validation-v01/publication-inventory.json)
- [CPU transient terminal report](../runs/gaussian-campaign-v01/p2-consistent-transient-v01/REPORT.md)

One of ten CPU reference trajectories hit its fixed 300 s resource cap; nine
completed with strict residual and constraint checks. This is recorded as an
incomplete stage, not a numerical pass. The seven GPU parity cases used the
first available completed CPU prefix selected before their GPU outcomes.

Raw fields and media stay local. Only source and compact reports are pushed.
The next stages resolve the broader force/support combinations and drops,
then admit qualified data for strictly causal prediction.
The small material/support posterior study is a separate reference problem;
neither numerical parity nor that reference alone establishes general learned
boundary conditions or useful uncertainty-guided probing.

The [completed small reference](GAUSSIAN_C38_BAR_REFERENCE.md) now passes all
64 objects and 32 independent integration checks. Its acquisition results match
the simple opposite-endpoint heuristic rather than showing an advantage over it.
A fresh learned-posterior control is being prepared; no broad physical RGB-D
training dataset has been admitted.
