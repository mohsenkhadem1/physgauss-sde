# Rectangular blocks, fixed supports, gravity and impact

This campaign adds a 0.24 × 0.08 × 0.08 m block with density 625 kg/m³, Poisson ratio 0.35 and gravity (0, 0, −9.81) m/s². The nominal mass is 0.96 kg. It follows the [load/mesh diagnosis](GAUSSIAN_LOAD_MESH_REPAIR.md); previous failed gates and the sealed prediction comparison remain unchanged.

## Recorded Isaac scenarios

Eight prospective scenarios completed with one GPU0 application at a time. Five gravity cases use E=300 kPa and exact fixed nodes on the left face, right face, both ends, bottom face, or two separated bottom patches. Each patch occupies the full bottom width over the first/last 0.06 m of the block. A no-floor freefall control uses E=300 kPa. Two floor drops use E=300/600 kPa and an initial bottom-to-floor gap of 0.05 m. There are no load handles or attachments.

Native checks read back material E/ν, rest geometry and tetrahedra, gravity, fixed/free target flags, actual step clocks and GPU identity. Fixed-node drift was zero. Native nodal masses are not exposed, and authored density is not presented as a direct mass readback. The exact native constitutive law was not established from the installed sources; matching E/ν does not prove equivalence to the independent Neo-Hookean reference.

| Scenario | Peak recorded proper-rigid-removed coordinate RMS |
|---|---:|
| Left face fixed, gravity | 3.068 mm |
| Right face fixed, gravity | 2.923 mm |
| Both ends fixed, gravity | 0.689 mm |
| Bottom face fixed, gravity | 0.048 mm |
| Two bottom patches, gravity | 0.240 mm |
| Free fall, no floor | approximately 0.000006 mm |
| Drop, E=300 kPa, dense capture | 0.652 mm |
| Drop, E=600 kPa, dense capture | 0.411 mm |

These are development demonstrations, not a trained prediction benchmark. Rigid-motion removal uses a proper mass-weighted fit with volume-lumped weights; it is an oracle diagnostic over the full recorded body, not an available sparse-view predictor.

The freefall control passes its predeclared velocity and shape checks: maximum centre-velocity error 1.71 µm/s and maximum |J−1| about 2.30×10⁻⁶. Position agrees with kick-drift integration to about 9.53 nm coordinate RMS but differs from the exact constant-acceleration trajectory by 1.384 mm RMS. Both comparisons are retained: this exposes a time-discretization bias rather than certifying exact ballistic positions.

## Time and contact accuracy remain unresolved

The separately frozen 240/480/960 Hz native cantilever check fails the 20 µm maximum material-query tolerance. Differences are **15.514 mm** for 240→480 Hz and **5.645 mm** for 480→960 Hz. All native setup checks pass, so correct authored settings alone do not establish a resolved dynamic response. A controller directory error in the first refinement attempt occurred before any Isaac call; that failure is preserved, and a separately tested second controller ran the unchanged experiment.

Drops recorded at 30 fps missed the brief peak. A separately frozen observation-only variant retains identical physics and saves all 145 physics states at 240 Hz. These reveal peak strain about 14.0%/13.8% and maximum nodal floor penetration **3.529/2.826 mm**. The floor is a finite static collider. These results do not qualify contact accuracy or exact nonpenetration.

Videos are rendered from actual recorded nodes, not native RGB images or predictions. They use true geometric scale, a 45° camera orbit, five-times-slow playback and a separate rigid-removed panel. The dense impact videos display every saved state at 60 fps with no motion interpolation. The local eight-scenario overview is `runs/gaussian-campaign-v01/block-support-gravity-overview-v01/supports-gravity-and-drop.mp4`. Raw trajectories and all media remain local; only source and compact provenance are published. The [development dataset index](../runs/gaussian-campaign-v01/block-support-gravity-v01/dataset-index.json) binds all eight scenarios and explicitly marks them ineligible as validated native physical-inference references.

## Independent static reference

A separate compressible Neo-Hookean solver uses exact physical support masks, uniform face traction and consistent body gravity. The prospective P1 suite covers six load/support cases on 96/768/6144 tetrahedra. All 18 solves finish without inversion, but no case jointly passes its original spatial/balance gates. The axial case produces 3.79% maximum strain and 1.29 mm volume-integrated nonrigid RMS. Gravity cantilevers sag about 15.7 mm, but their rigid-removed bulk RMS is only 0.45 mm. The bottom-supported case has a 3.85 µm spatial difference but narrowly fails its independent 10⁻⁸ N global balance gate.

P1 displacement differences for informative cases remain hundreds to thousands of micrometres. A separately frozen P2 investigation adds globally shared edge nodes, positive volume quadrature and the same continuum loads/supports. All eight solves complete in **378.16 s**. Force/moment balance, fixed constraints and 27/64-point quadrature comparisons pass. A sufficient global positive-J bound is also checked using the affine P2 deformation-gradient field. Spatial differences fall to **22.353 µm axial / 163.344 µm gravity**, both still above the unchanged 20 µm gate. The axial improvement is about sixteen-fold relative to P1. Its sampled peak strain is 5.91% and volume nonrigid RMS 1.295 mm.

A separately declared axial-only resolution continuation retains those failures and reuses immutable prior captures. It adds a 12×6×6-cell mesh between the previous 8×4×4 and 16×8×8 levels. Qualification requires the same physical/query gates plus a first-order fine-error estimate three times the 12→16 discrepancy below 20 µm. This is an estimate under at-least-first-order asymptotic convergence, not a rigorous continuum error bound. One optional 20×10×10 refinement is allowed only if the fixed rule fails and the prospectively recorded resource forecast fits. No load, material, boundary or acceptance-threshold tuning is allowed. The intermediate mesh gives 7.997 µm discrepancy and a 23.991 µm first-order estimate, which **fails** 20 µm. A resource-only amendment based on already measured runtimes was recorded before reading that new result; it permits a separate 900-second/8-GiB final stage without changing physics or gates.

The final **12,000-tetrahedron P2 axial case passes the declared static fixed-query checks AND the excitation goal**. Both integration rules are solved from rest. The 6,144→12,000 difference is **3.522 µm**, with conditional first-order fine-error estimate **14.088 µm**, below 20 µm. Changing quadrature from 27 to 64 points changes the queried positions by only **0.000101 µm**. Peak sampled principal strain is **6.487%**, volume nonrigid coordinate RMS **1.295 mm**, and maximum nodal displacement **7.609 mm**. Constraints remain exact; force/moment defects are approximately **3.67×10⁻¹² N / 7.24×10⁻¹² Nm**. Minimum sampled J is **0.989608**, with a sufficient whole-element determinant lower bound above **0.758**. The final pair takes **532.03 s**, with observed peak RSS **2.554 GiB**.

This is a numerically checked **static Neo-Hookean block at one prescribed load, material and known clamp**, at the fixed material query set and under the stated convergence assumption. It does not establish whole-field continuum error bounds, native dynamic equivalence, gravity/contact convergence or inference across unknown boundary conditions. [Independent read-only mechanics review](../runs/operator-dev/static-p2-independent-review-v01/review.json).

## Repeated-noise uncertainty check

The independent nonlinear bar inverse was checked with 64 fresh noise realizations at one fixed truth, using known support location/law and the unchanged prior and sensor noise. All 64 numerical checks pass. Coverage of nominal 90% intervals is 54/64 for E, 55/64 for K and held-left displacement, and 56/64 for held-right displacement. Exact marginal 95% binomial confidence intervals all include 90%; they remain wide. This is conditional coverage at one synthetic truth, not calibration over unknown boundaries, geometries or real physics. [Detailed coverage evidence](GAUSSIAN_BAR_NOISE_COVERAGE.md).

Neural training, generalized boundary inference and active probing remain gated on trustworthy forward predictions. The native time/contact failures are not bypassed by the passing small inverse.
