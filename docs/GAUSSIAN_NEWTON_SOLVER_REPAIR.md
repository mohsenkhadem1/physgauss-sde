# Solver repair: matched mechanics before a new dataset

This campaign tests two concrete repairs: increasing Isaac's deformable solver
iterations without changing the physical problem, and checking standalone
Newton VBD against an independent reference using its actual material equation.
Old captures and failed checks remain unchanged. Raw trajectories and videos
remain local.

## Fixed physical controls

The numerical controls use a 0.24 × 0.08 × 0.08 m block, density 625 kg/m³,
Poisson ratio 0.35 and exact fixed nodes. Isaac's gravity experiment uses the
previous 300 kPa left-clamped block; its iteration/timestep ablation changes
only numerical settings. Newton's separate static traction experiment uses
5 kPa and a 1 N load distributed over the right reference face. These are
separate experiments, not an engine-to-engine accuracy comparison.

## Material equation and mass correction

The isolated runtime pins Newton 1.7.0 and Warp 1.18.0. It does not replace the
project or Isaac environments. Installed source, runtime versions and experiment
sources are recorded by hash before each run.

Newton VBD's tetrahedral kernel uses the stable Neo-Hookean energy

```
mu = E / (2 (1 + nu))
lambda = E nu / ((1 + nu) (1 - 2 nu))
kappa = lambda + mu
W(F) = mu/2 (||F||² - 3) - mu (J - 1) + kappa/2 (J - 1)²
P(F) = mu F + (kappa (J - 1) - mu) cof(F)
```

The builder receives standard Lamé parameters; it must **not** receive
`lambda + mu` as its lambda argument because the kernel already adds mu.
This law matches the previous logarithmic Neo-Hookean model at small strain,
but differs at finite strain. The independent CPU implementation uses the full
energy Hessian, including determinant second derivatives between different
vertices. The native per-vertex diagonal block is checked separately.

Particle masses are assembled from the actual tetrahedron reference volumes,
with one quarter of each element's physical mass assigned to each vertex.
Fixed particles retain their physical mass and are constrained through flags.
The block's total mass is 0.96 kg. Particle radius, gravity, material arrays,
fixed flags, rest geometry and native device assignment are checked explicitly.

## Interpretation

A passing same-mesh GPU/reference comparison establishes a discrete solver
check. It does not establish continuum accuracy, transient accuracy, calibrated
material inference or contact accuracy. Pseudo-time relaxation with velocity
resets is an algorithm for finding static equilibrium; its frames are not a
physical trajectory. Impact penetration and unknown boundary inference require
separate validation.

The initial tiny Newton kernel probe is preserved as a failure: the ideal
proper-rotation force discrepancy was 2.102 µN against a 2 µN absolute gate.
The follow-up evaluates the independent formula at the actual float32 inputs,
retains the ideal-rotation discrepancy and keeps the original error thresholds.
It separates input quantization from implementation parity rather than erasing
the failed result.

An initial Isaac GPU-selection attempt failed its device guard before any
recorded physics step. Its replacement uses the installed SDK's documented
`active_cuda_gpus` mapping and a physical GPU UUID, with two CPU threads
explicitly configured. This is a resource correction; physical settings are
unchanged.

## First controlled Isaac result

All four fresh runs completed, with unchanged geometry, force and material:

| Comparison | Maximum saved material-query displacement difference |
| --- | ---: |
| 32 → 128 iterations, 240 Hz fixed | 14.5732 mm |
| 32 → 128 iterations, 480 Hz fixed | 0.436532 mm |
| 240 → 480 Hz, 32 iterations fixed | 15.5143 mm |
| 240 → 480 Hz, 128 iterations fixed | 5.36750 mm |

The higher iteration count reduces the timestep discrepancy by about 65%,
but it still fails the unchanged 20 µm query threshold. Insufficient iterations
therefore contaminate the earlier timestep comparison; this experiment does
not yet separate all remaining iteration and integration error. The two fresh
32-iteration runs are bit-identical to their previous GPU-0 captures, despite
running on GPU 1 with the corrected resource mapping. Fixed-node drift remains
zero and sampled element determinants remain positive.

## Independent same-law CPU references

All six CPU solves completed in 211.53 s with strict original equilibrium,
constraint and global balance checks. P1 meshes with 96, 768 and 6,144 tetrahedra
provide references for corresponding discrete GPU models. The stable-law P2
768 → 6,144 spatial difference is 22.823 µm, which fails the 20 µm threshold;
quadrature refinement differs by only 0.0000485 µm. Thus these results provide
matched discrete references but do not extend the previous logarithmic-law
continuum qualification to Newton's material law.

The second tiny Newton probe still exceeds the near-zero 2 µN force-parity
threshold by a small amount after accounting for quantized inputs. That failed
check remains recorded. Static equilibrium is assessed separately with a
prospectively declared 1 mN original-force residual and 20 µm matched-reference
field threshold; it is not made conditional on relabelling the tiny probe.

The compact [matched-material report](../runs/gaussian-campaign-v01/newton-material-alignment-v01/REPORT.md)
records the derivation, reference solves and failed spatial check. The
[Isaac iteration report](../runs/operator-dev/isaac-timestep-forensic-v01/REPORT.md)
records the causal ablation. These reports retain failed outcomes as well as
passing setup checks.

## Isaac refinement outcome

The second controlled stage completed all four runs in 186.67 s. At 3,840 Hz,
128 → 255 iterations changes the fixed-query field by **9.739 µm**, passing
the 20 µm iteration check. Timestep refinement still fails:

| Fixed 128 iterations | Maximum fixed-query difference |
| --- | ---: |
| 480 → 960 Hz | 5.20989 mm |
| 960 → 1,920 Hz | 3.44838 mm |
| 1,920 → 3,840 Hz | 0.992987 mm |

The finest iteration check therefore passes while the tested timestep check
fails by about fifty times the threshold. This is a useful diagnosis, not a
qualified native dynamic dataset. Further expensive Isaac rate escalation is
paused; its contact and spatial accuracy also remain unresolved.

Newton's first static-relaxation attempt also failed its geometric guard.
The stock force predictor with a one-second **pseudo** timestep can propose
roughly 16.7 m of motion on a 0.24 m block. A separate initial-iterate repair
starts the minimizer at the last accepted configuration while preserving the
inertial target, energy, masses, loads and convergence thresholds. This repairs
our static optimization procedure; it is not evidence that Newton fails at
ordinary physical timesteps. The original failed attempt remains recorded.

## Newton static repair: passing result

The accepted-iterate repair completed all three predetermined runs from rest
in 8.67 s. Runtime checks prove that the inertial target is bit-identical before
and after the repair, while the minimizer starts at the accepted configuration.
The actual large predictor remains positive-J; the earlier failed capture does
not identify which later update inverted an element. The repair's effectiveness
is established by the following independent equilibrium checks:

| VBD iterations per outer solve | Outer steps | Original native free-force residual | Maximum CPU query error |
| --- | ---: | ---: | ---: |
| 32 | 30 | 18.55 µN | 2.566 µm |
| 128 | 10 | 13.01 µN | 0.4395 µm |
| 512 | 3 | 16.81 µN | 0.01471 µm |

All three pass the prospectively fixed 1 mN force, 0.5 µm consecutive drift,
20 µm same-mesh query, zero fixed drift and positive-J checks. Materials,
physical masses, load, physical damping and boundary conditions are unchanged.
This is a working, independently checked **96-tetrahedron static** solver.
[Full result and limitations](../runs/operator-dev/newton-vbd-static-v02/REPORT.md).

## Physical-time repair: passing controlled motion

The moving case uses the same 96-tetrahedron, 5 kPa, left-clamped block with a
1 N step traction for 0.1 s. Physical damping and gravity are zero. This is a
controlled force-driven trajectory, separate from the unresolved Isaac gravity
and impact examples.

An independent CPU solver checks backward Euler and second-order BDF2, solving
each nonlinear step against the original force equations. BDF2 starts with one
backward-Euler step. Its adjacent fixed-query differences decrease through
138.807, 76.513, 27.695, 7.321 and **1.819 µm** as the rate rises from 240 to
7,680 Hz. The final two references pass the unchanged original residual gates;
the maximum physical free-force residual is below 5.4e-14 N.

Two numerical repairs were necessary. The CPU reference retains displacement
relative to rest and evaluates the same constitutive equation directly from
that displacement. Its line search accounts for energy-evaluation roundoff and
still requires a decreasing original residual and the unchanged final force
tolerances. The repaired 3,840 Hz field matches the earlier completed reference
within 4.62e-14 m. Earlier failed fine-step runs remain preserved.

The GPU wrapper uses VBD to minimize the equivalent BDF2 inertial objective.
Computing velocity from tiny float32 position differences causes growing
roundoff error at high rates. The repaired wrapper accumulates momentum in
float64 using the **literal GPU internal forces**, known applied force and
read-back physical masses. It saves the chord velocity separately and checks
the resulting position/velocity kinematic defect. Native GPU positions are
never replaced by CPU predictions or by velocity-integrated positions. The
momentum residual is zero by construction and is explicitly **not** used as an
independent validation claim.

All four predetermined GPU comparisons pass:

| Physical rate | VBD iterations | Maximum position difference from matching CPU trajectory |
| --- | ---: | ---: |
| 1,920 Hz | 128 | 0.2512 µm |
| 3,840 Hz | 128 | 0.4385 µm |
| 7,680 Hz | 128 | 0.4810 µm |
| 3,840 Hz | 512 | 0.4545 µm |

The 1,920 Hz error falls from 19.925 µm with the original chord-velocity update
to 0.2512 µm with momentum recovery. The 128 → 512 iteration control differs
by 0.6406 µm at the fixed queries. Per-step kinematic defects stay below
21.92 nm against the prospectively fixed 100 nm gate; accumulated differences
from separately velocity-integrated positions remain below 0.532 µm. Those
integrated positions are diagnostics, not the simulated trajectory.

Posthoc saved-query timestep differences are 7.286 µm for 1,920 → 3,840 Hz
and 2.120 µm for 3,840 → 7,680 Hz. The selected 3,840 Hz GPU trajectory differs
from the 7,680 Hz CPU query field by 1.766 µm. These are finite comparisons,
not rigorous error bounds relative to a continuous-time or continuous-space
solution. CPU time qualification and GPU approximation checks are separate.

[CPU transient report](../runs/gaussian-campaign-v01/stable-transient-relative-v03/REPORT.md)
records all representations and retained failures. The selected local run is
`runs/gaussian-campaign-v01/newton-vbd-transient-v02/BDF2-hz3840-it128.npz`.
Its code path is `experiments/campaign_newton_vbd_transient_v02.py`, with
reusable momentum helpers in `experiments/campaign_newton_vbd_momentum.py`.

## Scope and next use

This establishes a checked static and short physical-time **discrete forward
model for one known material, load and clamp**. It supplies a usable controlled
Newton route for further experiments. It does not validate spatial convergence,
new loads/materials/supports, contact, physical parameter calibration or active
probing. The stable-law P2 spatial failure above remains unchanged. The next
mechanics gate is spatial refinement and held-out load histories, followed by
contact and uncertain-support tests; larger inference comparisons remain gated
on those checks.

The checked configuration and hash-bound posthoc diagnostics are recorded in
[configuration.json](../runs/operator-dev/newton-forward-selected-v01/configuration.json).
The local video is
`runs/gaussian-campaign-v01/newton-reference-video-v01/newton-and-cpu-reference.mp4`:
actual Newton nodes versus the independent same-step CPU reference, at true
geometric scale with a shared orbit camera. All 115 frames decode correctly;
recorded simulation states are held rather than interpolated. The video shows
simulation/reference agreement, not real-world measurements or learned forecasts.

Final validation: **48 focused tests pass** across constitutive derivatives,
independent time-stepping oracles, initialization preservation, momentum and
kinematic identities, resource guards and video provenance. All campaign
physics processes have exited; the unrelated job on GPU 3 was left untouched.
Code and compact text reports are published; raw arrays and media stay local.
