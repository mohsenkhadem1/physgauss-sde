# Nonlinear excitation development: finite strain achieved, forward accuracy fails

The prospective CPU2 benchmark completed all **39** captures in **108.20 s** under a 600 s cap. No solver failed, no tet inverted and no design candidate was removed. The current nonlinear reference is **not qualified for inverse inference or network training**: all three temporal gates and both mesh gates fail.

## Frozen design

The 5-node/two-tet material domain, public Gaussian centers/covariances, density625 kg/m³, Poisson ratio0.35 and damping stay fixed. Design pilots compare four known world-force patterns at0.25/1/4 N per patch. A0.2 s sin² pulse is distributed by normalized material-frame surface quadrature; controls are identical across E and fixture stiffness. Balanced pairs have zero total force; their moments are measured at actual deformed nodes and are not assumed constant.

The predeclared target requires2–8% maximum absolute principal stretch strain and at least1 mm coordinate RMS after removing the best proper rigid transform. The smallest passing amplitude is selected using design-only nominal E5000 Pa/fixture scale2. There are no posterior or test data in this selection.

| Pattern | Nominal accepted amplitude | Peak absolute strain at120 Hz | Peak nonrigid RMS |
|---|---:|---:|---:|
| legacy_single | 1 N/patch | 4.86% | 1.166 mm |
| balanced_axial | 1 N/patch | 7.40% | 2.442 mm |
| transverse_couple_1 | None | 1 N gives8.24%, above bound | 2.14 mm |
| transverse_couple_2 | 1 N/patch | 6.28% | 2.181 mm |

All12 pilot candidates remain saved. Larger4 N trials overshoot the strain target; they are failures of the excitation criterion despite solver completion. The three accepted patterns each receive all six E={2500,5000,10000} Pa × fixture-scale={2,10} cases. Only2/6 legacy,4/6 axial and3/6 transverse2 cases meet both excitation targets; no whole-factorial useful-strain envelope is claimed.

## Accuracy gates

| Nominal pattern | Full-trajectory60→120 /120→240 Hz error | Required final tolerance | Result |
|---|---:|---:|---|
| legacy_single | 329.9 / 182.6 µm | 63.8 µm | FAIL |
| balanced_axial | 218.8 / 160.4 µm | 129.8 µm | FAIL |
| transverse_couple_2 | 685.7 / 382.0 µm | 115.0 µm | FAIL |

Errors use the full trajectories and the same continuous force law with explicitly held midpoint quadrature at each integration rate. They contract but remain too large. The declared tolerance is max(2 µm,5% of finest peak nonrigid RMS); no zero-subtracted or rotation-removed trajectory substitutes for the gate.

The deterministic mesh pattern is balanced axial at1 N, E5000 Pa/fixture scale2,240 Hz. Conforming midpoint refinement preserves domain, shared vertices, boundary area, physical Gaussian widths and total applied force. Original nodes1/4 are the fixed held query locations.

| Mesh comparison | Held-node full-trajectory RMS error |10%/2 µm tolerance | Result |
|---|---:|---:|---|
|2→16 tets |0.911 mm |0.191 mm |FAIL |
|16→128 tets |0.618 mm |0.213 mm |FAIL |

The128-tet state has8.294% peak absolute principal strain,0.967 mm nonrigid RMS and minJ0.9600: it also misses the original excitation window. The coarse pass is not transferred to the finer model. No extrapolation or continuum qualification is made.

## Mechanics and separation diagnostics

At a finite affine deformation near6% strain, energies agree across2/16/128 tets to2.26e−17 J and autograd gradients agree with independent analytic first-Piola assembly to4.93e−15 N. These patch identities pass; they do not establish dynamic mesh accuracy. All physical Gaussian masses are conserved and positive-J guards remain active. Force equilibrium, dissipation and actuator-work bookkeeping are retained in every step ledger.

Exact implicit-stationarity tangent columns for logE/log fixture stiffness propagate through the nominal dynamics. Independent perturbed two-step solves verify those derivatives in tests. Public sparse queries use original nodes1/4 at0.05/0.1/0.15/0.2 s.

| Pattern | Sparse E / fixture derivative norms | Absolute column cosine | Small/large singular-value ratio |
|---|---:|---:|---:|
|legacy_single|2.21/14.02 mm per log-unit|0.047|0.157|
|balanced_axial|12.25/6.55 mm per log-unit|0.468|0.438|
|transverse_couple_2|12.73/31.38 mm per log-unit|0.313|0.378|

Stacking the three public programs gives column cosine0.138 and singular-value ratio0.501. This is promising **coarse discrete sensitivity**, not posterior recovery, calibrated uncertainty or proof of continuum identifiability. No inverse problem or network was run. Mesh error is presently too large to trust these sensitivities as a physical reference.

Four meaningful CPU tests pass in2.99 s: conforming refinement/domain/load preservation plus analytic patch; known-program invariance and rigid-motion removal; implicit material/support derivatives versus independent finite differences; and prospective budgets/confounding diagnostics.

The next necessary step is mesh-resolution diagnosis with a computationally suitable nonlinear solver, followed by temporal validation at the accepted mesh. Additional rates on the unqualified two-tet model alone cannot satisfy the user objective. Any new amplitudes/rates must be declared in a separate protocol; this result and its failures remain immutable.

Evidence: [prospective protocol](protocol.json), [compact review](report.json), [full retained scalar summary](../../gaussian-campaign-v01/nonlinear-excitation-v01/summary.json). Raw states and ledgers remain local. No sealed campaign sources or current cohort values were changed/read.
