# Independent dynamics mechanics and qualification review

Read-only mechanics review plus a separate evidence auditor. No native or independent simulation jobs were launched by this review, no reserved observations were opened, and the physical core, live runner and published gate were not modified.

## Reviewed revisions and scope

- Core `experiments/campaign_nonlinear_dynamics.py`: SHA256 `da0db7018b4913020bb228c2fc4df3c240e7dd872f17db4868de147eaaf78fa2`.
- Baseline runner `experiments/campaign_nonlinear_dynamics_case.py`: SHA256 `f062f0db2c2ac7db00a5897c846b2f588f7394cf92895b28a808c7aa4223b65d`.
- Published independent dynamic gate: SHA256 `962457955d72a97dd0d8e48a41aff0c6ead821931031343e081cf837c561b2a0`.
- Public static TRAIN fixture: SHA256 `073380f72559c50ccc674d5f69fdff1be7e4d0b30484ad66f0233695a813847f`.

No blocking mechanics issue was found for source publication. This is an independent float64, fixed-mesh log-J Neo-Hookean model, not established native Isaac equivalence or continuum truth. Its actual qualification remains false.

## Mechanics findings

The finite handle pose map is exact: every patch point shares one SO(3) rotation and translation; auxiliary rigid mass points reproduce the authored mass, COM and full inertia. Moving and fixed fixture positions are imposed exactly. Physical inertial mass makes the dynamic metric positive definite even for a point pin with static rotational null modes. No arbitrary diagonal stiffness repair is introduced.

The incremental potential includes body and handle inertia, current nonlinear elasticity, body-only viscous drag and held external force work. The stopping residual converts rotational generalized gradients to physical world torques and then to Newtons using a physical length; force normalization does not mix Newtons with raw Newton-metres. The state retains reduced translations and computes stable pose increments, avoiding subtraction of large absolute positions for tiny inertial increments. Elasticity still evaluates world positions: extremely small force tests below position roundoff do not establish accurate elastic response at that scale.

The integrator retains finite pose-map point chords as velocities. Those are not instantaneous SO(3) tangent velocities. Energy and angular momentum can have finite-step defects; the source reports these rather than claiming exact conservation. The cached first-state mass/stiffness metric affects only preconditioning. A positive-J line search and physical residual acceptance still govern the full nonlinear step.

The fixture reaction includes inertia, elastic force and drag. Its work is reaction dotted with prescribed displacement. The momentum ledger now includes body drag. The energy algebra includes the curved-constraint chord work and implicit velocity-increment loss; an algebraically small identity residual is bookkeeping, not proof of energy conservation. Baseline ledgers save each force but do not save every handle-center increment, so external handle work cannot be reconstructed independently from baseline trajectories sampled at 10 Hz. The auditor states this limitation.

## Test independence and limits

The saved mechanics prerequisite has 31 passing tests with frozen source/XML hashes. It includes rigid mass identities, exact finite rotation maps, nonlinear energy/gradient and linear-tangent checks, an independently stepped linear recurrence, moving-reaction and damped translation balances, nonzero COM offsets, and an asymmetric free-spin refinement. The latter reports decreasing relative energy/angular errors at steps .04/.02/.01 s. The linear recurrence shares spatial assembly with the model; these tests are not an independently implemented nonlinear continuum reference.

Large-rotation patch tests establish rigid objectivity. They do not prove global nonlinear convergence for every force/fixture combination, prevent all local minima, or qualify arbitrary point-pin rotations. The TRAIN matrix therefore retains actual point-pin, fixed-patch, two-patch and moving-patch trajectories, positive-volume checks and nonlinear histories. Optional 960/1920 Hz levels are in the original gate but the frozen baseline runner currently accepts only 120/240/480 Hz; a reviewed, separately published runner revision is needed before such new runs.

## Minimum approved factorized evidence

The approved matrix is 48 baseline runs plus two separately predeclared moving-zero accuracy runs:

| Slice | Cells | Settings |
|---|---:|---|
| Temporal | 24 | Four BCs, zero and diagonal response, 120/240/480 Hz, relative tolerance 1e-7 |
| Nonlinear tolerance | 8 additional | Four diagonal cases at 480 Hz and relative tolerance 1e-5/1e-9, reusing 1e-7 |
| Coordinate translation | 8 | Both responses and all BCs at 480 Hz, shift [0,0,-.25] m |
| Fresh reset replay | 8 | Both responses and all BCs at 480 Hz |
| Effective zero-force accuracy | 2 additional | Moving-zero 480 Hz at absolute residual floors 1e-8/1e-12 N, reusing the baseline 1e-10 N |

The absolute floor is normally 1e-10 N. Zero external force makes all nominal relative tolerances identical, including moving-fixture motion; duplicate nominal-zero runs would not demonstrate accuracy refinement. Static zero cases instead have an analytical exact-rest check. The moving-zero supplemental protocol has SHA256 `ab4d663af320b1e1863d07239b473982b340992b2b6e6ff7554c9a9350fc358e`.

The separate auditor checks full protocol/source/input hashes, exact canonical static arrays/hardware, all step and frame clocks, held commands and analytic impulse, accepted nonlinear residuals and positive volumes, saved-frame finite constraints/strain, every runtime constraint/J record, fixture work and momentum/energy bookkeeping. It requires 50 distinct indexed output slots, 29 comparison results and the mechanics prerequisite. Missing, incomplete, failed or invalid evidence cannot pass. A supplemental runner revision must be explicitly whitelisted and preserve every physical dependency; its saved helper and floor protocol hashes are also required.

Temporal metrics use complete zero and diagonal trajectories separately. Zero subtraction is used only for the force-induced signal in `max(2 micrometres, 5% finest force-induced coordinate RMS)`. Reset replay uses a separate 1e-12 numerical tolerance and reports bitwise equality. This is a factorized design: it does not test every time/accuracy/translation interaction.

## Initial temporal result retained

All 24 completed temporal cells passed protocol/hash/clock/force/residual/constraint/positive-J verification. Coordinate RMSEs below compare full trajectories, 120 to 240 Hz and 240 to 480 Hz:

| TRAIN boundary | Zero RMSE, micrometres | Diagonal RMSE, micrometres | Tolerance, micrometres | Initial temporal decision |
|---|---:|---:|---:|---|
| Point pin | 0 / 0 | 2.315 / 1.162 | 119.005 | Pass |
| Fixed patch | 0 / 0 | 3.890 / 2.159 | 3.781 | Pass |
| Two patches | 0 / 0 | .237 / .193 | 2.000 | Pass |
| Moving patch | 16.520 / 8.474 | 16.082 / 8.262 | 6.507 | **Fail both responses** |

The moving results contract but the current finest difference exceeds the predeclared threshold. This failure must remain visible if the optional 960 Hz level is attempted. Any extension should list all four actual rates and all adjacent errors, keep the initial three-level failure, and label the remaining tolerance/translation/replay/absolute-floor checks as 480 Hz evidence. They would not establish those interactions at 960 Hz.

## Runner extension review

Reviewed draft `runs/operator-dev/floor-runner-extension-draft/campaign_nonlinear_dynamics_case.py`, SHA256 `723243dcd9904906805aaa86280ed862784baa8c4eb818082dc758af60340b59`, and helper `experiments/campaign_dynamic_floor_protocol.py`, SHA256 `ed6dd2b56ddf8ce9bf0f511392046173220a887eb5da19832545279508215164`.

No blocker was found. The guard pins the published supplemental protocol, core, gate and canonical fixture, and admits only t0003/zero/480 Hz/relative1e-7/original coordinates/cap250 at the two new absolute floors. It rejects regenerating the middle baseline as new evidence. The draft saves step-start/end centers and the stable reduced-coordinate increment used by core force work; the auditor can independently reconstruct held-force work when those optional fields are present. This draft must not replace the live runner before the baseline processes terminate, and the new revision/helper must be published before either supplemental run.

## Operating envelope and next qualification

Even a completed passing matrix would qualify only the declared mesh, material/support laws, initial rest state, zero response and the specified [0.01,0.01,0.005] N handle0 pulse. It does not qualify original recipe peaks of 0.12–0.24 N, other force profiles, larger active probes, bulk original-amplitude capture, native conservation, material inference, uncertainty calibration or policy improvement. A prospective actual-force operating-envelope refinement is still required before general capture.

Separate auditor tests use an analytical exact-rest persisted dataset without calling the dynamic stepper, plus rehashed corruption cases and full-response cancellation traps. They test the audit itself, not new physical accuracy.
