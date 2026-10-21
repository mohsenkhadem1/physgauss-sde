# Prospective actual-authored-force qualification draft

The completed 60-cell small-force gate is a prerequisite, not an operating-envelope qualification for authored data. Its SHA256 is pinned as `53b4a43c2c86485180ee6786bc4d38b7e8897c9e3dcc737de85f7e7adfa8444d`. All original moving-case 480 Hz failures and broader false qualifications remain unchanged.

`protocol-draft-v01.json` is ready for parent review and is **not launch authorization**. It contains 119 initial physical slots/templates, exact public static/input/source pins, the force laws, staged decisions and bounded conditional-refinement proposals. New runner, authored-family auditor and resolved execution manifests still require review, prospective source freeze, publication and verification. No physical runs were performed for this task, and no live source, core, published gate, proof or checkpoint was changed.

## Exact authored amplitude and family coverage

All four public TRAIN fixtures happen to be authored `push_pulse` cases. Their exact amplitude coefficients are:

| TRAIN case | Boundary | Authored coefficient, N |
|---|---|---:|
| t0000 | Point pin | 0.030249222859091197 |
| t0001 | Fixed patch | 0.1862031455981184 |
| t0002 | Two patches | 0.14946816649260591 |
| t0003 | Moving patch | 0.21153619641676252 |

The point-pin coefficient already includes the generator's factor of .25; it must not be scaled again. The proposed 16 scenarios are four **force-family interventions** on each unchanged public layout/material/support/hardware context. They are not relabelled as 16 original episodes. The generator samples amplitude independently of force family; the intervention changes only the explicitly compiled force family.

Push applies negative reference normal force, pull positive normal force, shear the authored tangent, and the couple equal/opposite authored world-direction forces on the two handles. All directions are fixed in world axes, not follower loads. Both complete rigid handles remain present, including when one force is zero. The coefficient named `peak_force_N` is not necessarily the sampled maximum for every envelope; the shear envelope itself has maximum below one.

The pure compiler `campaign_authored_commands.py`, SHA256 `fdbf333b9531ce11700b6be05ab916fbdb7b30c2fb96cec3762369ae05a3c475`, has no identified blocker. It calls the original `isaac_dataset.forces_at`, preserves its float32 quantization by exact float64 promotion, checks roles/directions/clocks, and records original versus intervened family. No amplitude, dtype or force-application substitution is allowed. Its signed, absolute-impulse and first-moment formulas are consistent with the source law. Pulse/couple signed area is .675 s; pull area is 1.725 s. Shear signed area is zero while its first moment is .24/pi s²; signed impulse alone cannot check its waveform.

Every actual force command must match the promoted compiler values at every step. A pure offline check found no float32 differences between `(i+.5)/hz` and the original batch worker expression `(i+1)/hz-.5/hz` across all 16 interventions and rates 120, 240, 480, 960, 1920 and 3840 Hz. This establishes command parity for these inputs, not native-engine or trajectory equivalence.

For a couple, net force must be exactly zero at every held step. Its moment is evaluated from **actual moving handle origins**; it is not a supplied constant pure torque computed once from reference separation. Record both actual centers and stable center increments, compute moments at stated times/reference frames, and independently reconstruct per-handle work. Supplied pure torques and external nodal/gravity loads remain zero.

## Efficient staged matrix

| Slice | Physical runs/templates | Detail |
|---|---:|---|
| Driven temporal | 48 | 16 scenarios times three rates |
| Moving-zero temporal | 3 | Fresh 20 Hz controls at 240/480/960 Hz |
| Driven accuracy | 32 | Each scenario at relative tolerances 1e-5 and 1e-9, reusing nominal 1e-7 |
| Driven translation | 16 | Each scenario at its passing candidate rate, shifted [0,0,-.25] m |
| Driven fresh reset | 16 | Each scenario at its passing candidate rate |
| Moving-zero effective accuracy | 2 | Floors 1e-8 and 1e-12 N, reusing nominal 1e-10 N |
| Moving-zero translation/reset | 2 | At the adopted moving candidate rate |
| Total initial physical plan | **119** | **51 temporal first, then 68 matched** |

Initial static-BC rates are 120/240/480 Hz; moving rates are 240/480/960 Hz. Every new trajectory contains 61 full frames at 20 Hz over three seconds, matching the authored output cadence. Static zero responses have nine independent analytical rest reference grids; these are not captured trajectories or actual reset trials. Existing 31-frame moving-zero trajectories cannot be silently upsampled, so seven fresh moving-zero temporal/matched slots are included.

Matched rows are templates with an **accepted-candidate-rate selector**, not an unconditional default of 480 or 960 Hz. Only a passing full zero-plus-driven temporal candidate can resolve a matched manifest. The resolved output ID includes its rate, source/evidence hashes and exact settings. A solver-success boolean alone is insufficient proof.

The 119 count assumes one adopted moving-case rate. If different moving force families require different candidate rates, share zero controls only where their rate/settings/output grid/role are identical, and instantiate the declared extra zero controls for each distinct adopted rate. This is factorized coverage, not every possible rate-by-tolerance-by-translation interaction.

## Prospective pass/fail decisions

Temporal errors compare complete 61-frame driven and zero trajectories separately. Only the tolerance denominator uses zero subtraction: `max(2e-6 m, .05 * RMS(q_finest_driven - q_finest_zero))`. The finest adjacent difference must fall below this threshold and contract, except for the existing two-micrometre floor exception. Apply the same family-specific threshold to the matching zero branch; do not hide moving-support error by subtracting it before trajectory comparison.

At each passing candidate rate, compare both adjacent relative-tolerance differences and require the tightest pair to agree. Report effective residual thresholds and absolute-floor dominance. Moving-zero uses meaningful absolute-floor refinement instead of three identical nominal relative tolerances. Translation uses full trajectories after undoing the exact shift. Fresh reset compares all saved q/v/rigid poses/commands with a 1e-12 numeric threshold and reports bitwise equality. The actual reset evidence is distinct from the static analytical zero references.

Every accepted endpoint must be finite, have tet J>=.1, and satisfy the exact fixture/rigid constraints. All nonlinear evaluations require positive J and the frozen core's physical residual acceptance, with a 250-iteration cap and no stiffness repair. Record J, strain/stretch, displacement, speed, rotation and residual extrema. Persist the first failing state, exact clock/force index, nonlinear history and last valid state before terminating a failed cell. No amplitude reduction, dropped frame, material/damping change, relaxed threshold, erased failure or silent restart is allowed.

The declared backend models endpoints and does **not** model self-contact or external contact. This protocol does not add a continuous vertex-chord physical model, arbitrary clearance, stretch-validity threshold or continuous collision algorithm. A tested discrete output-surface crossing diagnostic may report evaluated/not-evaluated and its pair exclusions; sampled absence of crossings cannot establish continuous no-contact or real-material validity.

Conditional refinement is a proposal requiring a prospective exact request and parent authorization, not an automatic launch. Static families may add 960 then 1920 Hz; moving may add 1920 then 3840 Hz. Preserve all initial/coarse failures and every adjacent metric. Only a temporal failure with otherwise valid discrete mechanics triggers this route. A nonlinear cap, line-search or geometry failure is not silently replaced by another rate. Matched checks then occur at the actual passing rate. Exhausting the two extra levels leaves the case failed and requires another prospective decision.

## Clocks and production data admission

The engineering study intentionally ties command sampling/quadrature to solver Hz. Store both exact compiled integer-grid clocks and the actually accumulated solver-state clocks, reconcile them to 1e-12 s, and report both impulse ledgers. Compiler endpoint-difference impulses are compiled-grid evidence, not measured native time. The new authored-family auditor must handle these four force laws and 61 frames; the old small-force auditor hardcodes the .01 N pulse and 31 frames and must remain unchanged.

Per-BC or per-family selected integration rates are **internal numerical settings**, not neural public input. Exposing their `command_boundary_times_s`, adaptive step histories, tolerances, source IDs or provenance fingerprints could reveal a latent BC. Before any learning dataset, prospectively freeze one known actuator-command clock independent of BC/material/layout or inference hypothesis. Refined solver substeps must preserve that externally declared held-command history, or use a common conservatively qualified global rate. Every candidate hypothesis for the same experiment receives the same public command/timing input.

Passing this variable-clock engineering study therefore does not automatically qualify the final neural dataset. A common production command clock and its fixed held-force semantics need separate prospective qualification. Unknown passive forces and unknown support motion also remain latent, consistent with the new complete-hardware context contract.

The fixed reference is the independent log-J Neo-Hookean core with E/nu/rho and body drag. The original native elasticity-damping field .02 remains input lineage and is not implemented or claimed equivalent. No native/continuum, general-amplitude, learned-prediction, calibrated-inference or policy claim follows. These 16 templates also do not guarantee arbitrary new materials/layouts or all amplitudes in .12-.24 N; a new generation/admission plan must explicitly cover its actual cases without selecting thresholds on reserved outcomes.

## Execution preparation and validation

The draft proposes two CPU threads per worker, initially four concurrent workers. Up to eight workers on the 36-logical-core machine requires a separate prelaunch resource declaration after monitoring load/RAM/progress and preserving unrelated COMSOL jobs. No GPU backend is adopted based on speed alone; command/mechanics/trajectory parity remains a separate requirement.

32 tests pass in 2.01 s with two CPU threads: 16 draft-protocol checks plus 16 compiler checks. They inspect only public static inputs and pure force compilation, covering exact 16-scenario/119-slot coverage, staged candidate selectors, coefficient preservation, clocks/frames, original float32 values, oscillatory diagnostics, moving-origin couple moments, analytical-zero versus reset evidence, conditional-rate bounds and the neural-clock leakage restriction. They launch no physics, training, rendering or held-out access.
