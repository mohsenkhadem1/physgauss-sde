The old cantilever has a substantial unresolved timestep effect, but the inspected evidence does not identify a force-callback or material-setting bug. The 240→480 Hz maximum difference is 15.514 mm; 480→960 Hz is 5.645 mm. Both maxima occur at the final 0.6 s frame. The 20 µm gate remains failed.

The source and captured native checks establish these points:

- This case has gravity only: no handles, external-force callbacks, floor, or contacts. Gravity is set and read back once. A missing force replay cannot explain this experiment.
- Native rest points and tetrahedra match the supplied mesh; initial positions and zero velocities are checked after reset. Fixed-node flags are checked through both high-level and raw tensor APIs, and fixed drift is zero.
- E=300000 Pa and ν≈0.35 are read back after reset. Material density625 kg/m³ is authored before cooking. No nonzero body-mass override is authored; the installed schema says the default zero mass is ignored and material density determines mass. Native nodal masses are not exposed in the inspected DeformableBodyView API, so actual masses remain unmeasured.
- Linear, elasticity, and settling damping are explicitly zero; sleeping/settling thresholds are zero. Changing damping to make curves match would change the physical problem.
- The actual final clock is identical at all three rates:0.6000000312924385 s. The discrepancy is not a mismatched saved-frame clock.

The freefall control distinguishes an integration convention from an elastic failure. Velocity error is only1.71×10⁻⁶ m/s. Position differs from exact continuous ballistics by1.384 mm coordinate RMS, but matches the discrete kick-drift reference to9.53 nm. That is direct evidence of first-order position integration in this control, rather than evidence that gravity or material inference is broken.

The cantilever is still moving rapidly at the final frame. Mean tip vertical displacements are−72.741/−58.799/−53.749 mm at240/480/960 Hz, with velocities+0.389/+0.580/+0.656 m/s. A timestep-dependent transient phase/amplitude error is plausible. This is not a static-equilibrium comparison and does not separate temporal integration error from incomplete iterative solves.

The installed PhysxSchemaAddition describes `physxDeformableBody:solverPositionIterationCount` as iterations per timestep, range1–255. Current32 iterations have not been independently refined and no physical residual is available. The scene's PGS token is authored, but this alone does not establish which internal deformable constitutive/constraint algorithm ran. Installed schema documentation describes `enableExternalForcesEveryIteration` as improving **TGS** stability; toggling it for this gravity-only PGS study is not an evidence-based fix. Native nonlinear constitutive equivalence to the CPU Neo-Hookean model remains unestablished.

A controlled four-cell test was prospectively declared and completed: fresh GPU1 captures at 240/480 Hz ×32/128 deformable iterations, with identical geometry, gravity, boundaries, E/ν/density, damping and PGS setting.

| Controlled comparison | Maximum material-query difference |
|---|---:|
| 32→128 iterations at 240 Hz | 14.5732 mm |
| 32→128 iterations at 480 Hz | 0.4365 mm |
| 240→480 Hz at 32 iterations | 15.5143 mm |
| 240→480 Hz at 128 iterations | 5.3675 mm |

Increasing iterations reduces the measured timestep discrepancy by 65.4%, establishing substantial iteration error in the old timestep study. All four differences still exceed the unchanged 20 µm gate. The 128-iteration result is not a qualified fix: remaining iterative and integration errors are not fully separated, and no solver residual or continuum reference is available. No extra cells or changed physical parameters were used.

The first ablation attempt was rejected **before any captured physical step** by the strict own-process GPU1-only guard. Its frozen source/protocol/logs and failed terminal receipt are retained; three remaining cells were not launched. The requested all-GPU visibility plus physical renderer1/CUDA1 settings did not establish exclusive use of GPU1. A separately frozen resource-only v02 uses physical GPU1's UUID for CUDA_VISIBLE_DEVICES, `active_cuda_gpus=[0]`, physics GPU0 and World CUDA0 within that restricted visibility. This follows installed Isaac6.2 documentation and avoids Vulkan/CUDA numeric-index ambiguity. All four own-process checks identified only physical GPU1; records were persisted before each guard.

Two further implementation limits were identified. SimulationApp defaults to16 CPU threads and overrides inherited OpenBLAS limits; v02 explicitly sets `limit_cpu_threads=2`, verified in Kit/TBB startup arguments. Fresh GPU1 controls at 32 iterations reproduce the old GPU0 trajectories **bit for bit at both rates**, establishing that these resource changes did not alter this captured response. Also World.reset documents one internal step. The worker restores and verifies positions/velocities and sets fixtures afterward, but does not observe any hidden warm-start state. This is a possible initialization confound, not an established cause; the ablation keeps initialization unchanged.

The four v02 applications completed in 73.53 seconds, within the 300-second global and 100-second per-app caps. Controller and all worker PIDs are absent, all source hashes remain unchanged, and the session was reaped with exit0. Execution success does not mean the accuracy gate passed. See the [compact v02 results](../isaac-solver-ablation-v02/result-summary.json), [terminal receipt](../isaac-solver-ablation-v02/terminal-receipt.json), and [preserved v01 resource failure](../isaac-solver-ablation-v01/failed-terminal.json).

The companion evidence JSON binds installed primary SDK source hashes and original campaign source/results. No old source or result was edited. No mass, damping, force, material or solver-family retuning is justified by this inspection and controlled ablation.
