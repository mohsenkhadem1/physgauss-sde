Qualified local quasistatic push/withdraw example — all frozen gates pass

Setup frozen before execution: 64-node (4x4x4), 100 mm elastic block; Gaussian bottom bilateral support; Gaussian top patch. The supplied plane normal points down. Command pushes down 6 mm, then withdraws to 20 mm above its initial position. Gravity and floor are explicitly disabled. The comparison changes only top attachment type: original bilateral spring versus isolated unilateral plane penalty. CPU runtime: 18.2 seconds.

At maximum push, finest-command unilateral solution:
  Displacement coordinate RMS: 2.202 mm.
  Rigid-motion-removed coordinate RMS: 1.423 mm (about 65% of motion RMS).
  Minimum element Jacobian: 0.9483.
This is appreciable elastic deformation, not simply whole-object translation.

At complete withdrawal:
  Unilateral top force is EXACTLY ZERO; block returns to initial geometry within
  4.15e-9 mm coordinate RMS on the finest run.
  Bilateral top force remains nonzero (nodal force norm 0.06153 in dimensionless
  energy-gradient units), stretching the block upward: 7.354 mm displacement RMS
  and 4.553 mm rigid-removed deformation RMS.
The latter is artificial pulling for a nonadhesive pushing tool; it remains a
valid model for an adhesive/held attachment. Forces are not calibrated Newtons.

All six runs (both laws, 8/16/32 increments in each command phase) completed.
Maximum residual over all runs is 1.42e-9, below the frozen 1e-7 gate; minimum J
is 0.9414, above 0.03. Successive shared-command trajectory RMS differences are
at most 7.74e-12 m, far below the predeclared 0.1 mm gate. Both unilateral zero
separation-force and return-to-rest gates pass. No gate relaxation, case removal
or failed rerun occurred. The preferred Newton stopping target was 1e-9; the
independent declared admission criterion was always 1e-7.

This is COMMAND-STEP refinement for quasistatic equilibrium, not a physical
timestep convergence study. The potential has no inertia, restitution, friction
or impact law. Passing establishes a local supplied-plane mechanics example;
it does not establish inferred contact geometry, unknown material calibration,
mesh convergence, or real-data forecasting improvement. Original bilateral and
frozen eight-case campaign sources/results remain unchanged.

Files: plan.json (frozen source hashes and gates), results.json, individual run
records, trajectories.npz. Source: scripts/gaussian_boundary_flow_v02/unilateral_qualification.py.
