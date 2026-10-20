# Isaac Sim boundary-learning block

Actual GPU FEM recordings from Isaac Sim6.2, using a seeded random surface pin
and three random Newton-valued force pulses. The four-node fixture variant,
second seed, unloaded control and half-time-step case are recorded separately.
See [setup](../../docs/ISAAC_BOUNDARY_BLOCK.md), [results](results.json),
[actual RGB animation](preview.gif) and [diagnostic overview](overview.png).

All five final explicit-clock cases pass integrity replay, and three geometry/
force tests pass. Sleep and artificial settling damping are disabled. This is
forward simulation only; boundary inference is not implemented yet.

**Numerical convergence remains open.** The240/480Hz paths differ by15.08mm
coordinate RMS and134.13mm maximum node separation, largely through rotation
about a point pin. Peak elastic coordinate RMS changes0.717→0.674mm. Do not
use this as a mechanically calibrated boundary/force benchmark yet. Earlier
app-clock/default-sleeping runs are explicitly superseded development evidence.
Raw scenes, source snapshots, trajectories, Gaussians, RGB and logs are preserved
on the artifact branch, with reassembly and hash manifests.

[Permanent raw archive and checksums](raw-archive.json).
