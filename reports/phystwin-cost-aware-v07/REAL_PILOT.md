# Real-object pilot prepared for acquisition

No real-object measurements have been collected. The machine currently exposes no `/dev/video*`, `/dev/ttyACM*`, or `/dev/ttyUSB*` devices; that does not exclude network-connected hardware. The user has deferred hardware work and requested a simulation-only focus. This document is retained as a future protocol; no acquisition work is pending in this campaign.

Use a soft block, two independently adjustable clamps, calibrated RGB-D imaging, identifiable surface markers, and a measured force. Measure rest dimensions and mass. Record force uncertainty and the point and direction of application; a hand push without measured force cannot isolate material stiffness from load magnitude. A calibrated load cell is useful; measured suspended weights can support a quasi-static first experiment.

Start with six fixture configurations (two repeats of three distinct support layouts). For each, record rest, a weak initial load, six reset push/pull/twist actions, and a distinct oblique held-out load. Include repeated captures at constant load from the initial camera and a second camera pose. Record clamp geometry but withhold its label from inference. Let each action settle, release it, and verify return to rest before the next action. Compare equilibrium holds first; record transients without treating the current static FEM references as validated dynamics.

The controller may inspect only captures already acquired. Freeze its action and stopping settings before evaluating held-out configurations. Compare adaptive versus fixed actions with identical numbers of observed points and optimizer updates. Also compare total capture time and force applications. Keep the final oblique load hidden from fitting and action selection.

For evaluation, report deformation error after removing camera motion, rigid-aligned shape error, raw coordinate error, and uncertainty coverage per object. Keep failures and nearly rigid-motion trials visible. First establish that changing stiffness or supports produces a measurable shape difference above the repeat-capture noise. Only then use uncertainty to select actions.

Save an NPZ with `times_s (T)`, `tracked_points_m (T,N,3)`, `visibility (T,N)` as booleans, `forces_N (T,3)`, `force_application_m (T,3)`, and `camera_to_world (T,4,4)`. All positions are metres, forces newtons, time seconds, and transforms map camera coordinates into the shared world frame. Missing tracks may be NaN only when invisible. Synchronize force and images before export; do not silently interpolate across missing load transitions.

The JSON manifest contains `measured_geometry: true`, `nominal_mass_kg`, positive `force_sigma_N`, and an `episodes` list. Each episode specifies `start_frame`, exclusive `end_frame`, `role` (`fit`, `calibration`, or `held_out`), `object_id`, and `action`. Split by physical fixture/object instance when claiming generalization; disjoint frame ranges alone do not prevent object leakage.

Validate the export with:

```bash
.venv/bin/python scripts/phystwin/v07_pilot_data.py capture.npz manifest.json
```

The validator checks shapes, units encoded by the schema, timestamps, rigid camera transforms, and episode separation. It cannot verify that hardware measurements are accurate. A camera-specific acquisition adapter can be added once the hardware is identified.
