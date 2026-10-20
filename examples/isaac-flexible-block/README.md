# Visibly flexible Isaac block

The earlier video mostly showed a stiff body rotating about a point pin.
This demo uses a1kPa material, a20-node fixed patch and three9-node rigid load
patches. Finite load patches spread each applied force over a small material
region; loaded regions remain dynamic. Single-node loading of the softest
material inverted a tetrahedron and was rejected, with the failure state and
nonzero process exit preserved.

[Watch the soft block](soft/simulation.mp4), or the
[matched20kPa/1kPa comparison](comparison/simulation.mp4).
The comparison overlays measured simulation-mesh edges for visibility. Both
panels use the same crop, camera, force history and support/load geometry.
No deformation is amplified and no frames are interpolated. The plain soft
video contains actual Isaac RGB without a mesh overlay.

Peak elastic coordinate RMS is5.745mm for the soft block versus0.782mm for
the matched stiff control, a7.35× increase after removing rigid motion.
The soft run reaches84.7mm maximum displacement and retains positive
tetrahedron volumes (minimum ratio0.274 across all physics steps). Fixture,
load-patch, source, force and clock integrity checks pass. This remains a
development simulator with open timestep/spatial refinement questions, not
a mechanically calibrated material or boundary-recovery result.

```sh
bash experiments/launch_flexible_block.sh --output runs/flexible-new --keep-open

# Matched stiff control: all other settings stay the same.
bash experiments/launch_flexible_block.sh --youngs-modulus 20000 --output runs/stiff-new

.venv/bin/python experiments/validate_isaac_block.py runs/flexible-new
.venv/bin/python experiments/video_isaac_block.py runs/stiff-new runs/flexible-new --grid --output runs/new-comparison
```

Video conversion uses the optional `video` dependencies. Original sparse-pin
settings remain available through `launch_isaac_block.sh`. Full raw data,
source copies, rejected trials and logs are preserved by the archive manifest.

[Permanent raw archive and reassembly checksums](raw-archive.json).
