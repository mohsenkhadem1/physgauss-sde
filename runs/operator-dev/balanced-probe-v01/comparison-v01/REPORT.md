# Balanced-force elastic excitation diagnostic

All four captures are included: original A/B on previously seen CALIBRATION object 15, and both predeclared balanced amplitudes. Each finite capture completed once with 360 steps/61 frames. The frozen physical core, original inference and reserved evaluation remain unchanged.

| Program | Total coordinate RMS | Residual after best proper rigid fit | Residual/total |
|---|---:|---:|---:|
| Original A |3270.614µm|5.602µm|0.1713%|
| Original B |3004.263µm|5.447µm|0.1813%|
| Balanced .02N/handle |2.839µm|2.669µm|94.00%|
| Balanced .2N/handle |38.013µm|26.724µm|70.30%|

Rigid fits give uniform weight to all 455 nodes/XYZ and are computed separately per recorded frame. The original programs are dominated by bulk motion. At .2N the balanced pair increases nonrigid amplitude 4.77× versus original A while greatly reducing bulk motion. The .02N pair has less absolute nonrigid motion than original A, despite a much greater nonrigid fraction.

The .2N compatible material-Gaussian field has J 0.998444–1.004399, principal stretches 0.992981–1.011766 and reference-volume-weighted principal-strain RMS 0.00071069 (0.0711%). Maximum handle separation change is 458.144 µm versus 18.394/27.213 µm in original A/B. These extrema cover the 61 recorded frames, not unobserved inter-frame extrema. Shape covariance is transported from the same nodal field; it is not epistemic uncertainty. Deformation is still modest. Nonrigid coordinate RMS 26.7 µm is below the stipulated 0.2 mm per-ray noise scale; those different metrics alone do not establish sensor information or trustworthy material inference.

Both balanced programs have EXACT zero applied handle resultant. Using actual step-start/end handle origins, maximum applied moment is 1.032e-7 N m (.02 N) and 7.589e-6 N m (.2 N), versus .005970/.005347 N m for original A/B. The force pair is central at REST only; deformation produces small nonzero moments. This is the applied handle wrench, excluding reactions, drag, inertia and internal forces. No conservation claim follows.

The tool verifies terminal controller/two balanced cells, exact declarations and source/protocol/trajectory/step hashes before opening any capture array. It then opens only these four CALIBRATION 15 captures. Five synthetic tests check proper rigid fitting, compatible affine strain/objectivity, endpoint moment reconstruction, distance changes and the live-controller array-read barrier. No reserved values, simulator calls, fitting or new inference were used.

This is a posthoc excitation development diagnostic on one already seen object. It does not establish prediction improvement, uncertainty calibration, active-policy advantage, temporal/mesh convergence, continuum truth or native Isaac equivalence. All raw captures/steps remain LOCAL.

Reproduce into a fresh directory:

```bash
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 .venv/bin/python \
  experiments/summarize_balanced_probe.py \
  --controller runs/operator-dev/balanced-probe-v01/controller-terminal.json \
  --controller-sha256 3a9f270df2f53b3322711648170cdefd13e9cb78f01a66b5a8d61007d7e694e5 \
  --output runs/operator-dev/balanced-probe-v01/comparison-replay-v01
```
