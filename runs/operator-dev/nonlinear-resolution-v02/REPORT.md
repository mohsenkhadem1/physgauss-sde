# Sparse nonlinear reference: solver parity passes, resolution remains unqualified

**No nonlinear inverse or network is qualified.** The new sparse solver solves the unchanged finite-strain problem efficiently and agrees with the frozen Torch reference, but both further mesh comparisons fail the predeclared10% development gate. They also fail much stricter noise-scale numerical checks.

## What was changed

A new standalone CPU NumPy/SciPy implementation uses compatible P1 nodal deformation, compressible Neo-Hookean energy, analytic first Piola stress and its exact material tangent. Element12×12 matrices assemble into CSR. The same material-frame normalized Gaussian fixture/load quadrature, world K/D, body mass,1 N axial opposite-force law and0.2 s pulse remain unchanged. No old source, amplitude, target, noise level or cohort value changed.

Sparse LU solves original-objective Newton directions. A declared positive mass/fixture metric can supply descent if needed, without changing physical stiffness; it was never used. There is no diagonal shift, inversion repair or covariance clipping. Exact node-axis constraints use elimination. Positive-J Armijo and the original residual threshold remain required.

Six tests pass in2.69 s: analytic energy/gradient/Hessian and one-step parity against frozen Torch at2/16tets and about8% affine strain; exact moving constraints/reaction and actuator work; positive-J rejection; and distinct development/noise gates. A fresh complete128-tet trajectory agrees with frozen Torch to **4.16×10⁻¹⁷ m** maximum coordinate error.

## Prospective execution and retained outcomes

The independent protocol froze before all three finite calls. CPU2 execution completed in **46.46 s** against a900 s budget, without solver failures.

| Mesh | Vertices |240 Hz trajectory runtime | Peak absolute principal stretch strain | Peak rigid-removed coordinate RMS | MinJ |
|---|---:|---:|---:|---:|---:|
|128tets|55|0.57s|8.29%|0.967mm|0.96001|
|1024tets|285|3.29s|9.91%|0.854mm|0.95633|
|8192tets|1785|42.57s|11.52%|0.800mm|0.95316|

Every case preserves0.3125 kg solid mass, with balanced resultant below4.9×10⁻¹⁶ N. Actual deformed-coordinate moments are recorded and are not claimed constant. The maximum original residual is1.17×10⁻⁹ N at its declared force-relative tolerance; energy/work identity error is below3.34×10⁻¹⁹ J. All accepted states have positive Jacobian.

| Adjacent spatial comparison | Held-node full-trajectory RMS error |10%/2µm tolerance | Query maximum absolute coordinate error |
|---|---:|---:|---:|
|128→1024|381.17µm|231.33µm|1029.45µm|
|1024→8192|380.34µm|243.88µm|980.44µm|

**Both comparisons fail.** The1024→8192 error does not meaningfully contract from the preceding level. The reason remains unresolved; this report does not assert a corner singularity, traction error or solver defect as its cause.

The fixed public query nodes1/4 at0.05/0.1/0.15/0.2 s additionally use maximum-coordinate thresholds **2µm** (0.1×20µm displacement-noise diagnostic) and **20µm** (0.1×200µm existing-depth noise scale, only a coordinate proxy). The finest adjacent query error is980.44µm: both fail. No rendered depth discrepancy is implied. Passing a10% development gate would not by itself establish physical-inference accuracy.

No mesh qualifies, so the predeclared conditional temporal240/480/960 Hz stage and optional1920 Hz continuation were correctly not launched. The earlier2/16/128 mesh failures and60/120/240 temporal failures remain immutable. Peak strain on8192tets is11.52%, while nonrigid RMS is0.800mm, so the original2–8%/1mm excitation target also fails at the finest model. There was no response-based amplitude retuning or preferred-level selection.

The useful result is a fast parity-tested sparse mechanics engine and a clear remaining resolution barrier. The next task must diagnose that barrier or prospectively declare a better spatial loading/geometry design before nonlinear parameter recovery. No continuum, calibration, sensor-inference, general-contact or network-performance claim follows.

Evidence: [frozen protocol](protocol.json), [compact review](report.json), [retained scalar summary](../../gaussian-campaign-v01/nonlinear-resolution-v02/summary.json), and [original excitation failures](../nonlinear-excitation-v01/REPORT.md). Raw trajectories and full step ledgers remain local.
