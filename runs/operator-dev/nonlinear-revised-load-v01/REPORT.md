# Revised central Gaussian tension: clean strain, insufficient bulk motion

**Neither prospectively declared screen passes; no fine-mesh run, nonlinear inverse or network was launched. There is still no qualified Gaussian nonlinear forward.** Both128-tet runs completed in3.50 s against a300 s cap with no solver failure or inversion.

## Frozen new physical design

The new model uses the quality mesh and consistent positive P1 surface integration for both Gaussian load vectors and full fixture Gram matrices. Only load standard deviations broaden by1.5×; material E5000 Pa, density625 kg/m³, Poisson ratio0.35, fixture scale2, fixture Gaussian widths, damping and target fields remain unchanged.

A continuous rest-surface reference computes the actual broadened load centroids before dynamics. The public fixed world axis connects those centroids. Opposite forces now apply **tension** (−axis at patch0,+axis at patch1), not the previous compression. This is a fixed world dead-load law, not a follower force. Peak force remains1 N per patch. Two sin² pulses last0.4/0.8 s at240 Hz.

The screen protocol froze both300 s screening and600 s conditional-fine budgets before values. Eligibility requires completion, positiveJ/original stationarity,2–8% maximum absolute principal stretch strain and≥1 mm best-proper-rigid-removed coordinate RMS. Among eligible screens the closest-to5% strain wins, with a declared64eps strain tie window then shorter duration. No force search or retry is allowed.

| Screen | Steps | Peak absolute strain | Peak nonrigid RMS | MinJ | Result |
|---|---:|---:|---:|---:|---|
|0.4s|96|5.417%|0.628mm|0.998991|FAIL:nonrigidRMS<1mm|
|0.8s|192|5.326%|0.619mm|0.999212|FAIL:nonrigidRMS<1mm|

Rest torque is below7×10⁻¹⁷ Nm. Actual deformed-state moment peaks are0.368/0.297 mNm, recorded rather than assumed zero. Balanced resultant stays below3.2×10⁻¹⁶ N; physical mass remains0.3125 kg. Original residual peaks are2.53×10⁻¹¹/1.58×10⁻¹² N, and work-ledger discrepancy stays below9.1×10⁻²⁰ J. The coherent new mechanics and controls behave consistently, but that does not replace the excitation or accuracy gates.

The automatic selection is **none**. Accordingly, the predeclared1024/8192-tet stage is not admitted and its output directory was not created. Spatial/noise accuracy and timestep refinement remain untested for this revised drive. No prior failed gate is relabeled.

## Why a blind force increase is not the next step

A rough one-point proportional calculation for the0.4 s pulse needs1/0.628≈**1.59 N** to reach1 mm, implying about**8.63%** peak strain. For0.8 s the corresponding estimate is1.61 N and8.60%. These are **not simulated outcomes, not a measured derivative and not a validated nonlinear extrapolation**. They indicate that raising force alone is unlikely to satisfy both frozen targets.

The next useful design should change the ratio of bulk motion to peak local strain through a prospectively declared geometry or traction footprint, for example an elongated block/bar with distributed tractions. The independent analytic static bar implementation checks do not qualify this Gaussian model. Any new candidate budget, numerical gates and controls must freeze separately before solving; no network or inverse fit is justified yet.

Evidence: [prospective protocol](protocol.json), [compact report](report.json), [all retained screen metrics](../../gaussian-campaign-v01/nonlinear-revised-load-v01/screen/summary.json). Raw states and full ledgers remain local. No current cohort values were accessed, and no old pinned source was changed.
