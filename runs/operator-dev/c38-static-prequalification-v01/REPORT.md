# Supported static family: terminal qualification check

All 14 declared cases completed in 774.48 s. All 42 resolution solves and 14 independent coarse CPU/GPU controls passed the physical checks. Seven cases passed the unchanged 20 µm spatial indicator; seven failed. **The full family is not qualified and no dataset is admitted.**

The model uses polynomial stable Neo-Hookean material, E = 75–150 kPa, density 625 kg/m³, nu = .35, gravity 9.81 m/s², full consistent body loads and a fixed public top-midspan Fz. All solves started from rest. The checks retained the original Keast24/Exact80 residual, energy and gradient, exact supported traces, actual force and moment balance, and whole-element positive J.

| Case |20→24 indicator (µm)| Gate |Peak strain (%)|Bulk nonrigid RMS (mm)|
|---|---:|---|---:|---:|
|interior-gravity__both_ends|13.420|PASS|4.273|0.4111|
|interior-gravity__separated_bottom_patches|37.922|FAIL|5.141|0.1586|
|Elo__both_ends__down2|24.436|FAIL|7.660|0.7791|
|Elo__both_ends__gravity|19.010|PASS|6.060|0.5812|
|Elo__both_ends__up2|14.117|PASS|4.478|0.3850|
|Elo__separated_bottom_patches__down2|73.838|FAIL|9.706|0.3228|
|Elo__separated_bottom_patches__gravity|54.023|FAIL|7.250|0.2251|
|Elo__separated_bottom_patches__up2|34.777|FAIL|4.801|0.1307|
|Ehi__both_ends__down2|12.125|PASS|3.804|0.3893|
|Ehi__both_ends__gravity|9.477|PASS|3.016|0.2907|
|Ehi__both_ends__up2|7.080|PASS|2.233|0.1928|
|Ehi__separated_bottom_patches__down2|36.274|FAIL|4.879|0.1601|
|Ehi__separated_bottom_patches__gravity|26.680|FAIL|3.642|0.1119|
|Ehi__separated_bottom_patches__up2|17.256|PASS|2.410|0.0651|

## Refinement planning only

The worst completed corner is `Elo__separated_bottom_patches__down2`: 73.838 µm. A fit to its two max-differences gives apparent order 1.054; this extrapolates to about 43.2 µm for 32→40, so common uniform 32/40 is unlikely to qualify all corners. These are descriptive fits, not error bounds: maxima can occur at different points and only two pairs exist.

Measured mesh 20/24 times for that case were 17.39/34.80 s. An empirical two-point cost fit projects roughly 104 s at 32 and 243 s at 40 for that case. No such runs have been launched.

A source-only fixed boundary-graded candidate preserves the x = .06/.18 patch edges, physical loads and supports, and true affine T10 reference geometry. It has seven pure CPU controls but no mechanical qualification. A separately frozen worst-bottom plus both-end control pilot is the next prospective test. All original failed comparisons, 63 query points and the 20 µm gate remain.

Exit 0 means bounded completion, not that all scientific gates passed. Static dead-load checks do not establish dynamics, contact, imaging, inference or training validity. Raw captures and checkpoints remain local. The terminal proof binds 14 result hashes, 42 capture hashes, all 30 unchanged runtime source pins and absent controller/worker PIDs.
