# Completed finite14 static displacement check

All 14 declared engineering configurations pass the unchanged conditional actual-h displacement indicator at the same 63 reference material points. This joins 12 new cases (24 new graded32/40 fields) with the exact two previously completed cases (four reused fields). The new phase took 1,881.10 s; session75768 was reaped with exit0. Controller and all12 workers are absent, GPU2 is idle, all37 source pins and all28 capture hashes match, and the complete new case JSON bytes are cached by SHA.

The numerical acceleration changes HVP evaluation only: all101 returned Warp-CG directions pass independent original Torch checks, and the four reused fields retain20 original Torch-CG residual checks. All28 fields pass the original24/80 equilibrium, energy/gradient agreement, force/moment balance, exact-constraint and whole-element positive-J checks. Early inexact Newton-CG residuals use the original relative/absolute linear bound; the final static residual remains1e-10 N.

| Case | E (Pa) | Support | Fz (N) |32→40 q63 indicator (µm)|
|---|---:|---|---:|---:|
|interior-gravity__both_ends|106066|both_ends|0|2.968411|
|interior-gravity__separated_bottom_patches|106066|separated_bottom_patches|0|6.640206|
|Elo__both_ends__down2|75000|both_ends|-2|5.503593|
|Elo__both_ends__gravity|75000|both_ends|0|4.221536|
|Elo__both_ends__up2|75000|both_ends|2|3.299659|
|Elo__separated_bottom_patches__down2|75000|separated_bottom_patches|-2|13.380541|
|Elo__separated_bottom_patches__gravity|75000|separated_bottom_patches|0|9.646750|
|Elo__separated_bottom_patches__up2|75000|separated_bottom_patches|2|6.223692|
|Ehi__both_ends__down2|150000|both_ends|-2|2.712426|
|Ehi__both_ends__gravity|150000|both_ends|0|2.090316|
|Ehi__both_ends__up2|150000|both_ends|2|1.662448|
|Ehi__separated_bottom_patches__down2|150000|separated_bottom_patches|-2|6.250962|
|Ehi__separated_bottom_patches__gravity|150000|separated_bottom_patches|0|4.617898|
|Ehi__separated_bottom_patches__up2|150000|separated_bottom_patches|2|3.035035|

All finest indicators lie between1.66245 and13.38054 µm, below20 µm. These are finite engineering-point comparisons, not a uniform certificate over continuous E, an error bound over all nodal/surface positions, dense RGB-D qualification, or dataset admission. No observations, population roles or training were created.

Finest sampled peak principal strains span5.258–61.269%; the volume99th-percentile strains span0.6124–3.2565%, volume-RMS strains0.1480–1.0532%, and bulk nonrigid coordinate RMS0.06771–0.78140 mm. Whole-element J lower bounds span0.45728–0.96036. Growing clamp/patch-edge peaks remain unresolved: this does not qualify converged peak stress, moderate local strain or material realism.

Historical negatives remain intact: the original uniform matrix has seven failed finest spatial comparisons; both graded12/20/24 pilots failed; and the bottom24→32 comparison remains22.5644 µm>20 µm. The incident that overwrote one earlier full graded-pilot JSON remains explicit. Its original full bytes were not recovered. Original accepted captures, preincident compact summary, independently recorded integrity proof and separately derived read-only revalidation remain the lineage; no reconstructed report is presented as an original.

[Compact exact14 evidence](completed-summary.json) · [Terminal/hash/PID proof](terminal-proof.json) · [Completed-byte cache receipt](completed-case-cache-receipt.json). All raw fields/checkpoints remain LOCAL. A separate supplied-physics linear prediction stage may run under its own frozen protocol; nonlinear comparison or static dataset admission needs its separate reviewed scope.
