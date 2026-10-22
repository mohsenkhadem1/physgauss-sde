# Graded 32/40 continuation: selected displacement checks

Both declared cases completed in 1096.12 s. All four NEW mesh solves passed the original physical checks. Both selected 32→40 displacement indicators pass 20 µm. The bottom case still failed 24→32; all earlier uniform and graded failures remain. **This does not qualify all 14 cases or admit a dataset.**

| Case |24→32 (µm)|32→40 (µm)|40 sampled peak strain (%)|40 volume99 strain (%)|40 bulk nonrigid RMS (mm)|
|---|---:|---:|---:|---:|---:|
|Elo__separated_bottom_patches__down2|22.564|13.381|61.269|2.419|0.3341|
|Elo__both_ends__down2|9.845|5.504|18.489|3.256|0.7814|

Both new resolutions started from rest. The previous 24 fields supplied only the common 63 material query values; no 24 or coarse solve was repeated. Material, exact support geometry, full consistent gravity/surface loads, original Keast24/Exact80 residual/energy/gradient checks, actual wrench, exact traces and whole-element positive J were unchanged. The indicator assumes order at least one and is not a rigorous error bound.

Peak strain increases with refinement at fixed support edges: bottom 31.57%→46.08%→61.27%; ends 12.08%→15.35%→18.49%. The 99th-volume strains are much smaller, but these growing local maxima preclude a moderate local-strain or converged peak-stress claim. The constitutive law and realistic material validity have not been established by these numerical checks.

The original full both-end graded-pilot report JSON was lost in an unlaunched metadata preparation bug. Its expected SHA and overwritten bytes remain explicit in the incident receipt. All original raw capture hashes matched the preincident compact summary; a NEW read-only six-field evaluation independently reproduced checkpoint and physical metrics. This continuation binds that new provenance and the old compact summary. It never reads or reconstructs the lost full JSON.

No more physics, observations, object draws or training were launched. GPU2 is released. Controller session 26344 reaped exit 0; controller and both workers are absent. All 32 source pins, eight metadata bindings and four new/two reused capture hashes were checked. Full NEW terminal case JSON bytes were additionally cached by SHA outside the run directory; raw captures, checkpoints and caches remain local.

Reproduction requires the LOCAL hash-bound original 24 captures and compact/revalidation evidence. The source/protocol alone do not contain those raw fields.
