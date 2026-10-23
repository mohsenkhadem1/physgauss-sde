# C39 SIDE static q63 reference: ten states completed

All ten declared lateral-load states completed their fresh graded 32/40 solves and passed the original physical and fixed 63-point spatial gates. The five materials are the exact unchanged C38 tuples: 75, 89.1905, 106.0660, 126.1345 and 150 kPa, crossed with both-end clamps and separated bottom patches. Three material nodes are table nodes and two are withheld interpolation-validation nodes; these are numerical uses, not independently sampled population roles.

The new action is the same public 2 N world +Y traction on the conforming top-middle material band, with rest moment (−0.08, 0, 0) Nm about the block center. Gravity, density 625 kg/m³, ν = 0.35, stable Neo-Hookean law, exact supports, Keast24 bulk integration, independent exact80 checks, original full HVP/CG acceptance, and from-rest initialization were preserved. The force compiler never receives material/support labels. Thirty existing C38 gravity/down/canonical states are referenced unchanged, giving a potential 40-state composite; no existing physical solve was repeated.

The independent linear screening preceded these nonlinear solves. Both graded 32/40 linear screens met the fixed q63 criterion and showed support-dependent local material Fisher winners: downward for both ends, lateral for separated patches. This motivated the new action space; it is not an active-policy result.

| Material | Both-end q63 indicator | Bottom-patch q63 indicator |
|---|---:|---:|
| 75 kPa | 6.169 µm | 17.962 µm |
| 89.1905 kPa | 5.180 µm | 14.614 µm |
| 106.0660 kPa | 4.350 µm | 12.000 µm |
| 126.1345 kPa | 3.654 µm | 9.915 µm |
| 150 kPa | 3.069 µm | 8.226 µm |

All indicators are below the unchanged 20 µm threshold. They use actual mesh-edge ratios and an assumed spatial order p ≥ 1; they are conditional engineering indicators, not rigorous continuum bounds or a full-field accuracy certificate.

Across all 20 fields, maximum original free residual24/80 was 8.31×10⁻¹¹ N, force/moment balance defects were 9.97×10⁻¹¹ N/1.93×10⁻¹¹ Nm, and maximum 24/80 gradient/energy differences were 1.45×10⁻¹³ N/6.94×10⁻¹⁸ J. Exact masks and every independent original-Torch linear-direction check passed. All whole-element Bernstein positivity checks passed. The lowest sampled/whole bound was approximately 0.10836, just above the sampled 0.1 guard. Maximum local principal stretch-strain reached 90.44% near constrained edges: this is explicitly **not** a moderate-local-strain, converged-stress, physical-material calibration, contact, or image-rendering certificate. No self-contact law or general global-injectivity certificate is supplied.

The owned three-GPU controller finished in 833.23 s. All ten mechanics workers exited normally with code 0; all 20 capture hashes and ten exact case-JSON caches were checked. No physical failure, retry, filtering, material/load change, or observation/noise draw occurred. Raw nodal/q63 arrays remain LOCAL; the public compact summary contains scalar checks and exact capture references only. Maximum reported Torch allocation was 1.095 GiB, within the guarded 8 GiB combined GPU and 12 GiB RSS budgets.

Two software issues are preserved openly. Before any GPU call, root review caught a missing `validate_cpu` import; unexecuted v01 source/test/protocol bytes were retained and v02 passed a new recursive startup-name control (3 CPU tests). During execution, parent signal masking around Popen was inherited by workers/watchdogs, so completed workers waited for a blocked daemon termination. A separately frozen, root-approved resource-only helper verified ownership, exact source/protocol/capture hashes and strict physical completion before killing **only** each daemon watchdog. All mechanics workers then exited 0 without any solver/field/source change. The first helper's disappearing-/proc exit race is preserved; v02 handled it and finalized all ten cases. Future controllers must restore child signal masks explicitly. See resource-cleanup-incident.json and the terminal receipt.

The next admission step is separately approved point-only export and the fixed 1 µm withheld interpolation check. Until that succeeds, these direct static fields do not admit a composite inverse/observation provider. Even after numerical interpolation checks, active utility, calibrated uncertainty, new noisy observations, RGB-D, unknown arbitrary supports/material fields and causal dynamics require their own evidence.
