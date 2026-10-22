# Axial resolution continuation: retained intermediate failure

The separately frozen 2,592-tetrahedron, 27-point P2 axial solve completed from rest in 16.60 s. Physical loads, material, constraints, and fixed probes match the earlier 6,144-tetrahedron reference. The maximum query discrepancy is 7.9971 µm, below 20 µm, but the prospectively required minimum-first-order fine-error indicator is three times that difference: 23.9912 µm. This fails the unchanged 20 µm criterion. It remains a failure even though the raw adjacent difference passes.

The factor three is the Richardson fine-error multiplier for h-coarse/h-fine = 4/3 under an assumed asymptotic order p = 1; p ≥ 1 makes that choice conservative within the assumed power-law regime. It is an engineering convergence indicator, not a proved continuum-error bound or a proof that this regime has been reached.

The earlier finest 27/64-point quadrature comparison is hash-bound and reused, with maximum query change 0.000123 µm. The original 22.353 µm initial spatial failure and the gravity failure are not overwritten.

The initial continuation source and protocol froze with a 600 s budget. Root's authorization to allow 900 s arrived just after that launch. A separate resource-only amendment was frozen **before this agent read any new intermediate-mesh result**, based on the earlier 85.97/99.69 s finest-mesh runtimes. The live 600 s source was kept unchanged; it correctly retained the optional finer pair as unlaunched because its forecast did not fit that budget. The amendment prospectively permits exactly one 12,000-tetrahedron 27/64-point pair under a new wrapper, the same criteria, two CPU threads, and an external 8 GiB RSS/900 s guard. No additional mesh level or load search is allowed.

See the [protocol](../../operator-dev/static-p2-axial-continuation-v01/protocol.json), [resource amendment](../../operator-dev/static-p2-axial-continuation-v01/budget-amendment-v02.json), and [separate final-pair report](../static-p2-axial-fine-v01/REPORT.md). This intermediate stage does not qualify static prediction, dynamics, native/contact behavior, or inference.
