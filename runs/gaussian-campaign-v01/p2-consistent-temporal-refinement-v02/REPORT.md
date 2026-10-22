# Consistent P2 smooth-ramp temporal continuation

The six-tet smooth-ramp fixed-discretization temporal check passes. Across rates 240, 480, 960, 1920 and 3840 Hz, full-trajectory maximum Euclidean differences at all 25 common times and 122 probes are **99.7218, 29.5702, 7.73121 and 1.96926 um**. The finest difference is below the unchanged 20 um criterion and the full rate sequence contracts. Earlier coarse failures remain recorded.

The prospectively conditional 96-tet ramp at 1920 Hz also completed. Its 480-to-1920 difference is **40.2011 um**, failing the same criterion. There is no successive-level contraction evidence for that mesh and it is not temporally qualified. The earlier unchanged 96-tet ramp at 960 Hz was not retried after its original wall timeout. Abrupt-step failures are not replaced by these ramp results.

All three new runs use the identical E=5000 Pa, nu=0.35, rho=625 kg/m3 block, exact left clamp, normalized 1 N right axial load, full analytic consistent T10 mass, stable Neo-Hookean law and BE-started BDF2 equations. The ramp is sin²((pi/2)*min(t/0.05,1)), applied at implicit right endpoints over 0.1 s. The frozen original objective tolerance is 1e-10 N; physical momentum tolerance is 1e-9 N. Maximum physical residual across the new runs is 8.6321e-11 N, sampled J exceeds 0.9885, and clamp displacement/velocity are exactly zero.

Six meaningful core/orchestration tests passed in 1.90 s. A first preflight attempt incorrectly JSON-parsed a binary input after hashing; it failed before model construction or any trajectory. That source/protocol/failure is preserved. The separate v02 source fixes only binary hash validation and adds a real NPZ tamper regression. It changes no physical inputs or equations.

The conditional larger run was authorized by the frozen rule: the six-tet temporal check passed and its 233.19 s cost forecast plus 5 s fit within the remaining 300 s budget. All three runs finished in 255.31 s with two CPU threads and peak RSS 588,685,312 bytes. The worker/watchdog are absent and session 12681 was reaped with exit 0. All source and complete capture hashes were reverified.

This is small fixed-mesh temporal evidence. It does not qualify spatial resolution, other force histories, contact, production datasets or inference. [Protocol](../../operator-dev/p2-consistent-temporal-refinement-v02/protocol.json), [results](summary.json), [terminal receipt](../../operator-dev/p2-consistent-temporal-refinement-v02/terminal.json).
