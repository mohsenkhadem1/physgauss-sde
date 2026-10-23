# Resolved target-quantile compilers: manufactured controls

Two NEW opt-in modules copy the frozen public-prefix and typed SIDE54 inference loops. The likelihood, prior, material calculations, moments and numerical comparison gates are unchanged. The prefix compiler uses the already validated exact memo classes. Only the coordinate-quantile call and its resolution metadata change. C38/C39 production modules are untouched.

The new coordinate resolver requires a bracket no wider than min(10 nm, 1e-8 × global polynomial range), together with returned numerical CDF bounds inside p ± 1e-6. Atoms retain their explicit semantics. Unresolved CDF uncertainty still fails; no credible interval is widened.

All six manufactured controls passed in 45.16 seconds (46.22 seconds including process startup/supervision), CPU30/31 with two threads, no GPU, peak RSS 635,887,616 bytes. The owned process exited 0 and was reaped. Both owned PIDs are absent and all 48 pinned source files remain unchanged.

For both prefix and SIDE compilers, every non-quantile posterior/material/moment output was binary64 bit-identical to the frozen implementation at target amplitudes 1e-12 m and 1e-3 m. Cache counters, quantile-dependent endpoint diagnostics and new metadata are explicitly excluded from that identity claim. A prior-loguniform analytical target A×75000/E reproduces the old tiny-scale interval collapse; the new equal-tail quantiles match the independent formula A×2**(p−1), remain distinct and satisfy both numerical criteria. All three methods and 162 GL64 endpoint checks pass. Public-role and typed SIDE barriers and AST identity of the inference loops are also checked.

This is synthetic numerical verification only. No actual campaign observations, arrays, parameter labels, targets or noise draws were read; no actual inference was repeated. No physical uncertainty, population coverage or policy-improvement claim follows from these controls.

Reproduce the bounded control command from protocol.json after checking its source/runtime pins. The files expose infer_public_prefix(candidate, public, *, design) and infer_public_probe(composite, public, SIDEprobe, *, design), respectively. Any later actual-data diagnostic requires a separate versioned execution record and must preserve the original results.
