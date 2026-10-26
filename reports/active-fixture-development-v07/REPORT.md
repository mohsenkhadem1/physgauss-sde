# Active fixture development v07: comparison blocked by runtime admission

The nonlinear mechanics reference completed successfully, but the full noisy one-poke comparison did not run. The unchanged runtime gate stopped the operator after its first evaluation checkpoint. No adaptive-versus-fixed/random or nonlinear-versus-linear performance scores are reported from that single trial.

All **264 nonlinear fields** were retained: eight prescribed support/material setups, eleven loads and three meshes. The existing independent saved-field audit passed all **88/88 load–setup reference checks**. Its largest n6→n12 public-query coordinate difference was **8.540 µm**, within the frozen 20 µm gate. Maximum sampled symmetric strain was **3.861%**, and minimum sampled Jacobian was **0.974359**. These are static, sampled/numerical checks; conditional refinement indicators are not rigorous continuum bounds.

Excitation remains separate from physical/reference validity: **264/264** fields met the unchanged 100 µm nonrigid-RMS and 0.3 nonrigid-fraction goal; **264/264** met the legacy load-eligibility predicate, which includes its original weak-prefix exception. Every field and action was retained.

All **32 continuous choices** and **64 matched finite choices** were sealed before their mapped outcomes could be used. The evaluation checkpoint retained one trial and all eight fresh prefix-plus-one-poke posteriors; all eight were marked qualified by the producing numerical procedure. Their full independent terminal audit was not run.

The slowest checkpoint posterior took **49.321 seconds**. The predeclared conservative projection was:

`3 × 49.320520 s × 8 actions × 11 trials = 13020.617 s`

That exceeds the frozen **10,800-second** worker budget by **2220.617 seconds**. This is a runtime-admission failure, not a failed checkpoint posterior. The projection is an estimate with its frozen safety factor, not measured runtime for the unfinished queue. The remaining **248 continuous** one-poke outcomes and all **512 matched-finite** posterior updates were not evaluated. The two full terminal audits and reviewed result plot therefore remain blocked.

An additional closed-metadata check found that the frozen finite-choice verifier rejects **1/64** producer-qualified seals: forty level-prior witnesses differ from their top-level prior by at most **2.0679515313825692e-25 m²** (1.72e-16 relative). Every risk is finite and nonnegative. The mismatch is consistent with floating-point normalization roundoff at an exact-equality assertion; it has not been silently accepted, corrected or labelled a quadrature failure. All frozen sources and original seals remain unchanged.

The finite nonlinear/linear pair uses the same restricted uniform eight-state prior with known fixture stiffness. The continuous model has a different prior with unknown stiffness; a future comparison must preserve that distinction. Before/adaptive/frozen-fixed-5/random error and uncertainty tables, and physics-versus-discretization oracle metrics, are withheld until the complete paired comparison is admitted. No score is substituted from the completed checkpoint.

All eight executed stage receipts report exit 0, completed/reaped children, and absent owned PIDs/process groups. The outer operator is absent without its completion marker. Root retains ownership of unified-execution session 69047. No simulation, posterior, acquisition utility, noise draw or NPZ array was replayed for this report. Raw fields, models, observations, noise and media remain local. This fixed development design supports no population/SBC, general neural-model, contact or dynamic-forecast claim.

Compact scalar evidence is in [summary.json](summary.json), with exact source/closure/result commitments in [provenance.json](provenance.json).
