# Qualified reference and probing campaign v05 — in progress

The corrected P2 n=4 simulator bank is complete: 324 physical hypotheses across
four fixture families, each evaluated under four loads. Per-object reference
qualification is running. No calibration, test, or probing-performance result
is claimed at this checkpoint.

The study preserves the v04 displacement, strain, traction, energy and equilibrium
checks, requires two successive spatial passes, and verifies quadrature on every
accepted reference object. Development cases exposed an exact zero-strain
translation case and stiffer-support boundary layers; the explicit addenda record
how these are handled without relaxing the nonzero-deformation thresholds.

FP64 GPU multigrid accelerates the current Newton tangent solves. It uses a
[PyAMG hierarchy](https://pyamg.readthedocs.io/en/latest/generated/pyamg.aggregation.html)
and a GPU V-cycle, with an independent CPU check of the original linear residual.
[Solver validation](solver-validation.json) compares complete nonlinear solutions
against the retained direct-solver results. GPU working-memory pressure was fixed
by streaming quadrature gradients and batching tangent-contraction temporaries.

The pending inference study will qualify its forward surrogate against development
references before fitting the joint discrepancy. It will separately report held
force prediction, raw physical-parameter interval coverage, calibrated prediction
bands, and comparisons against fixed, random and inferred-fixture-side probing.
Fresh calibration and test parameter draws are retained in the frozen protocol.

Scope: static known geometry and forces, unknown stiffness and compliant fixture
category. This remains a numerical reference study, not learned PhysTwin uncertainty
or a demonstration on real RGB-D video. Pooled calibration on a balanced fixture
cohort has nominal coverage; empirical test coverage and object-level uncertainty
will be reported rather than assuming an unconditional finite-sample guarantee.

## Development checkpoint, 27 October 2026

[The numerical diagnosis](NUMERICS.md) records the current development checks and
an independent Q2/P2 comparison. The alternative space agrees with the checked P2
fields but does not resolve the stiff-support refinement failure; it is not used
as campaign truth. The same P2 study now permits n=32/40 while preserving every
accuracy gate. These largest meshes run one at a time to limit host memory.

The finer n=8 hypothesis bank is being generated because the n=4 bank misses the
30 μm development accuracy target. The frozen selection rule determines whether
n=8, a specified extrapolation, or n=12 is acceptable. No held-out outcome is used
for that selection.

The durable runner is `scripts/phystwin/supervise_qualified_campaign.py`. Its local
status is `runs/phystwin-qualified-support-v05/status/supervisor.json`. It resumes
atomic checkpoints, verifies all development references, selects the surrogate,
runs development inference, qualifies/calibrates the new calibration cohort,
then qualifies/evaluates the reserved test cohort. It stops on an unsuccessful
reference or posterior gate. The final report is generated only after completion.
It also records shape error after rigid alignment, raw physical interval coverage,
and a privileged post-test action audit; the latter is not an eligible baseline.

Validation at this checkpoint: 22 geometry, mechanics, GPU, and inference-audit
tests passed, plus two supervisor failure-handling tests. The raw data remain local.
This remains an in-progress campaign, not a completed uncertainty/probing result.

Memory follow-up: these GPUs have 11 GiB each. Linear systems whose actual sparse
hierarchy does not fit use CPU multigrid with batched GPU material assembly. The iterative memory
estimate now budgets for a multigrid hierarchy rather than a direct factorization;
the assembly cap and serialization remain. The equations and residual tolerances
are unchanged. [Execution notes](memory-execution-addendum.json) retain the GPU0
allocation incident and its resolution. The updated full suite passed 25 tests;
an additional memory-budget regression test passed separately (26 total).
GPU hierarchy uploads also now perform a memory admission check using the actual
sparse hierarchy, work vectors and currently free device memory. If it does not
fit, the same hierarchy is used by CPU CG. A forced-fallback test verifies the
original operator across matrix changes; the affected GPU/CPU tests pass (27
unique tests across the checkpoint suites).

## Support-edge candidate validation

Concentrating Q2 hexahedral resolution near the support cut and outer edges is
more effective on the stiff fixture than increasing the original mesh uniformly.
The exploratory n=12-to-16 check passes all four loads; the difficult same-side
push has 0.34% traction change, compared with the 2% gate. This is one passing
refinement, not yet qualification of that fixture.

The formal candidate sequence is n=6,10,14,18,22, with x-power-2 grading, cosine
grading toward y edges and z support edges, and four times as many z cells.
The intermediate n=18 mesh is sized for the available GPUs. It retains two
successive spatial passes and q3/q4 checks under all four loads. The hard object
is still completing this validation.

Development object 11 has qualified at Q2 n=14. Its independent comparison against
the already-qualified P2 n=20 reference passes every gate for every load: worst
coordinate RMS difference 0.00427 mm, traction difference 0.257%, and energy
difference 0.135%. [Cross-space checks](tensor-cross-progress.json) and
[candidate qualification](tensor-development-11.json) retain the evidence.

`scripts/phystwin/validate_tensor_reference.py` cannot enable the method until both
development fixtures qualify and the independent comparison passes. If enabled,
new half-face references will use the edge-refined Q2 mesh directly; full-face
references retain P2 with Q2 fallback after failed qualification. Already-qualified
P2 fields and all earlier failed attempts are preserved. This meshing rule belongs
to the simulator; the true fixture category remains hidden from inference and
probing policies. The hypothesis bank and physical laws are unchanged.

The integrated mechanics, inference, and failure-handling suite passes 40 tests.
GPU Newton solves now release the obsolete fine matrix before tangent assembly,
avoiding overlap between two large allocations. Physical-interval reporting also
includes object-clustered bootstrap uncertainty, so noise repeats are not treated
as independent objects. No held-out probing result is available at this checkpoint.
