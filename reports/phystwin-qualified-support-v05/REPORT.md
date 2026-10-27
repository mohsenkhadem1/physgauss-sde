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

Memory follow-up: these GPUs have 11 GiB each. The largest n=40 linear systems use
CPU multigrid while retaining batched GPU material assembly. The iterative memory
estimate now budgets for a multigrid hierarchy rather than a direct factorization;
the assembly cap and serialization remain. The equations and residual tolerances
are unchanged. [Execution notes](memory-execution-addendum.json) retain the GPU0
allocation incident and its resolution. The updated full suite passed 25 tests;
an additional memory-budget regression test passed separately (26 total).
