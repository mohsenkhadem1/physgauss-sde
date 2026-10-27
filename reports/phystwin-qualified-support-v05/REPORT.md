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
