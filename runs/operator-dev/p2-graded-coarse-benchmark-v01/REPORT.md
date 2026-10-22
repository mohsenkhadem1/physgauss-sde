# Fixed-system graded-coarse benchmark

Both preconditioners pass the same physical linear/candidate checks, but the graded coarse model is **slower overall** in this one-correction benchmark. The old uniform 4×2×2 model takes 270 CG iterations and 113.50 s for the linear solve; the new graded 8×4×4 model takes 224 iterations and 94.97 s (16.3% less linear time). Initialization grows from 13.00 to 40.00 s, however, so total case time grows from **129.96 to 138.40 s**, about 6.5% slower. No total speedup is claimed.

The two Keast24 systems use the same immutable graded48 Q3 field, exact physical Hessian and residual, material, load and clamp. Their correction fields differ by at most 1.056e-13 m. Both original linear and candidate nonlinear residuals are approximately 1e-12 N. Candidate fields agree with the independently completed exact80 equilibrium within 9.60e-14 m, with positive whole-element Bernstein bounds and exact clamps. This is two linear solves and two candidate evaluations, not a full nonlinear rerun or a new spatial qualification.

The bounded stage completed in 268.70 s; the controller was reaped with exit zero in 270.89 s. Both owned PIDs were absent, and GPU0 was released. The earlier 2400 s graded64 wall-cap attempt and all failed spatial checks remain. This modest linear improvement does not trigger an automatic full64 retry. Raw corrections remain local.
