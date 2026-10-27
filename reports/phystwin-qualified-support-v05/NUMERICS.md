# Development numerical diagnosis

Snapshot: 2026-10-27T17:39:35Z. 10/12 development references qualified; the comparison is not yet complete.

The displacement changes become small sooner than the support-force changes. A small position error therefore does not, by itself, validate the inferred support response. The original energy, strain, traction and equilibrium gates remain in force. Failed objects are retained.

![Development refinement](development-convergence.png)

An independent Q2 hexahedral space uses the same log-J material and consistent surface Robin law. Basis reproduction, analytic affine energy, energy/force/tangent differentiation, surface integration and an exact full-support translation are tested. At n=20, the initial-load Q2/P2 comparisons are:

| Development object | Displacement RMS difference (mm) | Traction relative difference | All spatial cross-checks pass |
|---|---:|---:|---|
| 1 | 0.000999 | 1.659% | True |
| 11 | 0.002197 | 0.082% | True |

This cross-space agreement is useful but does not replace successive-refinement qualification. The stiff object still failed the Q2 n=16-to-20 traction gate; increasing normal-direction grading did not fix it. Q2 remains a diagnostic, not campaign truth. The P2 campaign continues to n=32/40 where needed, serializing these largest meshes to bound memory.

Reproduce the diagnostic with `.venv/bin/python scripts/phystwin/diagnose_tensor_fixture.py --case 1` (or `--case 11`; optionally `--x-power 2`). Raw fields remain under `runs/phystwin-qualified-support-v05/`. The campaign supervisor records its current stage and logs under `status/`, and stops on failed references or inference checks. No calibration/test result is inferred from this development snapshot.
