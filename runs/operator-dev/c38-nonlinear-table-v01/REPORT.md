# Forty-state nonlinear static table: completed forward checks

All 26 new declared cases completed and passed. The three independent GPU phases finished in 1366.04, 1354.25 and 1235.00 seconds. Their sessions were reaped with exit code zero, all 29 controller/worker PIDs are absent, the external cache contains all 26 terminal case JSONs, and the cache monitor has exited. Each GPU was verified idle at its individual release. Source and protocol hashes remain unchanged. No retries, redraws, additional forces or material values were used.

The complete catalog contains 40 static states: five prescribed Young's moduli across 75–150 kPa, two exact support layouts, and the same known −2, 0, +2 and target +1 N top-band force menu across hypotheses, with known body gravity. Three material values are TABLE_NODE states; the two intervening values remain WITHHELD_INTERPOLATION_VALIDATION. The 14 previous states and their 28 fields were reused exactly; the 26 new states contribute 52 fields. Every new 32/40 graded-mesh field was solved from rest with the original polynomial stable Neo-Hookean energy, full original force/constraint/wrench checks and unchanged tolerances.

At the same 63 material points, all 40 conditional actual-h displacement indicators pass: **1.662–13.381 µm**, below 20 µm. This indicator assumes convergence order at least one and is not a rigorous continuum error bound. Across the 80 fields, original free residuals are at most 9.469e-11 N, below 1e-10 N, and whole-element Jacobian lower bounds are at least 0.457276. The 24/80 exact-polynomial quadrature checks and every required independent original-Torch linear-direction check passed.

Local clamp strain remains unresolved: sampled peak strain spans 4.40–61.27%, while volume-weighted 99th-percentile strain spans 0.612–3.256%. Bulk nonrigid coordinate RMS spans 0.0676–0.7814 mm. A point-displacement pass does not establish converged local stress, moderate peak strain, a ≥1 mm bulk-deformation target, or material realism.

The historical uniform/early graded spatial failures remain evidence. The earlier graded case-JSON loss remains explicitly unrecovered; no old JSON was reconstructed or silently promoted. Exact retained fields, compact preincident metadata and revalidation lineage remain bound in the prerequisite.

**Admission still pending:** this completion seal does not evaluate inverse-modulus table interpolation, generate observations/object roles/noise, train any model, or establish population uncertainty. It does not qualify whole-field/surface-pixel errors, dynamics, contact, images or active policies. The separately frozen interpolation/archive procedure must pass before any new observation stage.

Reproduce the completion metadata without loading trajectory arrays:

```bash
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 .venv/bin/python -m experiments.summarize_c38_nonlinear_table --output runs/operator-dev/c38-nonlinear-table-v01/new-summary.json
```

Raw P2 fields, checkpoints, histories and terminal case-byte caches remain local. The compact summary retains their exact source/hash records.
